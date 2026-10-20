#!/usr/bin/env python3
"""NMS 전 품목 확률로 이미지 점수를 뽑는다 — 기존 점수와 같은 forward 에서 함께 (#43).

    python3 ml/scripts/predict_dense_scores.py ml/runs/detect/e1_smoke/weights/best.pt \\
        --out-dir ml/runs/dense_20261019/e1 --compare-dir ml/runs/predictions

출력:
    <out-dir>/nms/<split>.csv     기존 규칙 — NMS 뒤 그 품목 박스 신뢰도 최댓값, 없으면 0
    <out-dir>/dense/<split>.csv   새 규칙 — NMS 전 8,400개 앵커의 그 품목 확률 최댓값
    <out-dir>/summary.md          품목별 p₀(점수 0 인 양성 비율), dense ≥ nms 검사, 기존 CSV 와의 일치율

두 폴더 모두 e2_compare.py --pred-dir 로 바로 평가한다. 같은 가중치 · 전처리 · forward 라 점수 규칙만 다르다.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hancut.config import load_items  # noqa: E402
from hancut.data import yolo  # noqa: E402
from hancut.eval import cli, dense, scores  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "yolo"
SPLITS = ("calib", "test_easy", "test_hard", "test_hidden")
TEST_SPLITS = SPLITS[1:]


def p0_table(nms_rows, dense_rows, names) -> list[str]:
    zn, zd = dense.zero_rate_by_item(nms_rows), dense.zero_rate_by_item(dense_rows)
    out = ["| 품목 | 양성 | p₀ nms | p₀ dense |", "|---|---:|---:|---:|"]
    for n in names:
        a, tot = zn.get(n, (0, 0))
        b, _ = zd.get(n, (0, 0))
        out.append(f"| {n} | {tot:,} | {a} ({a / tot:.1%}) | {b} ({b / tot:.1%}) |" if tot else f"| {n} | 0 | - | - |")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--compare-dir", type=Path, help="기존 predict_scores.py 출력 폴더 — nms 재현 확인용")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.001, help="nms 쪽 신뢰도 하한 (predict_scores.py 와 같게)")
    parser.add_argument("--iou", type=float, default=0.7, help="nms IoU (ultralytics predict 기본값)")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:  # 8.3 이하
        from ultralytics.utils.ops import non_max_suppression

    names = yolo.class_names(load_items())
    model = YOLO(str(args.weights))
    if [model.names[i] for i in sorted(model.names)] != names:
        raise SystemExit("가중치의 클래스 순서가 items.json 과 다르다")
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    net = model.model.to(device).eval()
    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)

    def load(path: Path):
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"읽지 못했다: {path}")
        img = letterbox(image=img)
        return np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))  # BGR → RGB, HWC → CHW

    rows = {"nms": {}, "dense": {}}
    pool = ThreadPoolExecutor(max_workers=8)
    for split in args.splits:
        images = sorted((DATA / "images" / split).glob("*.png"))
        if not images:
            raise SystemExit(f"{DATA / 'images' / split} 에 이미지가 없다")
        lbl_dir = DATA / "labels" / split
        start, out_n, out_d = time.time(), [], []
        for i in range(0, len(images), args.batch):
            chunk = images[i:i + args.batch]
            x = torch.from_numpy(np.stack(list(pool.map(load, chunk)))).to(device).float() / 255
            with torch.no_grad():
                raw = net(x)
            raw = raw[0] if isinstance(raw, (list, tuple)) else raw  # (B, 4 + nc, 앵커)
            dense_max = raw[:, 4:4 + len(names), :].amax(dim=2).clamp(0, 1).cpu().tolist()
            kept = non_max_suppression(raw, conf_thres=args.conf, iou_thres=args.iou, max_det=300, nc=len(names))
            for path, dmax, det in zip(chunk, dense_max, kept):
                present = scores.read_yolo_classes(lbl_dir / f"{path.stem}.txt")
                dets = [(int(c), float(cf)) for cf, c in det[:, 4:6].cpu().tolist()]
                out_n += scores.rows_for_image(path.stem, dets, present, names)
                out_d += scores.rows_for_image(path.stem, list(enumerate(dmax)), present, names)
        rows["nms"][split], rows["dense"][split] = out_n, out_d
        scores.write_predictions(out_n, args.out_dir / "nms" / f"{split}.csv")
        scores.write_predictions(out_d, args.out_dir / "dense" / f"{split}.csv", fmt=".9g")
        print(f"{split}: {len(images):,}장, {time.time() - start:.0f}초")

    # 검사와 요약 — 저장된 CSV 를 다시 읽어 쓴 그대로를 검사한다
    reread = {k: {s: cli.read_predictions(args.out_dir / k / f"{s}.csv") for s in args.splits} for k in rows}
    lines = ["# NMS 전 품목 확률 점수 (#43)", "", f"가중치: `{args.weights}` · imgsz {args.imgsz} (정사각 letterbox) · "
             f"nms conf {args.conf} · IoU {args.iou}", "", "## 검사", ""]
    bad = sum(len(dense.dominance_violations(reread["dense"][s], reread["nms"][s])) for s in args.splits)
    lines.append(f"- dense < nms 인 행: **{bad}** (0 이어야 한다)")
    if args.compare_dir:
        for s in args.splits:
            ref = args.compare_dir / f"{s}.csv"
            if ref.is_file():
                same, total = dense.agreement(reread["nms"][s], cli.read_predictions(ref))
                lines.append(f"- {s}: nms 와 기존 CSV 의 점수 차 ≤ 0.001 인 행 {same:,}/{total:,} ({same / total:.2%})")
    tests = [s for s in args.splits if s in TEST_SPLITS]
    if tests:
        n_rows = [r for s in tests for r in reread["nms"][s]]
        d_rows = [r for s in tests for r in reread["dense"][s]]
        lines += ["", f"## 품목별 p₀ — 시험 {', '.join(tests)} 전체", ""] + p0_table(n_rows, d_rows, names)
    report = "\n".join(lines) + "\n"
    (args.out_dir / "summary.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
