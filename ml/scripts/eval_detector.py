#!/usr/bin/env python3
"""검출기 세부 성능 — val 과 시험 세 난이도에서 전체 · 품목별 (#35).

    python3 ml/scripts/eval_detector.py
    python3 ml/scripts/eval_detector.py --weights ml/runs/detect/os/weights/best.pt

split 마다 ultralytics 의 val 을 돌려 정밀도 · 재현율 · mAP50 · mAP50-95 를 표로 낸다.
결과는 ml/runs/eval_detector/<시각>/summary.md (숫자만, 이미지 없음).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hancut.config import load_items  # noqa: E402
from hancut.data import yolo  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "yolo"
SPLITS = ("val", "test_easy", "test_hard", "test_hidden")
LABELS = {"val": "val (학습 분포)", "test_easy": "시험 · 쉬움", "test_hard": "시험 · 겹침", "test_hidden": "시험 · 숨김"}
METRICS = ("p", "r", "map50", "map")
HEAD = {"p": "정밀도", "r": "재현율", "map50": "mAP50", "map": "mAP50-95"}


def split_yaml(path: Path, split: str, names: list[str]) -> Path:
    """split 하나를 val 로 가리키는 데이터 설정. ultralytics 는 train 키도 요구한다."""
    lines = [f"path: {DATA.resolve()}", "train: images/train", f"val: images/{split}", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def overall_table(results: dict[str, dict]) -> list[str]:
    out = ["| 평가 사진 | 사진 | " + " | ".join(HEAD[m] for m in METRICS) + " |", "|---|---:|" + "---:|" * len(METRICS)]
    for split, r in results.items():
        out.append(f"| {LABELS[split]} | {r['images']:,} | " + " | ".join(f"{r['all'][m]:.3f}" for m in METRICS) + " |")
    return out


def per_class_table(results: dict[str, dict], names: list[str], metric: str) -> list[str]:
    splits = list(results)
    out = ["| 품목 | " + " | ".join(LABELS[s] for s in splits) + " |", "|---|" + "---:|" * len(splits)]
    for c, n in enumerate(names):
        cells = []
        for s in splits:
            v = results[s]["per_class"].get(c, {}).get(metric)
            cells.append("-" if v is None else f"{v:.3f}")
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    return out


def build_report(weights: str, results: dict[str, dict], names: list[str], imgsz: int = 640) -> str:
    lines = ["# 검출기 세부 성능 (#35)", "", f"가중치: `{weights}` · imgsz {imgsz} · IoU 0.5 기준 매칭", "",
             "## 전체 (12종 평균)", ""] + overall_table(results)
    for m in ("map50", "r", "p", "map"):
        lines += ["", f"## 품목별 {HEAD[m]}", ""] + per_class_table(results, names, m)
    return "\n".join(lines) + "\n"


def evaluate(model, yaml: Path, device: str, batch: int, imgsz: int = 640) -> dict:
    m = model.val(data=str(yaml), split="val", imgsz=imgsz, batch=batch, device=device, plots=False, verbose=False)
    box = m.box
    per = {}
    for i, c in enumerate(m.ap_class_index):
        per[int(c)] = {"p": float(box.p[i]), "r": float(box.r[i]), "map50": float(box.ap50[i]), "map": float(box.ap[i])}
    return {"all": {"p": float(box.mp), "r": float(box.mr), "map50": float(box.map50), "map": float(box.map)}, "per_class": per}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=REPO / "ml/runs/detect/e1_smoke/weights/best.pt")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.imgsz <= 0:
        parser.error("--imgsz 는 양수여야 한다")

    if not args.weights.is_file():
        print(f"없다: {args.weights}")
        return 2
    from ultralytics import YOLO  # 인스턴스에서만

    names = yolo.class_names(load_items())
    out = args.out_dir or REPO / "ml/runs/eval_detector" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    results = {}
    for split in args.splits:
        images = len(list((DATA / "images" / split).glob("*.png")))
        print(f"{split}: {images:,}장 평가 중...")
        results[split] = evaluate(model, split_yaml(out / f"{split}.yaml", split, names), args.device, args.batch, args.imgsz)
        results[split]["images"] = images

    report = build_report(str(args.weights.relative_to(REPO) if args.weights.is_relative_to(REPO) else args.weights), results, names, args.imgsz)
    (out / "summary.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"저장: {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
