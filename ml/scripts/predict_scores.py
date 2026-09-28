#!/usr/bin/env python3
"""학습된 검출기로 split 하나를 돌려 평가 CLI 용 예측 CSV 를 만든다 (#23).

    python3 ml/scripts/predict_scores.py <가중치.pt> calib
    python3 ml/scripts/predict_scores.py <가중치.pt> test_hidden --out ml/runs/pred/hidden.csv

출력은 `image_id,item,y_true,score` 이고 사진 × 품목마다 한 행이다.
점수는 품목별 박스 신뢰도의 최댓값, 박스가 없으면 0 (E1 기준선).
정답(y_true)은 data/yolo/labels 의 라벨에서 읽는다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hancut.config import load_items  # noqa: E402
from hancut.data import yolo  # noqa: E402
from hancut.eval import scores  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "yolo"
SPLITS = ("val", "calib", "test_easy", "test_hard", "test_hidden")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", type=Path)
    parser.add_argument("split", choices=SPLITS)
    parser.add_argument("--out", type=Path)
    # 하한을 높이면 낮은 점수가 0 으로 뭉쳐 low 임계값을 그 아래로 내릴 수 없다
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from ultralytics import YOLO  # 인스턴스에서만 필요하다. 테스트는 이 스크립트를 부르지 않는다

    names = yolo.class_names(load_items())
    model = YOLO(str(args.weights))
    model_names = [model.names[i] for i in sorted(model.names)]
    if model_names != names:
        raise SystemExit(f"가중치의 클래스 순서가 items.json 과 다르다.\n  가중치: {model_names}\n  설정:   {names}")

    img_dir, lbl_dir = DATA / "images" / args.split, DATA / "labels" / args.split
    images = sorted(img_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"{img_dir} 에 이미지가 없다. make_yolo_dataset.py 를 먼저 돌린다")
    out = args.out or REPO / "ml" / "runs" / "predictions" / f"{args.split}.csv"

    print(f"{args.split}: {len(images):,}장, 신뢰도 하한 {args.conf}")
    start, rows, seen = time.time(), [], set()
    for r in model.predict(source=str(img_dir), stream=True, conf=args.conf, imgsz=args.imgsz,
                           batch=args.batch, device=args.device, verbose=False):
        stem = Path(r.path).stem
        dets = zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist())
        present = scores.read_yolo_classes(lbl_dir / f"{stem}.txt")
        rows.extend(scores.rows_for_image(stem, dets, present, names))
        seen.add(stem)

    unseen = {p.stem for p in images} - seen
    if unseen:
        raise SystemExit(f"{len(unseen):,}장이 예측에서 빠졌다. 예: {sorted(unseen)[:3]}")

    n = scores.write_predictions(sorted(rows, key=lambda r: (r["image_id"], r["item"])), out)
    took = time.time() - start
    print(f"`{out}` — {n:,}행 ({len(seen):,}장 × {len(names)}종), {took:.0f}초 ({len(seen) / took:.1f}장/초)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
