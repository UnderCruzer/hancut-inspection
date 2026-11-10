#!/usr/bin/env python3
"""물체 크기 구간별 재현율 (#54) — 같은 품목 안에서 작은 물체일수록 더 놓치는가.

    python3 ml/scripts/size_recall.py --weights ml/runs/detect/e1_smoke/weights/best.pt --name E1 --out-dir $OUT/sizes_e1
    python3 ml/scripts/size_recall.py --weights $SMALL --imgsz 960 --items Lighter Bullet --name small --out-dir $OUT/sizes_small

정답 박스마다 신뢰도 --conf 이상 · IoU 0.5 이상인 같은 품목 예측이 있으면 맞힌 것으로 센다.
크기는 640 입력 기준 박스 한 변(px)이라, 입력 크기가 다른 모델도 같은 구간으로 비교한다.
재현율은 그 신뢰도 하나에서의 값이다. 운영점(F)의 놓침률이 아니다. 이미지는 어디에도 올리지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hancut.config import load_items  # noqa: E402
from hancut.data import yolo  # noqa: E402
from hancut.eval import misses, sizes  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "yolo"
SPLITS = ("test_easy", "test_hard", "test_hidden")


def read_labels(path: Path) -> list[tuple[int, tuple[float, ...]]]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            c, *xywh = line.split()
            out.append((int(c), tuple(float(v) for v in xywh)))
    return out


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def infer(weights: Path, paths: list[Path], conf: float, imgsz: int, batch: int, device: str) -> dict[str, list]:
    from ultralytics import YOLO  # 인스턴스에서만

    model = YOLO(str(weights))
    out = {}
    for start in range(0, len(paths), batch * 4):
        chunk = [str(p) for p in paths[start:start + batch * 4]]
        for r in model.predict(source=chunk, stream=True, conf=conf, imgsz=imgsz, batch=batch, device=device, verbose=False):
            out[Path(r.path).stem] = [
                misses.Pred(int(c), float(cf), tuple(float(v) for v in b))
                for c, cf, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xywhn.tolist())
            ]
    return out


def collect(data: Path, splits, keep: set[int], names, preds_for, conf: float, edges) -> list[tuple[str, str, str, bool]]:
    """(분할, 품목, 크기 구간, 맞힘) 목록. preds_for(분할, 사진 경로들) → {stem: [Pred]}."""
    records = []
    for split in splits:
        labels = sorted((data / "labels" / split).glob("*.txt"))
        images = {p.stem: data / "images" / split / f"{p.stem}.png" for p in labels}
        preds = preds_for(split, [images[p.stem] for p in labels])
        for path in labels:
            gts = [(c, b) for c, b in read_labels(path) if c in keep]
            if not gts:
                continue
            width, height = image_size(images[path.stem])
            hits = sizes.match(gts, preds.get(path.stem, []), conf)
            for (c, b), ok in zip(gts, hits):
                records.append((split, names[c], sizes.bin_of(sizes.side_px(b, width, height), edges), ok))
    return records


def table_md(title: str, table: dict, items, edges) -> list[str]:
    bins = sizes.bin_names(edges)
    lines = [f"## {title}", "", "| 품목 | " + " | ".join(f"{b}px" for b in bins) + " | 전체 |", "|---|" + "---:|" * (len(bins) + 1)]
    for item in items:
        cells = table.get(item, {})
        row = []
        for b in bins + ["전체"]:
            h, n = (sum(v[0] for v in cells.values()), sum(v[1] for v in cells.values())) if b == "전체" else cells.get(b, (0, 0))
            row.append(f"{h / n:.1%} ({n:,})" if n else "-")
        lines.append(f"| {item} | " + " | ".join(row) + " |")
    return lines + [""]


def report(records, items, name: str, weights: Path, imgsz: int, conf: float, edges) -> str:
    lines = ["# 크기 구간별 재현율 (#54)", "",
             f"모델 `{name}` · `{weights}` · 입력 {imgsz} · 신뢰도 {conf:g} 이상 · IoU 0.5. "
             "칸은 재현율 (정답 박스 수). 크기는 640 입력 기준 박스 한 변.", ""]
    hidden = sizes.recall_table((i, b, ok) for s, i, b, ok in records if s == "test_hidden")
    every = sizes.recall_table((i, b, ok) for s, i, b, ok in records)
    lines += table_md("숨김 (test_hidden)", hidden, items, edges)
    lines += table_md("시험 전체 (easy · hard · hidden)", every, items, edges)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--items", nargs="+", help="기본은 12종 전부")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--edges", type=float, nargs="+", default=list(sizes.DEFAULT_EDGES))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()

    names = yolo.class_names(load_items())
    items = args.items or names
    unknown = sorted(set(items) - set(names))
    if unknown:
        raise SystemExit(f"모르는 품목: {unknown}")
    sizes.bin_names(args.edges)  # 경계가 잘못됐으면 추론 전에 멈춘다
    keep = {names.index(i) for i in items}

    preds_for = lambda split, paths: infer(args.weights, paths, args.conf, args.imgsz, args.batch, args.device)  # noqa: E731
    records = collect(args.data, args.splits, keep, names, preds_for, args.conf, args.edges)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "records.json").write_text(json.dumps(
        {"name": args.name, "weights": str(args.weights), "imgsz": args.imgsz, "conf": args.conf, "edges": args.edges,
         "records": records}, ensure_ascii=False) + "\n", encoding="utf-8")
    text = report(records, items, args.name, args.weights, args.imgsz, args.conf, args.edges)
    (args.out_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text + f"\n저장: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
