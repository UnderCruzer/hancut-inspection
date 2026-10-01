#!/usr/bin/env python3
"""미검출 분석 (#31) — 품목이 있는데 검출기가 그 품목 박스를 내지 않은 경우, 그 자리에서 무엇을 했나.

    python3 ml/scripts/analyze_misses.py

1. 예측 CSV 에서 미검출 이미지(y=1, score=0)를 고른다
2. 그 이미지만 신뢰도 하한을 0.0001 로 낮춰 다시 추론한다 (원래 하한은 0.001)
3. 정답 박스마다 가장 많이 겹치는 예측으로 분류한다 — 착각 / 점수 부족 / 위치 어긋남 / 아무것도 없음
4. 정답 박스 모양을 (학습) / (시험·잡음) / (시험·놓침) 으로 비교한다

이미지는 어디에도 올리지 않는다. 결과는 숫자뿐이다.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hancut.config import load_items  # noqa: E402
from hancut.data import yolo  # noqa: E402
from hancut.eval import cli, misses, resplit  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "yolo"
DIFFICULTIES = ("easy", "hard", "hidden")
DEFAULT_ITEMS = ("Gun", "Knife", "Sprayer", "Bullet")  # p₀ 가 5% 를 넘은 품목 (#25)
ORIGINAL_FLOOR = 0.001


def read_labels(path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            c, *xywh = line.split()
            out.append((int(c), tuple(float(v) for v in xywh)))
    return out


def collect(pred_dir: Path, items):
    rows = [r for d in DIFFICULTIES for r in cli.read_predictions(pred_dir / f"test_{d}.csv")]
    missed = {i: [] for i in items}
    detected = {i: [] for i in items}
    positives = Counter()
    for r in rows:
        if r["item"] in missed and r["y_true"] == 1:
            positives[r["item"]] += 1
            if r["score"] == 0.0:
                missed[r["item"]].append(r["image_id"])
            elif r["score"] >= 0.5:
                detected[r["item"]].append(r["image_id"])
    return rows, missed, detected, positives


def image_path(stem: str) -> Path:
    return DATA / "images" / f"test_{resplit.difficulty_of(stem)}" / f"{stem}.png"


def label_path(stem: str) -> Path:
    d = resplit.difficulty_of(stem)
    return DATA / "labels" / (f"test_{d}" if d else "train") / f"{stem}.txt"


def infer(weights: Path, stems, conf: float, device: str) -> dict[str, list]:
    from ultralytics import YOLO  # 인스턴스에서만

    model = YOLO(str(weights))
    out = {}
    paths = [str(image_path(s)) for s in sorted(stems)]
    for start in range(0, len(paths), 64):
        for r in model.predict(source=paths[start:start + 64], stream=True, conf=conf, imgsz=640,
                               device=device, verbose=False):
            out[Path(r.path).stem] = [
                misses.Pred(int(c), float(cf), tuple(float(v) for v in b))
                for c, cf, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xywhn.tolist())
            ]
    return out


def pct(n, d) -> str:
    return f"{n / d:.1%}" if d else "-"


def shape_row(label: str, s: dict) -> str:
    if not s["n"]:
        return f"| {label} | 0 | - | - |"
    a, r = s["area"], s["aspect"]
    return (f"| {label} | {s['n']:,} | {a[1]:.2%} ({a[0]:.2%}~{a[2]:.2%}) | "
            f"{r[1]:.2f} ({r[0]:.2f}~{r[2]:.2f}) |")


def build_report(items, names, rows, missed, detected, positives, preds, train_labels_dir: Path) -> str:
    idx = {n: i for i, n in enumerate(names)}
    lines = ["# 미검출 분석 (#31)", "",
             f"미검출 = 품목이 있는데 그 품목 박스 점수가 0 (원래 신뢰도 하한 {ORIGINAL_FLOOR}). "
             "미검출 이미지만 하한을 낮춰 다시 추론해, 정답 박스 자리에서 검출기가 한 일을 분류했다.", ""]

    train_boxes = {i: [] for i in items}
    for f in sorted(train_labels_dir.glob("*.txt")):
        for c, b in read_labels(f):
            name = names[c] if c < len(names) else None
            if name in train_boxes:
                train_boxes[name].append(b)

    lines += ["## 요약", "", "| 품목 | 양성 | 미검출 | 착각 | 점수 부족 | 위치 어긋남 | 아무것도 없음 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    detail = []
    for item in items:
        target = idx[item]
        cats, wrong_as, per_diff = Counter(), Counter(), Counter()
        missed_boxes = []
        for stem in missed[item]:
            per_diff[resplit.difficulty_of(stem)] += 1
            gts = [b for c, b in read_labels(label_path(stem)) if c == target]
            missed_boxes += gts
            for gt in gts:
                try:
                    cat, p, _ = misses.classify_miss(gt, target, preds.get(stem, []), floor=ORIGINAL_FLOOR)
                except ValueError:
                    cat, p = "재추론에서는 검출", None
                cats[cat] += 1
                if cat == misses.WRONG_CLASS and p is not None:
                    wrong_as[names[p.cls]] += 1
        n_box = sum(cats.values())
        lines.append(f"| {item} | {positives[item]:,} | {len(missed[item]):,} ({pct(len(missed[item]), positives[item])}) | "
                     + " | ".join(f"{pct(cats[c], n_box)}" for c in misses.CATEGORIES) + " |")

        detected_boxes = [b for stem in detected[item] for c, b in read_labels(label_path(stem)) if c == target]
        alt, _ = misses.top_alternatives(rows, item)
        detail += [f"## {item}", "",
                   f"미검출 {len(missed[item]):,}장 — easy {per_diff['easy']} · hard {per_diff['hard']} · hidden {per_diff['hidden']}. "
                   f"분류는 정답 박스 {n_box:,}개 기준.", ""]
        if wrong_as:
            detail.append("착각한 품목: " + ", ".join(f"{k} {v}" for k, v in wrong_as.most_common(5)))
        if cats.get("재추론에서는 검출"):
            detail.append(f"재추론에서는 검출: {cats['재추론에서는 검출']} (배치·하한 차이로 보임)")
        detail.append("대신 점수가 가장 높았던 품목(≥0.25, CSV 기준): "
                      + (", ".join(f"{k} {v}" for k, v in list(alt.items())[:5]) or "-"))
        detail += ["", "| 정답 박스 | 개수 | 면적 비율 중앙값 (사분위) | 가로/세로 중앙값 (사분위) |", "|---|---:|---:|---:|",
                   shape_row("학습", misses.shape_stats(train_boxes[item])),
                   shape_row("시험 · 잡음 (score ≥ 0.5)", misses.shape_stats(detected_boxes)),
                   shape_row("시험 · 놓침", misses.shape_stats(missed_boxes)), ""]
    return "\n".join(lines + [""] + detail) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=REPO / "ml/runs/detect/e1_smoke/weights/best.pt")
    parser.add_argument("--items", nargs="+", default=list(DEFAULT_ITEMS))
    parser.add_argument("--pred-dir", type=Path, default=REPO / "ml/runs/predictions")
    parser.add_argument("--conf", type=float, default=0.0001, help="재추론 신뢰도 하한 (원래 0.001 보다 낮게)")
    parser.add_argument("--device", default="0")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    names = yolo.class_names(load_items())
    unknown = set(args.items) - set(names)
    if unknown:
        print(f"모르는 품목: {sorted(unknown)}. 가능한 이름: {names}")
        return 2
    for need in [args.weights] + [args.pred_dir / f"test_{d}.csv" for d in DIFFICULTIES]:
        if not need.is_file():
            print(f"없다: {need}")
            return 2

    rows, missed, detected, positives = collect(args.pred_dir, args.items)
    stems = {s for v in missed.values() for s in v}
    print(f"미검출 이미지 {len(stems):,}장을 신뢰도 하한 {args.conf} 로 다시 추론한다...")
    start = time.time()
    preds = infer(args.weights, stems, args.conf, args.device)
    print(f"{time.time() - start:.0f}초\n")

    report = build_report(args.items, names, rows, missed, detected, positives, preds, DATA / "labels" / "train")
    out = args.out_dir or REPO / "ml/runs/miss_analysis" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"저장: {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
