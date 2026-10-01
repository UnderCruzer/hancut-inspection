#!/usr/bin/env python3
"""헷갈리는 품목이 든 학습 사진을 여러 번 보여주는 학습셋 (#33).

    python3 ml/scripts/make_oversampled_train.py                       # 총기·칼·가위 3번
    python3 ml/scripts/make_oversampled_train.py --repeat Knife=3 Scissors=2
    python3 ml/scripts/make_oversampled_train.py --overwrite           # 이미 있으면 다시 만든다

make_yolo_dataset.py 가 만든 data/yolo 의 train 을 읽어 옆에 train_os 를 만든다.
반복은 이름이 다른 링크로 건다 (xray_00123.png, xray_00123__r1.png, ...). 로더의 중복 처리에 기대지 않는다.
val · calib · 시험셋은 건드리지 않고, pidray_os.yaml 은 val 을 원래 것 그대로 가리킨다.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hancut.config import load_items  # noqa: E402
from hancut.data import yolo  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "yolo"
DEFAULT_REPEAT = ("Gun=3", "Knife=3", "Scissors=3")


def parse_repeat(specs, names: list[str]) -> dict[int, int]:
    out = {}
    for spec in specs:
        name, _, n = spec.partition("=")
        if name not in names or not n.isdigit():
            raise SystemExit(f"'{spec}' 를 읽지 못했다. 형식은 품목=횟수, 품목은 {names}")
        out[names.index(name)] = int(n)
    return out


def read_classes(path: Path) -> list[int]:
    return [int(line.split()[0]) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", nargs="+", default=list(DEFAULT_REPEAT), help="품목=횟수")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--name", default="train_os")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    names = yolo.class_names(load_items())
    repeat = parse_repeat(args.repeat, names)
    if not args.name.startswith("train_"):
        # 원래 split 을 지우지 않도록 이름으로 막는다
        raise SystemExit(f"--name 은 train_ 으로 시작해야 한다: {args.name}")

    src_img, src_lbl = args.data / "images" / "train", args.data / "labels" / "train"
    labels = sorted(src_lbl.glob("*.txt"))
    if not labels:
        raise SystemExit(f"{src_lbl} 에 라벨이 없다. make_yolo_dataset.py 를 먼저 돌린다")

    out_img, out_lbl = args.data / "images" / args.name, args.data / "labels" / args.name
    if out_img.exists() or out_lbl.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_img} 가 이미 있다. 다시 만들려면 --overwrite")
        shutil.rmtree(out_img, ignore_errors=True)
        shutil.rmtree(out_lbl, ignore_errors=True)
        (args.data / "labels" / f"{args.name}.cache").unlink(missing_ok=True)
    out_img.mkdir(parents=True)
    out_lbl.mkdir(parents=True)

    classes = {p.stem: read_classes(p) for p in labels}
    counts = yolo.repeat_counts(classes, repeat)
    before, after = Counter(), Counter()
    for stem, n in counts.items():
        src = (src_img / f"{stem}.png").resolve()
        if not src.is_file():
            raise SystemExit(f"이미지가 없다: {src_img / stem}.png")
        text = (src_lbl / f"{stem}.txt").read_text()
        for k in range(n):
            name = stem if k == 0 else f"{stem}__r{k}"
            os.symlink(src, out_img / f"{name}.png")
            (out_lbl / f"{name}.txt").write_text(text)
        for c in set(classes[stem]):
            before[c] += 1
            after[c] += n

    total = sum(counts.values())
    print(f"# 반복 학습셋 (#33)\n\n`{out_img}` — 사진 {len(counts):,} → {total:,}장 ({total / len(counts):.2f}배)\n")
    print("| 품목 | 반복 | 사진 전 | 사진 후 |\n|---|---:|---:|---:|")
    for i, n in enumerate(names):
        print(f"| {n} | {repeat.get(i, 1)} | {before[i]:,} | {after[i]:,} |")

    yaml = [f"path: {args.data.resolve()}", f"train: images/{args.name}", "val: images/val",
            "test: images/test_hidden", "names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    out_yaml = args.data / f"pidray_{args.name.removeprefix('train_')}.yaml"
    out_yaml.write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"\n`{out_yaml}` 를 썼다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
