#!/usr/bin/env python3
"""고른 품목의 라벨만 남긴 학습셋 (#54).

    python3 ml/scripts/make_item_subset.py --items Lighter Bullet            # train_small · val_small
    python3 ml/scripts/make_item_subset.py --items Lighter Bullet --overwrite

make_yolo_dataset.py 가 만든 data/yolo 의 train · val 옆에 `<split>_<suffix>` 를 만든다. 사진은 원본에 링크를 걸고,
라벨은 고른 품목 줄만 남긴다. 다른 물건만 든 사진도 빈 라벨로 남겨 배경으로 배우게 한다.
12종 이름과 번호는 그대로라 점수 추출 · 평가 스크립트를 바꾸지 않고 쓴다.
"""
from __future__ import annotations

import argparse
import json
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
SPLITS = ("train", "val")


def keep_lines(text: str, keep: set[int]) -> list[str]:
    """라벨 파일 내용에서 고른 품목 줄만. 줄은 고치지 않고 그대로 둔다."""
    return [line for line in text.splitlines() if line.strip() and int(line.split()[0]) in keep]


def build_split(data: Path, split: str, suffix: str, keep: set[int], overwrite: bool) -> dict:
    src_img, src_lbl = data / "images" / split, data / "labels" / split
    labels = sorted(src_lbl.glob("*.txt"))
    if not labels:
        raise SystemExit(f"{src_lbl} 에 라벨이 없다. make_yolo_dataset.py 를 먼저 돌린다")
    name = f"{split}_{suffix}"
    out_img, out_lbl = data / "images" / name, data / "labels" / name
    if out_img.exists() or out_lbl.exists():
        if not overwrite:
            raise SystemExit(f"{out_img} 가 이미 있다. 다시 만들려면 --overwrite")
        shutil.rmtree(out_img, ignore_errors=True)
        shutil.rmtree(out_lbl, ignore_errors=True)
        (data / "labels" / f"{name}.cache").unlink(missing_ok=True)
    out_img.mkdir(parents=True)
    out_lbl.mkdir(parents=True)

    kept, with_items = Counter(), 0
    for path in labels:
        src = (src_img / f"{path.stem}.png").resolve()
        if not src.is_file():
            raise SystemExit(f"이미지가 없다: {src_img / path.stem}.png")
        os.symlink(src, out_img / f"{path.stem}.png")
        lines = keep_lines(path.read_text(), keep)
        (out_lbl / path.name).write_text("".join(f"{line}\n" for line in lines))
        kept.update(int(line.split()[0]) for line in lines)
        with_items += bool(lines)
    return {"images": len(labels), "with_items": with_items, "boxes": dict(sorted(kept.items()))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", nargs="+", required=True, help="남길 품목 (어노테이션 영문 이름)")
    parser.add_argument("--suffix", default="small")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.suffix.isidentifier():
        raise SystemExit(f"--suffix 는 영문 · 숫자 · _ 만 쓴다: {args.suffix}")
    names = yolo.class_names(load_items())
    unknown = sorted(set(args.items) - set(names))
    if unknown:
        raise SystemExit(f"모르는 품목: {unknown}. 고를 수 있는 것: {names}")
    keep = {names.index(i) for i in args.items}

    result = {split: build_split(args.data, split, args.suffix, keep, args.overwrite) for split in SPLITS}

    yaml = [f"path: {args.data.resolve()}", f"train: images/train_{args.suffix}", f"val: images/val_{args.suffix}",
            "test: images/test_hidden", "names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    out_yaml = args.data / f"pidray_{args.suffix}.yaml"
    out_yaml.write_text("\n".join(yaml) + "\n", encoding="utf-8")
    manifest = {"items": sorted(args.items, key=names.index), "class_indices": sorted(keep), "splits": result}
    manifest_path = args.data / f"{args.suffix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"# 품목 부분 라벨셋 (#54)\n\n남긴 품목: {', '.join(manifest['items'])}\n")
    print("| 분할 | 사진 | 품목이 든 사진 | " + " | ".join(manifest["items"]) + " |\n|---|---:|---:|" + "---:|" * len(keep))
    for split, r in result.items():
        boxes = " | ".join(f"{r['boxes'].get(names.index(i), 0):,}" for i in manifest["items"])
        print(f"| {split}_{args.suffix} | {r['images']:,} | {r['with_items']:,} | {boxes} |")
    print(f"\n`{out_yaml}`, `{manifest_path}` 를 썼다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
