#!/usr/bin/env python3
"""PIDray → YOLO 학습 데이터 (#23).

    python3 ml/scripts/make_yolo_dataset.py            # 인자 없이: data/pidray 를 찾아 data/yolo 로 만든다
    python3 ml/scripts/make_yolo_dataset.py --overwrite  # 이미 있으면 지우고 다시 만든다

이미지는 복사하지 않고 심볼릭 링크로 건다. 11GB 를 두 번 쓰지 않는다.

    data/yolo/
      images/{train,val,calib,test_easy,test_hard,test_hidden}/*.png   원본으로의 링크
      labels/{같은 split}/*.txt                                         YOLO 라벨
      pidray.yaml                                                       ultralytics 설정
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hancut.config import load_items  # noqa: E402
from hancut.data import yolo  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "yolo"
CANDIDATES = (REPO / "data" / "pidray" / "pidray", REPO / "data" / "pidray")

# 어노테이션 파일 → 이미지 폴더, 출력 split 이름
TEST_SPLITS = {
    "xray_test_easy": ("easy", "test_easy"),
    "xray_test_hard": ("hard", "test_hard"),
    "xray_test_hidden": ("hidden", "test_hidden"),
}


def find_root(explicit: Path | None) -> Path:
    if explicit is not None:
        if not (explicit / "annotations" / "xray_train.json").is_file():
            raise SystemExit(f"{explicit} 에 annotations/xray_train.json 이 없다")
        return explicit
    for c in CANDIDATES:
        if (c / "annotations" / "xray_train.json").is_file():
            return c
    raise SystemExit("PIDray 를 찾지 못했다. 다음 위치를 봤다:\n" + "\n".join(f"  - {c}" for c in CANDIDATES))


def link_split(out: Path, name: str, stems: list[str], src_dir: Path, labels: dict, missing: list[str]) -> int:
    img_out, lbl_out = out / "images" / name, out / "labels" / name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    boxes = 0
    for stem in stems:
        src = src_dir / f"{stem}.png"
        if not src.is_file():
            missing.append(str(src))
            continue
        os.symlink(src.resolve(), img_out / src.name)
        lines = [b.line() for b in labels[stem]]
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        boxes += len(lines)
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true", help="data/yolo 가 있으면 지우고 다시 만든다")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, help="PIDray 폴더 (annotations/ 가 들어 있는 곳). 생략하면 data/ 에서 찾는다")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    root = find_root(args.root)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{out} 가 이미 있다. 다시 만들려면 --overwrite")
        if out.name != "yolo":
            # 엉뚱한 폴더를 통째로 지우지 않도록 이름으로 한 번 더 막는다
            raise SystemExit(f"--overwrite 는 이름이 yolo 인 폴더에만 쓴다: {out}")
        shutil.rmtree(out)

    items = load_items()
    index_map, names = yolo.class_index_map(items), yolo.class_names(items)
    print(f"# YOLO 데이터 생성\n\n원본: `{root}`\n출력: `{out}`\n")

    report, missing, dropped_total = [], [], 0

    train_data = json.loads((root / "annotations" / "xray_train.json").read_text(encoding="utf-8"))
    train_labels, dropped = yolo.labels_from_coco(train_data, index_map)
    dropped_total += dropped
    parts = yolo.split_train(train_labels, seed=args.seed)
    for name in ("train", "val", "calib"):
        boxes = link_split(out, name, parts[name], root / "train", train_labels, missing)
        report.append((name, len(parts[name]), boxes, yolo.class_counts(parts[name], train_labels)))

    for stem_name, (folder, name) in TEST_SPLITS.items():
        data = json.loads((root / "annotations" / f"{stem_name}.json").read_text(encoding="utf-8"))
        labels, dropped = yolo.labels_from_coco(data, index_map)
        dropped_total += dropped
        stems = sorted(labels)
        boxes = link_split(out, name, stems, root / folder, labels, missing)
        report.append((name, len(stems), boxes, yolo.class_counts(stems, labels)))

    if missing:
        print(f"**이미지 {len(missing):,}장이 없다.** 어노테이션과 폴더가 맞지 않는다. 앞의 5개:")
        for m in missing[:5]:
            print(f"  - {m}")
        return 1

    print("| split | 이미지 | 박스 | 품목 누락 |")
    print("|---|---:|---:|---|")
    for name, n, boxes, counts in report:
        absent = [names[i] for i in range(len(names)) if counts[i] == 0]
        print(f"| {name} | {n:,} | {boxes:,} | {', '.join(absent) or '없음'} |")

    calib_counts = next(c for n, _, _, c in report if n == "calib")
    print("\ncalib 품목별 양성 (임계값 보정 표본):")
    print("  " + " · ".join(f"{names[i]} {calib_counts[i]}" for i in range(len(names))))
    print(f"\n이미지 밖으로 나가 버린 박스: {dropped_total}")

    yaml = [f"path: {out}", "train: images/train", "val: images/val", "test: images/test_hidden", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
    (out / "pidray.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"\n`{out / 'pidray.yaml'}` 를 썼다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
