#!/usr/bin/env python3
"""manifest가 있는 dense CSV → seed 고정 설정 F·5% bundle 및 E2(5/2/1%) 결과."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hancut.config import load_items
from hancut.data.yolo import class_names
from hancut.eval.bundle import export_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pred_dir", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("새 출력 폴더를 지정하세요")
    bundle, report = export_bundle(args.pred_dir, class_names(load_items()), args.seed)
    args.out.mkdir(parents=True)
    for name, value in (("dense-bundle.json", bundle), ("e2.json", report)):
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
