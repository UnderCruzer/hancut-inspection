#!/usr/bin/env python3
"""갈래마다 품목별 이미지 AP 를 분할별로 비교한다 (#56).

    python3 ml/scripts/image_ap.py --arm E1=$E1 --arm E1+cls=$OUT/e1_cls \\
        --items Lighter Bullet --out $OUT/image_ap.md

이미지 AP 는 사진 × 품목 점수로 "그 품목이 있는 사진"을 얼마나 위에 올리는가다. 박스 위치는 보지 않는다.
검출기 AP50 과 다른 값이다. 판정에 쓰는 점수 그대로를 잰다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hancut.eval import cli, fusion  # noqa: E402

SPLITS = ("test_easy", "test_hard", "test_hidden")


def parse_arms(specs: list[str]) -> dict[str, Path]:
    out = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"'{spec}' 를 읽지 못했다. 형식은 이름=예측 폴더")
        out[name] = Path(path)
    return out


def item_aps(rows: list[dict], items: list[str]) -> dict[str, tuple[float, int]]:
    """품목 → (이미지 AP, 양성 수)."""
    out = {}
    for item in items:
        mine = [r for r in rows if r["item"] == item]
        if not mine:
            raise SystemExit(f"품목 {item} 의 행이 없다")
        out[item] = (fusion.average_precision([r["y_true"] for r in mine], [r["score"] for r in mine]), sum(r["y_true"] for r in mine))
    return out


def report(table: dict[str, dict[str, dict[str, tuple[float, int]]]], items: list[str], splits: list[str]) -> str:
    arms = list(table)
    lines = ["# 품목별 이미지 AP (#56)", ""]
    for split in splits:
        lines += [f"## {split}", "", "| 품목 | 양성 | " + " | ".join(arms) + " |", "|---|---:|" + "---:|" * len(arms)]
        for item in items:
            n = table[arms[0]][split][item][1]
            lines.append(f"| {item} | {n:,} | " + " | ".join(f"{table[a][split][item][0]:.3f}" for a in arms) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, help="이름=예측 폴더 (여러 번)")
    parser.add_argument("--items", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--out", type=Path, required=True, help="보고서 .md")
    args = parser.parse_args()

    arms = parse_arms(args.arm)
    table = {}
    for name, folder in arms.items():
        table[name] = {}
        for split in args.splits:
            path = folder / f"{split}.csv"
            if not path.is_file():
                raise SystemExit(f"없다: {path}")
            table[name][split] = item_aps(cli.read_predictions(path), args.items)
    text = report(table, args.items, args.splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
