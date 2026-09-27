#!/usr/bin/env python3
"""PIDray 어노테이션 → 품목 단위 이진 인덱스 CSV (#6).

    python3 ml/scripts/build_index.py data/pidray/pidray/annotations data/index.csv

세 가지를 함께 한다.
  1. 인덱스 생성 — 이미지 x 품목 행
  2. 누수 검사 — 같은 사진이 여러 split 에 있으면 실패로 끝낸다
  3. 대조 — 품목별 양성 수를 ml/configs/items.json 의 실측값과 맞춰 본다
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hancut.config import load_items  # noqa: E402
from hancut.data import pidray  # noqa: E402


def cross_check(summary: list[dict]) -> list[str]:
    """items.json 에 적힌 실측 양성 수와 비교한다. 어긋나면 어느 쪽이든 틀린 것이다."""
    expected = {i["dataset_name"]: i["positives"] for i in load_items()
                if "positives" in i}
    if not expected:
        return ["items.json 에 positives 가 없어 대조를 건너뛴다"]

    problems = []
    for row in summary:
        want = expected.get(row["item"])
        if want is None:
            problems.append(f"{row['item']}: items.json 에 없는 품목")
            continue
        for split in ("train", "test_easy", "test_hard", "test_hidden"):
            got = row.get(split, 0)
            if got != want[split]:
                problems.append(f"{row['item']} {split}: 인덱스 {got:,} vs items.json {want[split]:,}")
    missing = set(expected) - {r["item"] for r in summary}
    problems += [f"{m}: 인덱스에 없다" for m in sorted(missing)]
    return problems


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    annotations_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])

    print(f"# 인덱스 생성\n\n어노테이션: `{annotations_dir}`")
    rows = pidray.build_index(annotations_dir)
    images = {r.image_id for r in rows}
    print(f"\n이미지 **{len(images):,}장**, 행 **{len(rows):,}개**")

    print("\n## 누수 검사\n")
    found = pidray.leakage(rows)
    if found:
        print(f"**같은 사진이 여러 split 에 있다 — {len(found):,}건.** 공식 분할을 쓸 수 없다.\n")
        for image_id, splits in sorted(found.items())[:20]:
            print(f"- `{image_id}` → {', '.join(sorted(splits))}")
        if len(found) > 20:
            print(f"- … 외 {len(found) - 20:,}건")
        print("\n인덱스를 쓰지 않고 끝낸다. 분할을 다시 짜야 한다.")
        return 1
    print("누수 없음. 네 split 이 서로 겹치지 않는다.")

    summary = pidray.summarize(rows)
    print("\n## 품목별 양성 수\n")
    print(pidray.format_table(summary))

    print("\n## items.json 과 대조\n")
    problems = cross_check(summary)
    if problems:
        print("**어긋난다.** 어느 쪽이든 틀린 것이므로 고치기 전에는 인덱스를 쓰지 않는다.\n")
        for p in problems[:30]:
            print(f"- {p}")
        return 1
    print("12종 x 4 split 모두 일치한다.")

    written = pidray.write_index(rows, out_path)
    size = out_path.stat().st_size / 1024 / 1024
    print(f"\n## 결과\n\n`{out_path}` — {written:,}행, {size:.1f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
