#!/usr/bin/env python3
"""받은 데이터셋의 구조를 조사해 마크다운으로 보고한다 (#14).

표준 라이브러리만 쓴다. 맥·윈도우·EC2 어디서든 설치 없이 돌아간다.

    python3 ml/scripts/inspect_dataset.py data/pidray

확인하려는 것은 넷이다.
  1. 판본 — 47,677장(전부 양성)인가 124,486장(음성 포함)인가
  2. 장비 메타데이터가 실제로 있는가  → 없으면 E4 불성립
  3. 난이도 라벨(easy/hard/hidden)이 어디까지 붙어 있는가  → E5 설계가 갈린다
  4. 클래스 이름과 id 의 실제 대응  → items.json 의 class_id 를 채운다
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ANNOT_EXT = {".json", ".xml", ".txt", ".csv"}
KNOWN_TOTALS = {47677: "v1 — 전부 양성. 음성 없음", 124486: "확장판 — 음성 포함"}
DIFFICULTY = ("easy", "hard", "hidden")
DEVICE_HINT = re.compile(r"device|machine|scanner|camera|source|sensor|model", re.I)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}GB"


def walk(root: Path) -> list[Path]:
    return sorted((p for p in root.rglob("*") if p.is_file()), key=str)


def section(title: str) -> None:
    print(f"\n## {title}\n")


def report_tree(root: Path, depth: int = 2) -> None:
    section("최상위 구조")
    entries = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if len(rel.parts) > depth:
            continue
        if p.is_dir():
            files = [f for f in p.rglob("*") if f.is_file()]
            entries.append((str(rel) + "/", len(files), sum(f.stat().st_size for f in files)))
    if not entries:
        print("_하위 디렉터리 없음_")
        return
    print("| 경로 | 파일 수 | 용량 |")
    print("|---|---:|---:|")
    for name, count, size in entries[:40]:
        print(f"| `{name}` | {count:,} | {human(size)} |")
    if len(entries) > 40:
        print(f"\n_… 외 {len(entries) - 40}개 디렉터리_")


def report_extensions(files: list[Path]) -> None:
    section("확장자별")
    counts: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    for f in files:
        ext = f.suffix.lower() or "(없음)"
        counts[ext] += 1
        sizes[ext] += f.stat().st_size
    print("| 확장자 | 파일 수 | 용량 |")
    print("|---|---:|---:|")
    for ext, count in counts.most_common(15):
        print(f"| `{ext}` | {count:,} | {human(sizes[ext])} |")


def report_edition(files: list[Path]) -> None:
    section("1. 판본")
    images = [f for f in files if f.suffix.lower() in IMAGE_EXT]
    total = len(images)
    print(f"이미지 **{total:,}장**")
    match = KNOWN_TOTALS.get(total)
    if match:
        print(f"\n→ **{match}**")
    else:
        near = min(KNOWN_TOTALS, key=lambda k: abs(k - total))
        print(f"\n→ 알려진 수치와 다르다. 가장 가까운 것은 {near:,} ({KNOWN_TOTALS[near]}). "
              f"차이 {abs(near - total):,}장. 공식 split 과 대조가 필요하다.")


def report_difficulty(files: list[Path]) -> None:
    section("2. 난이도 라벨 (E5)")
    hits: Counter[str] = Counter()
    where: dict[str, Counter[str]] = {d: Counter() for d in DIFFICULTY}
    for f in files:
        low = str(f).lower()
        for d in DIFFICULTY:
            if d in low:
                hits[d] += 1
                parts = f.relative_to(f.anchor).parts if f.is_absolute() else f.parts
                where[d][parts[0] if parts else "?"] += 1
    if not hits:
        print("경로·파일명에서 난이도 단서를 찾지 못했다. 어노테이션 파일 안을 봐야 한다.")
        return
    print("| 난이도 | 경로에 포함된 파일 수 |")
    print("|---|---:|")
    for d in DIFFICULTY:
        print(f"| {d} | {hits[d]:,} |")
    print("\n**학습셋에도 난이도가 붙어 있는지**가 핵심이다. "
          "test 쪽에만 있으면 E5 는 시험셋에서만 본다.")


def report_annotations(files: list[Path], root: Path) -> None:
    section("3·4. 어노테이션 — 장비(E4)와 클래스 id")
    annots = [f for f in files if f.suffix.lower() in ANNOT_EXT]
    if not annots:
        print("어노테이션 후보 파일이 없다.")
        return
    annots.sort(key=lambda f: f.stat().st_size, reverse=True)
    print("가장 큰 어노테이션 파일 5개\n")
    print("| 파일 | 용량 |")
    print("|---|---:|")
    for f in annots[:5]:
        print(f"| `{f.relative_to(root)}` | {human(f.stat().st_size)} |")

    for f in annots[:3]:
        if f.suffix.lower() != ".json":
            continue
        print(f"\n### `{f.relative_to(root)}`\n")
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # 형식이 다르거나 너무 클 수 있다
            print(f"읽지 못했다: {exc}")
            continue
        if isinstance(data, dict):
            print("최상위 키: " + ", ".join(f"`{k}`" for k in list(data)[:15]))
            cats = data.get("categories")
            if isinstance(cats, list) and cats:
                print("\n**클래스 id 실제 대응** — `ml/configs/items.json` 의 `class_id` 에 넣을 값\n")
                print("| id | name |")
                print("|---:|---|")
                for c in cats:
                    if isinstance(c, dict):
                        print(f"| {c.get('id')} | {c.get('name')} |")
            imgs = data.get("images")
            if isinstance(imgs, list) and imgs and isinstance(imgs[0], dict):
                keys = list(imgs[0])
                print(f"\n`images[0]` 키: " + ", ".join(f"`{k}`" for k in keys))
                dev = [k for k in keys if DEVICE_HINT.search(k)]
                print(f"\n장비 후보 필드: **{', '.join(dev) if dev else '없음'}**")
                if dev:
                    for k in dev:
                        vals = Counter(str(i.get(k)) for i in imgs[:20000])
                        print(f"\n`{k}` 상위 값: " + ", ".join(
                            f"`{v}`({n:,})" for v, n in vals.most_common(8)))
                else:
                    print("\n→ **장비 메타데이터가 없으면 E4(장비 교차)는 성립하지 않는다.** "
                          "파일명 규칙에 장비가 숨어 있는지 아래를 확인한다.")
                print(f"\n`images[0]` 예시: `{json.dumps(imgs[0], ensure_ascii=False)[:300]}`")
        elif isinstance(data, list) and data:
            print(f"리스트, 길이 {len(data):,}. 첫 항목: `{json.dumps(data[0], ensure_ascii=False)[:300]}`")


def report_filenames(files: list[Path]) -> None:
    section("파일명 규칙")
    images = [f for f in files if f.suffix.lower() in IMAGE_EXT]
    if not images:
        print("이미지가 없다.")
        return
    print("예시 10개\n")
    for f in images[:10]:
        print(f"- `{f.name}`")
    stems = [re.sub(r"\d+", "#", f.stem) for f in images]
    print("\n숫자를 `#` 로 바꾼 패턴 상위 10개\n")
    print("| 패턴 | 파일 수 |")
    print("|---|---:|")
    for pat, n in Counter(stems).most_common(10):
        print(f"| `{pat}` | {n:,} |")
    print("\n장비가 어노테이션에 없다면 이 패턴에 숨어 있을 수 있다.")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.is_dir():
        print(f"디렉터리가 아니다: {root}")
        return 1

    print(f"# 데이터셋 조사 — `{root.name}`\n")
    files = walk(root)
    total_size = sum(f.stat().st_size for f in files)
    print(f"파일 {len(files):,}개, 합계 {human(total_size)}")

    report_tree(root)
    report_extensions(files)
    report_edition(files)
    report_difficulty(files)
    report_annotations(files, root)
    report_filenames(files)

    section("다음")
    print("이 출력을 그대로 붙여넣으면 `ml/configs/items.json` 의 `class_id` 와 "
          "`devices.ids` 를 채우고, 라벨 로더(#6) 설계를 확정한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
