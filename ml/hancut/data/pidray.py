"""PIDray 어노테이션에서 품목 단위 이진 인덱스를 만든다 (#6).

정답의 단위는 **이미지 × 품목**이다. 받은 47,677장은 전부 양성이라
"빈 가방"이라는 음성이 없다. 대신 품목 c 를 기준으로 본다 —
c 가 든 가방이 양성, 안 든 가방이 음성이다. 12종이면 이진 문제 12개다.

주의: `images[].id` 는 어노테이션 파일마다 0 부터 다시 시작한다.
train 의 0 번과 test_easy 의 0 번은 다른 사진이다. 파일명(확장자 제외)을 키로 쓴다.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

INDEX_FIELDS = ("image_id", "file_name", "split", "difficulty", "item", "y_true")

# 어노테이션 파일명 → (split, difficulty)
SPLIT_OF = {
    "xray_train": ("train", ""),
    "xray_test_easy": ("test", "easy"),
    "xray_test_hard": ("test", "hard"),
    "xray_test_hidden": ("test", "hidden"),
}


@dataclass(frozen=True)
class Row:
    image_id: str
    file_name: str
    split: str
    difficulty: str
    item: str
    y_true: int


def annotation_files(annotations_dir: Path) -> list[Path]:
    """`xray_*.json` 만 고른다. 맥 압축 부산물(`._*`)은 제외한다."""
    found = sorted(p for p in annotations_dir.glob("xray_*.json") if not p.name.startswith("._"))
    names = {p.stem for p in found}
    missing = set(SPLIT_OF) - names
    if missing:
        raise FileNotFoundError(
            f"어노테이션 파일이 빠졌다: {sorted(missing)}. split 하나가 통째로 사라지므로 계속하지 않는다"
        )
    unknown = names - set(SPLIT_OF)
    if unknown:
        raise ValueError(f"모르는 어노테이션 파일이다: {sorted(unknown)}. SPLIT_OF 에 추가할지 판단한다")
    return found


def rows_from_file(path: Path) -> Iterator[Row]:
    """한 어노테이션 파일에서 이미지 × 품목 행을 만든다."""
    split, difficulty = SPLIT_OF[path.stem]
    data = json.loads(path.read_text(encoding="utf-8"))

    names = {c["id"]: c["name"] for c in data["categories"]}
    if not names:
        raise ValueError(f"{path.name} 에 categories 가 없다")

    positive: dict[int, set[str]] = {}
    for a in data["annotations"]:
        item = names.get(a["category_id"])
        if item is None:
            raise ValueError(f"{path.name}: categories 에 없는 category_id {a['category_id']}")
        positive.setdefault(a["image_id"], set()).add(item)

    every_item = sorted(names.values())
    for image in data["images"]:
        file_name = image["file_name"]
        image_id = Path(file_name).stem
        present = positive.get(image["id"], set())
        for item in every_item:
            yield Row(image_id, file_name, split, difficulty, item, int(item in present))


def build_index(annotations_dir: Path) -> list[Row]:
    return [row for path in annotation_files(annotations_dir) for row in rows_from_file(path)]


def write_index(rows: Iterable[Row], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: getattr(row, f) for f in INDEX_FIELDS})
            written += 1
    return written


def read_index(path: Path) -> list[Row]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != list(INDEX_FIELDS):
            raise ValueError(f"열이 다르다: {reader.fieldnames}")
        return [Row(r["image_id"], r["file_name"], r["split"], r["difficulty"],
                    r["item"], int(r["y_true"])) for r in reader]


def leakage(rows: Iterable[Row]) -> dict[str, set[str]]:
    """같은 사진이 여러 split 에 들어가 있으면 반환한다. 비어 있어야 정상이다.

    공식 분할을 그대로 믿지 않는다. Weqaa 에서 같은 영상 프레임이 섞인 사례를 겪었다.
    """
    where: dict[str, set[str]] = {}
    for row in rows:
        where.setdefault(row.image_id, set()).add(f"{row.split}/{row.difficulty}".rstrip("/"))
    return {image_id: splits for image_id, splits in where.items() if len(splits) > 1}


def summarize(rows: Iterable[Row]) -> list[dict]:
    """품목별·split 별 양성 수. items.json 의 실측값과 대조하는 데 쓴다."""
    positives: Counter[tuple[str, str]] = Counter()
    images: dict[str, set[str]] = {}
    for row in rows:
        key = f"{row.split}_{row.difficulty}".rstrip("_")
        images.setdefault(key, set()).add(row.image_id)
        if row.y_true:
            positives[(row.item, key)] += 1

    splits = sorted(images)
    items = sorted({item for item, _ in positives})
    out = []
    for item in items:
        entry = {"item": item}
        entry.update({s: positives[(item, s)] for s in splits})
        entry["test_total"] = sum(positives[(item, s)] for s in splits if s.startswith("test"))
        out.append(entry)
    return out


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "_비어 있다_"
    cols = [c for c in rows[0] if c != "item"]
    lines = ["| 품목 | " + " | ".join(cols) + " |",
             "|---|" + "---:|" * len(cols)]
    for r in sorted(rows, key=lambda r: -r["test_total"]):
        lines.append(f"| {r['item']} | " + " | ".join(f"{r[c]:,}" for c in cols) + " |")
    return "\n".join(lines)
