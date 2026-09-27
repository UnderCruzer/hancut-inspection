"""
층화 표본 추출 — 품목 × 위해물품 유무로 층을 나눠 뽑는다.

**현재 파이프라인에서는 쓰지 않는다.** PIDray 인덱스는 `hancut.data.pidray` 가 만들고,
학습·평가는 공식 split 을 그대로 쓴다. 이 모듈은 임계값 보정용 검증셋을 학습셋에서
품목별로 층화해 떼어낼 때 재사용할 수 있어 남겨둔다.

입력 형식이 `build_index.py` 가 만드는 인덱스(`image_id,file_name,split,difficulty,item,y_true`)와
다르다. 재사용하려면 그 인덱스를 아래 형식으로 옮기거나, 이 모듈을 인덱스 형식에 맞춰 고친다.

    image_id,item,clear,source
    xray_00000,Gun,0,train

`clear` 는 1 = 해당 품목 없음이다. `y_true` 와 방향이 반대이므로 옮길 때 뒤집는다.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

INDEX_FIELDS = ("image_id", "item", "clear", "source")


@dataclass(frozen=True)
class Record:
    image_id: str
    item: str
    clear: bool
    source: str = ""

    @property
    def stratum(self) -> tuple[str, bool]:
        return (self.item, self.clear)


def read_index(path: Path) -> list[Record]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(INDEX_FIELDS[:3]) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"인덱스 CSV에 필요한 열이 없다: {sorted(missing)}")
        return [
            Record(
                image_id=row["image_id"],
                item=row["item"],
                clear=str(row["clear"]).strip() in {"1", "true", "True"},
                source=row.get("source", ""),
            )
            for row in reader
        ]


def write_index(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "image_id": r.image_id,
                "item": r.item,
                "clear": int(r.clear),
                "source": r.source,
            })


def stratified_subset(
    records: Sequence[Record],
    items: Sequence[str],
    per_stratum: int,
    seed: int,
) -> list[Record]:
    """
    품목 × 위해물품 유무마다 최대 per_stratum 장씩 뽑는다.

    - 주어진 품목 목록에 없는 기록은 버린다
    - 표본이 모자란 층은 있는 만큼만 가져간다 — 조용히 채우지 않는다
    - seed 로 완전히 재현된다. 같은 seed·입력이면 결과가 같다
    """
    if per_stratum <= 0:
        raise ValueError("per_stratum 은 1 이상이어야 한다")

    wanted = set(items)
    buckets: dict[tuple[str, bool], list[Record]] = {}
    for record in records:
        if record.item in wanted:
            buckets.setdefault(record.stratum, []).append(record)

    rng = random.Random(seed)
    picked: list[Record] = []
    for stratum in sorted(buckets, key=lambda s: (s[0], s[1])):
        pool = sorted(buckets[stratum], key=lambda r: r.image_id)
        picked.extend(pool if len(pool) <= per_stratum else rng.sample(pool, per_stratum))
    return picked


def summarize(records: Sequence[Record]) -> list[dict]:
    """품목별 통과(clear)/적발(threat) 장수 — 추출 결과를 눈으로 확인하는 용도."""
    counts = Counter((r.item, r.clear) for r in records)
    items = sorted({item for item, _ in counts})
    return [
        {
            "item": item,
            "clear": counts.get((item, True), 0),
            "threat": counts.get((item, False), 0),
            "total": counts.get((item, True), 0) + counts.get((item, False), 0),
        }
        for item in items
    ]


def shortfalls(summary: Sequence[dict], per_stratum: int) -> list[str]:
    """목표 장수를 채우지 못한 층 — 데이터가 부족하다는 신호이므로 보고한다."""
    short: list[str] = []
    for row in summary:
        if row["clear"] < per_stratum:
            short.append(f"{row['item']} 통과 {row['clear']}/{per_stratum}")
        if row["threat"] < per_stratum:
            short.append(f"{row['item']} 적발 {row['threat']}/{per_stratum}")
    return short
