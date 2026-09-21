from __future__ import annotations

"""
1단계 서브셋 추출 — 시설 종류 × 준수 여부로 층화 표본을 뽑는다.

전체 608,280장(약 962GB)을 다 쓰지 않는다. 점검 빈도가 높은 8종만, 시설마다
준수·미준수를 같은 수씩 뽑아 약 8만 장 규모로 맞춘다 (docs/data.md · B안).

입력은 라벨에서 만든 인덱스 CSV다. 파서는 W2에 라벨 스키마를 확인한 뒤 작성한다.

    image_id,facility,compliant,source
    01_001_0001_O_F_00000001,소형소화기,1,site
"""

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

INDEX_FIELDS = ("image_id", "facility", "compliant", "source")


@dataclass(frozen=True)
class Record:
    image_id: str
    facility: str
    compliant: bool
    source: str = ""

    @property
    def stratum(self) -> tuple[str, bool]:
        return (self.facility, self.compliant)


def read_index(path: Path) -> list[Record]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(INDEX_FIELDS[:3]) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"인덱스 CSV에 필요한 열이 없다: {sorted(missing)}")
        return [
            Record(
                image_id=row["image_id"],
                facility=row["facility"],
                compliant=str(row["compliant"]).strip() in {"1", "true", "True"},
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
                "facility": r.facility,
                "compliant": int(r.compliant),
                "source": r.source,
            })


def stratified_subset(
    records: Sequence[Record],
    facilities: Sequence[str],
    per_stratum: int,
    seed: int,
) -> list[Record]:
    """
    시설 × 준수 여부마다 최대 per_stratum 장씩 뽑는다.

    - 주어진 시설 목록에 없는 기록은 버린다 (1단계는 8종만)
    - 표본이 모자란 층은 있는 만큼만 가져간다 — 조용히 채우지 않는다
    - seed 로 완전히 재현된다. 같은 seed·입력이면 결과가 같다
    """
    if per_stratum <= 0:
        raise ValueError("per_stratum 은 1 이상이어야 한다")

    wanted = set(facilities)
    buckets: dict[tuple[str, bool], list[Record]] = {}
    for record in records:
        if record.facility in wanted:
            buckets.setdefault(record.stratum, []).append(record)

    rng = random.Random(seed)
    picked: list[Record] = []
    for stratum in sorted(buckets, key=lambda s: (s[0], s[1])):
        pool = sorted(buckets[stratum], key=lambda r: r.image_id)
        picked.extend(pool if len(pool) <= per_stratum else rng.sample(pool, per_stratum))
    return picked


def summarize(records: Sequence[Record]) -> list[dict]:
    """시설별 준수/미준수 장수 — 추출 결과를 눈으로 확인하는 용도."""
    counts = Counter((r.facility, r.compliant) for r in records)
    facilities = sorted({facility for facility, _ in counts})
    return [
        {
            "facility": facility,
            "compliant": counts.get((facility, True), 0),
            "noncompliant": counts.get((facility, False), 0),
            "total": counts.get((facility, True), 0) + counts.get((facility, False), 0),
        }
        for facility in facilities
    ]


def shortfalls(summary: Sequence[dict], per_stratum: int) -> list[str]:
    """목표 장수를 채우지 못한 층 — 데이터가 부족하다는 신호이므로 보고한다."""
    short: list[str] = []
    for row in summary:
        if row["compliant"] < per_stratum:
            short.append(f"{row['facility']} 준수 {row['compliant']}/{per_stratum}")
        if row["noncompliant"] < per_stratum:
            short.append(f"{row['facility']} 미준수 {row['noncompliant']}/{per_stratum}")
    return short
