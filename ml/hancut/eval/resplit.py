"""
시험 사진을 보정용 / 평가용으로 나눈다 (#25).

첫 E2 에서 학습 분포로 정한 임계값이 시험 분포에서 넘쳤다. 시험 분포로 보정하려면
시험셋 일부로 임계값을 정하고, 겹치지 않는 나머지로 평가해야 한다.

- 사진 단위로 나눈다. 한 사진의 품목 12행은 반드시 같은 쪽으로 간다. 갈라지면 같은
  사진이 보정과 평가 양쪽에 들어가 누수가 된다.
- 난이도(easy / hard / hidden) × 가장 드문 품목으로 층을 나눈다. hidden 이나 경봉이
  한쪽에 몰리면 그쪽 결과만 좋아 보이거나 나빠 보인다.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Iterable, Mapping, Sequence

_DIFFICULTY = re.compile(r"xray_(easy|hard|hidden)\d+$")


def difficulty_of(image_id: str) -> str:
    """xray_hidden00012 → 'hidden'. 시험 사진이 아니면 빈 문자열."""
    m = _DIFFICULTY.match(image_id)
    return m.group(1) if m else ""


def image_table(rows: Iterable[Mapping]) -> dict[str, tuple[str, frozenset[str]]]:
    """예측 행 → {사진: (난이도, 그 사진에 있는 품목들)}."""
    present: dict[str, set[str]] = {}
    for r in rows:
        items = present.setdefault(r["image_id"], set())
        if r["y_true"] == 1:
            items.add(r["item"])
    return {i: (difficulty_of(i), frozenset(items)) for i, items in present.items()}


def split_images(
    images: Mapping[str, tuple[str, frozenset[str]]],
    fraction: float = 0.5,
    seed: int = 0,
) -> tuple[set[str], set[str]]:
    """(보정용, 평가용) 사진 집합. 같은 seed·입력이면 결과가 같다."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction 은 0 과 1 사이여야 한다")
    frequency = Counter(item for _, items in images.values() for item in items)

    def stratum(entry: tuple[str, frozenset[str]]) -> tuple[str, str]:
        difficulty, items = entry
        rarest = min(items, key=lambda i: (frequency[i], i)) if items else ""
        return difficulty, rarest

    strata: dict[tuple[str, str], list[str]] = {}
    for image_id in sorted(images):
        strata.setdefault(stratum(images[image_id]), []).append(image_id)

    rng = random.Random(seed)
    calib: set[str] = set()
    for key in sorted(strata):
        ids = strata[key]
        rng.shuffle(ids)
        calib.update(ids[: round(len(ids) * fraction)])
    return calib, set(images) - calib


def partition(rows: Sequence[Mapping], calib_ids: set[str]) -> tuple[list, list]:
    """행을 사진 기준으로 (보정, 평가) 로 나눈다."""
    calib = [r for r in rows if r["image_id"] in calib_ids]
    rest = [r for r in rows if r["image_id"] not in calib_ids]
    return calib, rest
