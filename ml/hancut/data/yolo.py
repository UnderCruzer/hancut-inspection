"""
PIDray COCO 어노테이션 → YOLO 학습 데이터 (#23).

클래스 번호는 `ml/configs/items.json` 의 `class_id` 에서 읽는다. 논문 순서를 가정하지 않는다(D5).
YOLO 는 0 부터 시작하는 번호를 쓰므로 class_id 를 정렬해 0..11 로 옮긴다.

학습 이미지는 train / val / calib 세 조각으로 나눈다.
- train: 가중치 학습
- val: 학습 중 에폭 선택
- calib: 임계값 보정(E2). val 과 섞으면 임계값이 낙관적으로 나온다
시험셋(easy / hard / hidden)은 여기서 다루지 않는다. 학습에도 보정에도 쓰지 않는다.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SPLIT_FRACTIONS = {"train": 0.85, "val": 0.05, "calib": 0.10}


@dataclass(frozen=True)
class Box:
    """YOLO 라벨 한 줄. 좌표는 이미지 크기로 나눈 0..1 값이다."""

    class_index: int
    cx: float
    cy: float
    w: float
    h: float

    def line(self) -> str:
        return f"{self.class_index} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


def class_index_map(items: Iterable[Mapping]) -> dict[int, int]:
    """COCO category_id → YOLO 클래스 번호(0 부터)."""
    raw = [i["class_id"] for i in items]
    # 정렬보다 먼저 본다. None 이 섞이면 정렬이 TypeError 로 먼저 터져 원인이 가려진다.
    if any(i is None for i in raw):
        raise ValueError("class_id 가 비어 있다. 실제 어노테이션에서 읽어 채운 뒤 쓴다")
    if len(raw) != len(set(raw)):
        raise ValueError("items.json 에 같은 class_id 가 두 번 있다")
    return {cid: n for n, cid in enumerate(sorted(raw))}


def class_names(items: Iterable[Mapping]) -> list[str]:
    """YOLO 클래스 번호 순서의 품목 이름 (어노테이션의 영문 이름)."""
    return [i["dataset_name"] for i in sorted(items, key=lambda i: i["class_id"])]


def to_yolo(bbox: Sequence[float], width: int, height: int, class_index: int) -> Box | None:
    """COCO bbox(좌상단 x, y, 너비, 높이 — 픽셀) → YOLO Box.

    이미지 밖으로 나간 부분은 잘라낸다. 잘라낸 뒤 넓이가 0 이면 None 을 돌려준다.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"이미지 크기가 잘못됐다: {width}x{height}")
    x, y, w, h = (float(v) for v in bbox)
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return Box(
        class_index,
        (x0 + x1) / 2 / width,
        (y0 + y1) / 2 / height,
        (x1 - x0) / width,
        (y1 - y0) / height,
    )


def labels_from_coco(data: Mapping, index_map: Mapping[int, int]) -> tuple[dict[str, list[Box]], int]:
    """어노테이션 파일 하나 → {파일명(확장자 제외): [Box, ...]}, 버린 박스 수.

    박스가 하나도 없는 이미지도 빈 목록으로 넣는다. 빠뜨리면 YOLO 가 그 이미지를 배경으로
    배우지 못하고, 점수 추출에서도 행이 사라진다.
    """
    images = {img["id"]: img for img in data["images"]}
    labels: dict[str, list[Box]] = {Path(img["file_name"]).stem: [] for img in images.values()}
    dropped = 0
    for a in data["annotations"]:
        if a["category_id"] not in index_map:
            raise ValueError(f"items.json 에 없는 category_id {a['category_id']}")
        img = images[a["image_id"]]
        box = to_yolo(a["bbox"], img["width"], img["height"], index_map[a["category_id"]])
        if box is None:
            dropped += 1
            continue
        labels[Path(img["file_name"]).stem].append(box)
    return labels, dropped


def split_train(
    labels: Mapping[str, Sequence[Box]],
    fractions: Mapping[str, float] = SPLIT_FRACTIONS,
    seed: int = 0,
) -> dict[str, list[str]]:
    """학습 이미지를 나눈다. 이미지마다 '가장 드문 품목'으로 층을 나눠, 드문 품목이 한쪽에 몰리지 않게 한다.

    같은 seed·입력이면 결과가 같다.
    """
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("분할 비율의 합이 1 이 아니다")

    frequency = Counter(b.class_index for boxes in labels.values() for b in boxes)

    def stratum(boxes: Sequence[Box]) -> int:
        if not boxes:
            return -1
        return min({b.class_index for b in boxes}, key=lambda c: (frequency[c], c))

    strata: dict[int, list[str]] = {}
    for stem in sorted(labels):
        strata.setdefault(stratum(labels[stem]), []).append(stem)

    rng = random.Random(seed)
    names = list(fractions)
    out: dict[str, list[str]] = {n: [] for n in names}
    for key in sorted(strata):
        stems = strata[key]
        rng.shuffle(stems)
        n = len(stems)
        # 앞의 조각은 반올림, 마지막 조각이 나머지를 가져간다
        start = 0
        for i, name in enumerate(names):
            size = n - start if i == len(names) - 1 else round(n * fractions[name])
            out[name].extend(stems[start:start + size])
            start += size
    return {n: sorted(v) for n, v in out.items()}


def class_counts(stems: Iterable[str], labels: Mapping[str, Sequence[Box]]) -> Counter:
    """품목별로 그 품목이 든 이미지 수."""
    counts: Counter = Counter()
    for stem in stems:
        for c in {b.class_index for b in labels[stem]}:
            counts[c] += 1
    return counts
