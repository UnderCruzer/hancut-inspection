"""
사진 단위 품목 분류기의 입력과 정답 (#56).

검출기는 박스를 찾아야 점수가 나온다. 판정은 사진 × 품목의 "있다 / 없다" 라서, 사진 전체를 보고
품목마다 있을 확률만 내는 분류기를 따로 둔다. 정답은 YOLO 라벨 파일에 그 품목 줄이 있는가다.
torch 없이 시험할 수 있게 계산만 여기 두고, 학습은 `ml/scripts/train_presence.py` 가 한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..eval.fusion import average_precision
from ..eval.scores import read_yolo_classes


def multi_hot(present: set[int], n_classes: int) -> list[float]:
    if any(not 0 <= c < n_classes for c in present):
        raise ValueError(f"클래스 번호가 범위 밖이다: {sorted(present)}")
    return [float(i in present) for i in range(n_classes)]


def letterbox_shape(height: int, width: int, size: int) -> tuple[int, int, int, int]:
    """
    긴 변을 size 에 맞춘 크기와 가운데 정렬 여백 (새 높이, 새 너비, 위 여백, 왼쪽 여백).
    E1 의 dense 점수와 같은 정사각 letterbox 규칙이다.
    """
    if height <= 0 or width <= 0 or size <= 0:
        raise ValueError(f"크기가 잘못됐다: {height}x{width}, size {size}")
    scale = min(size / height, size / width)
    nh, nw = max(1, round(height * scale)), max(1, round(width * scale))
    return nh, nw, (size - nh) // 2, (size - nw) // 2


def split_samples(data: Path, split: str) -> list[tuple[Path, set[int]]]:
    """분할의 (사진 경로, 정답 클래스 번호 집합). 사진 이름 순."""
    images = sorted((data / "images" / split).glob("*.png"))
    if not images:
        raise ValueError(f"{data / 'images' / split} 에 사진이 없다")
    return [(p, read_yolo_classes(data / "labels" / split / f"{p.stem}.txt")) for p in images]


def item_ap(labels: Sequence[Sequence[float]], probs: Sequence[Sequence[float]], names: Sequence[str]) -> dict[str, float]:
    """품목별 이미지 AP. 양성이 없는 품목은 빠진다."""
    if len(labels) != len(probs):
        raise ValueError("정답과 확률의 사진 수가 다르다")
    out = {}
    for i, name in enumerate(names):
        y = [int(row[i]) for row in labels]
        if sum(y):
            out[name] = average_precision(y, [row[i] for row in probs])
    return out
