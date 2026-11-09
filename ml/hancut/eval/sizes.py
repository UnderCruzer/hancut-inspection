"""
물체 크기 구간별 재현율 (#54) — 같은 품목 안에서 작은 물체일수록 더 놓치는가.

크기는 박스 넓이의 제곱근(한 변)을 640 입력 기준 픽셀로 잰다. 사진은 높이 448 · 너비가 제각각이라
이미지 비율로 재면 같은 물체도 사진마다 크기가 달라진다. 모델 입력이 960 이어도 구간은 640 기준으로 같게 둔다.
박스는 YOLO 형식(중심 x, 중심 y, 너비, 높이 — 0..1)이다.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from .misses import Pred, iou

REFERENCE = 640
DEFAULT_EDGES = (32, 64, 96)


def side_px(box: Sequence[float], width: int, height: int, reference: int = REFERENCE) -> float:
    """긴 변을 reference 로 맞춘 입력에서 박스 한 변(넓이의 제곱근)의 픽셀 길이."""
    scale = reference / max(width, height)
    return math.sqrt(box[2] * width * scale * box[3] * height * scale)


def bin_names(edges: Sequence[float] = DEFAULT_EDGES) -> list[str]:
    edges = list(edges)
    if not edges or edges != sorted(set(edges)):
        raise ValueError("구간 경계는 비어 있지 않고 오름차순이어야 한다")
    names = [f"<{edges[0]:g}"] + [f"{a:g}~{b:g}" for a, b in zip(edges, edges[1:])] + [f"≥{edges[-1]:g}"]
    return names


def bin_of(side: float, edges: Sequence[float] = DEFAULT_EDGES) -> str:
    """경계값은 위 구간에 넣는다 (32 는 32~64)."""
    names = bin_names(edges)
    for name, edge in zip(names, edges):
        if side < edge:
            return name
    return names[-1]


def match(gts: Sequence[tuple[int, Sequence[float]]], preds: Iterable[Pred], conf: float, iou_thr: float = 0.5) -> list[bool]:
    """
    정답 박스마다 맞혔는지. 신뢰도 높은 예측부터, 같은 품목의 아직 안 맞힌 정답 중 IoU 가 가장 큰 것과 짝짓는다.
    예측 하나는 정답 하나만 맞힌다 — mAP 계산과 같은 규칙이다.
    """
    hit = [False] * len(gts)
    for p in sorted((p for p in preds if p.conf >= conf), key=lambda p: -p.conf):
        best, best_iou = None, 0.0
        for k, (cls, box) in enumerate(gts):
            if hit[k] or cls != p.cls:
                continue
            v = iou(box, p.box)
            if v >= iou_thr and (best is None or v > best_iou):  # 동점이면 앞의 정답
                best, best_iou = k, v
        if best is not None:
            hit[best] = True
    return hit


def recall_table(records: Iterable[tuple[str, str, bool]]) -> dict[str, dict[str, tuple[int, int]]]:
    """(품목, 구간, 맞힘) → {품목: {구간: (맞힌 수, 정답 수)}}."""
    table: dict[str, dict[str, list[int]]] = {}
    for item, name, ok in records:
        cell = table.setdefault(item, {}).setdefault(name, [0, 0])
        cell[0] += bool(ok)
        cell[1] += 1
    return {item: {name: (h, n) for name, (h, n) in bins.items()} for item, bins in table.items()}
