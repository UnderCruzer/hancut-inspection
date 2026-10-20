"""
NMS 전 품목 확률로 이미지 점수 만들기 (#43).

YOLO 의 후처리(NMS, multi_label=False)는 앵커마다 가장 높은 품목 하나만 남긴다. 그래서 정답 자리에서
총기 0.3 · 펜치 0.4 이면 펜치 박스만 남고 총기 점수는 0 이 된다(#31 의 '다른 품목으로 착각').
dense 점수는 NMS 전 모든 앵커에서 그 품목 확률의 최댓값이라 버려지지 않는다.

행렬 연산은 스크립트에서 torch 로 하고, 여기에는 검사와 요약만 둔다 (torch 없이 테스트한다).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def class_maxima(class_probs: Sequence[Sequence[float]]) -> list[float]:
    """[품목][앵커] 확률 → 품목별 최댓값."""
    return [max(row) if len(row) else 0.0 for row in class_probs]


def dominance_violations(dense: Iterable[Mapping], nms: Iterable[Mapping], tol: float = 1e-4) -> list[tuple]:
    """
    dense 점수는 같은 forward 의 nms 점수보다 작을 수 없다 (NMS 는 앵커를 덜어낼 뿐이다).
    어긴 (image_id, item, dense, nms) 목록. 비어 있어야 정상이다.
    """
    by_key = {(r["image_id"], r["item"]): float(r["score"]) for r in nms}
    out = []
    for r in dense:
        key = (r["image_id"], r["item"])
        if key not in by_key:
            raise ValueError(f"nms 쪽에 없는 행: {key}")
        if float(r["score"]) + tol < by_key[key]:
            out.append((*key, float(r["score"]), by_key[key]))
    return out


def zero_rate_by_item(rows: Iterable[Mapping]) -> dict[str, tuple[int, int]]:
    """품목별 (점수 0 인 양성 수, 양성 수) — p₀ 의 분자와 분모."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        if int(r["y_true"]) == 1:
            c = counts[r["item"]]
            c[1] += 1
            if float(r["score"]) == 0.0:
                c[0] += 1
    return {k: (v[0], v[1]) for k, v in sorted(counts.items())}


def agreement(a: Iterable[Mapping], b: Iterable[Mapping], tol: float = 1e-3) -> tuple[int, int]:
    """두 예측에서 점수 차가 tol 이하인 행 수와 전체 행 수. 같은 규칙의 재현 확인용."""
    by_key = {(r["image_id"], r["item"]): float(r["score"]) for r in b}
    same = total = 0
    for r in a:
        key = (r["image_id"], r["item"])
        if key in by_key:
            total += 1
            same += abs(float(r["score"]) - by_key[key]) <= tol
    return same, total
