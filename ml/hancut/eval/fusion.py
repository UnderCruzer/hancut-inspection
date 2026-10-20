"""
여러 검출기의 품목별 이미지 점수를 하나로 합친다 (#41).

결합은 행마다 점수의 최댓값이다 — 어느 모델이든 그 품목을 높게 보면 높게 친다.
품목마다 어떤 모델 조합을 쓸지는 학습 분할의 calib 에서만 고른다. 시험셋으로 고르지 않는다.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Mapping, Sequence

Key = tuple[str, str]  # (image_id, item)


def index_rows(rows: Iterable[Mapping]) -> dict[Key, tuple[int, float]]:
    return {(r["image_id"], r["item"]): (int(r["y_true"]), float(r["score"])) for r in rows}


def check_aligned(by_model: Mapping[str, Mapping[Key, tuple[int, float]]]) -> None:
    """모든 모델의 (image_id, item, y_true) 가 같아야 한다. 다르면 무엇이 다른지 말하고 멈춘다."""
    names = list(by_model)
    ref = by_model[names[0]]
    for name in names[1:]:
        other = by_model[name]
        if ref.keys() != other.keys():
            missing = sorted(ref.keys() - other.keys())[:3]
            extra = sorted(other.keys() - ref.keys())[:3]
            raise ValueError(f"{names[0]} 와 {name} 의 행이 다르다. 빠짐 {missing}, 추가 {extra}")
        bad = [k for k in ref if ref[k][0] != other[k][0]]
        if bad:
            raise ValueError(f"{names[0]} 와 {name} 의 정답이 다르다: {sorted(bad)[:3]}")


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """
    이미지 단위 AP. 같은 점수는 한 덩어리로 처리한다 (순서에 따라 값이 바뀌지 않게).
    AP = Σ (재현율 증가분) × 그 임계값에서의 정밀도. 양성이 없으면 0.
    """
    total_pos = sum(labels)
    if total_pos == 0:
        return 0.0
    groups: dict[float, list[int]] = {}
    for y, s in zip(labels, scores):
        g = groups.setdefault(s, [0, 0])
        g[0] += y
        g[1] += 1
    tp = seen = 0
    ap = 0.0
    for s in sorted(groups, reverse=True):
        pos, n = groups[s]
        tp += pos
        seen += n
        ap += (pos / total_pos) * (tp / seen)
    return ap


def subsets(models: Sequence[str]) -> list[tuple[str, ...]]:
    """비어 있지 않은 모든 조합. 작은 조합이 먼저 — 동점이면 모델을 덜 쓰는 쪽을 고르기 위해서다."""
    return [c for k in range(1, len(models) + 1) for c in combinations(models, k)]


def fused_score(row_scores: Mapping[str, float], chosen: Iterable[str]) -> float:
    return max(row_scores[m] for m in chosen)


def select_by_ap(
    calib: Mapping[str, Mapping[Key, tuple[int, float]]],
    items: Iterable[str],
    min_gain: float = 0.001,
) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[tuple[str, ...], float]]]:
    """
    품목마다 calib AP 가 가장 큰 모델 조합을 고른다.
    더 많은 모델을 쓰는 조합은 더 작은 조합보다 AP 가 min_gain 넘게 높을 때만 고른다 (비용이 늘기 때문).
    (품목 → 조합, 품목 → 조합별 AP)
    """
    models = list(calib)
    ref = calib[models[0]]
    chosen, table = {}, {}
    for item in items:
        keys = [k for k in ref if k[1] == item]
        labels = [ref[k][0] for k in keys]
        aps = {}
        for combo in subsets(models):
            aps[combo] = average_precision(labels, [max(calib[m][k][1] for m in combo) for k in keys])
        best = None
        for combo in sorted(aps, key=len):  # 작은 조합부터
            if best is None or aps[combo] > aps[best] + (min_gain if len(combo) > len(best) else 0):
                best = combo
        chosen[item], table[item] = best, aps
    return chosen, table


def fuse(
    by_model: Mapping[str, Mapping[Key, tuple[int, float]]],
    chosen: Mapping[str, Sequence[str]],
) -> list[dict]:
    """조합에 따라 행마다 최댓값을 취한 예측 행. 순서는 (image_id, item)."""
    ref = next(iter(by_model.values()))
    rows = []
    for key in sorted(ref):
        image_id, item = key
        score = max(by_model[m][key][1] for m in chosen[item])
        rows.append({"image_id": image_id, "item": item, "y_true": ref[key][0], "score": score})
    return rows
