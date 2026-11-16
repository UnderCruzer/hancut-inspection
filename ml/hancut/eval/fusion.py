"""
여러 검출기의 품목별 이미지 점수를 하나로 합친다 (#41).

결합은 행마다 점수의 최댓값이다 — 어느 모델이든 그 품목을 높게 보면 높게 친다.
품목마다 어떤 모델 조합을 쓸지는 학습 분할의 calib 에서만 고른다. 시험셋으로 고르지 않는다.
"""

from __future__ import annotations

import math
import statistics
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


def assign(items: Iterable[str], models: Sequence[str], uses: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    """
    품목마다 모델 하나를 정해 준다 (#54). uses 에 없는 품목은 첫 모델을 쓴다.
    최댓값으로 섞지 않고 바꿔 끼운다 — 작은 품목만 전용 모델 점수로 교체할 때 쓴다.
    """
    items = list(items)
    unknown_items = sorted(set(uses) - set(items))
    if unknown_items:
        raise ValueError(f"모르는 품목: {unknown_items}")
    unknown_models = sorted(set(uses.values()) - set(models))
    if unknown_models:
        raise ValueError(f"모르는 모델: {unknown_models}")
    return {item: (uses.get(item, models[0]),) for item in items}


def add(items: Iterable[str], models: Sequence[str], adds: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    """
    고른 품목만 첫 모델과 그 모델의 최댓값으로 보탠다 (#56). 나머지는 첫 모델만 쓴다.
    #54 에서 바꿔 끼우면 손해, 최댓값으로 보태면 이득이었다.
    """
    items = list(items)
    assign(items, models, adds)  # 이름 검사
    return {item: (models[0],) if adds.get(item, models[0]) == models[0] else (models[0], adds[item]) for item in items}


def _logit(p: float, eps: float = 1e-12) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def neg_z_params(rows: Mapping[Key, tuple[int, float]], items: Iterable[str]) -> dict[str, tuple[float, float]]:
    """
    품목마다 calib 음성 점수 logit 의 평균 · 표준편차 (#56).
    척도가 다른 두 모델을 '배경에 비해 얼마나 튀나' 로 맞춰 최댓값을 취하려는 것이다. calib 은 학습 분할이다.
    """
    out = {}
    for item in items:
        neg = [_logit(s) for (_, it), (y, s) in rows.items() if it == item and y == 0]
        if len(neg) < 2:
            raise ValueError(f"{item}: calib 음성이 둘 이상 있어야 척도를 맞춘다")
        sd = statistics.pstdev(neg)
        if sd == 0:
            raise ValueError(f"{item}: calib 음성 점수가 모두 같아 척도를 맞출 수 없다")
        out[item] = (statistics.fmean(neg), sd)
    return out


def neg_z(score: float, params: tuple[float, float]) -> float:
    """z = (logit − 평균) / 표준편차 를 sigmoid 로 0..1 에 되돌린다. 순서는 바뀌지 않는다."""
    mean, sd = params
    return _sigmoid((_logit(score) - mean) / sd)


def normalize(
    by_model: Mapping[str, Mapping[Key, tuple[int, float]]],
    params: Mapping[str, Mapping[str, tuple[float, float]]],
    chosen: Mapping[str, Sequence[str]],
) -> dict[str, dict[Key, tuple[int, float]]]:
    """모델을 둘 이상 섞는 품목만 척도를 맞춘다. 한 모델만 쓰는 품목은 점수를 그대로 둔다."""
    return {m: {k: (y, neg_z(s, params[m][k[1]]) if len(chosen[k[1]]) > 1 else s) for k, (y, s) in rows.items()}
            for m, rows in by_model.items()}


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
