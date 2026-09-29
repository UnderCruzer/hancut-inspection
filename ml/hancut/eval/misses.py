"""
미검출 분석 (#31) — 품목이 있는데 검출기가 그 품목 박스를 내지 않은 경우, 그 자리에서 무엇을 했나.

박스는 YOLO 형식(중심 x, 중심 y, 너비, 높이 — 이미지 크기로 나눈 0..1)으로 다룬다.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

WRONG_CLASS = "다른 품목으로 착각"
LOW_SCORE = "같은 품목, 점수 부족"
MISPLACED = "위치 어긋남"
NOTHING = "아무것도 없음"
CATEGORIES = (WRONG_CLASS, LOW_SCORE, MISPLACED, NOTHING)


@dataclass(frozen=True)
class Pred:
    cls: int
    conf: float
    box: tuple[float, float, float, float]  # cx, cy, w, h (0..1)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """두 박스(cx, cy, w, h)의 IoU."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def classify_miss(
    gt_box: Sequence[float],
    target: int,
    preds: Iterable[Pred],
    floor: float = 0.001,
    match_iou: float = 0.5,
    near_iou: float = 0.1,
) -> tuple[str, Pred | None, float]:
    """
    정답 박스 하나가 왜 놓쳤는지 분류한다.

    같은 품목이면서 충분히 겹치는 예측이 있으면 그것을 먼저 본다 — 그 신뢰도가 원래 하한(floor)
    아래였다면 '점수 부족'이다. 없으면 가장 많이 겹치는 예측으로 착각·어긋남·없음을 가른다.
    """
    preds = list(preds)
    same = [(iou(gt_box, p.box), p) for p in preds if p.cls == target]
    same = [(v, p) for v, p in same if v >= match_iou]
    if same:
        v, p = max(same, key=lambda t: (t[1].conf, t[0]))
        if p.conf < floor:
            return LOW_SCORE, p, v
        # 하한 위의 같은 품목 박스가 있었다면 애초에 미검출이 아니다 — 호출자가 잘못 넘긴 것
        raise ValueError("같은 품목 박스가 하한 위에 있다. 미검출이 아니다")

    if not preds:
        return NOTHING, None, 0.0
    v, p = max(((iou(gt_box, p.box), p) for p in preds), key=lambda t: (t[0], t[1].conf))
    if v >= match_iou:
        return (WRONG_CLASS if p.cls != target else LOW_SCORE), p, v
    if v >= near_iou:
        return MISPLACED, p, v
    return NOTHING, p, v


def shape_stats(boxes: Iterable[Sequence[float]]) -> dict:
    """박스 면적 비율(이미지 대비)과 가로세로비의 사분위수. 박스가 없으면 n=0."""
    boxes = list(boxes)
    if not boxes:
        return {"n": 0}
    area = sorted(b[2] * b[3] for b in boxes)
    aspect = sorted(b[2] / b[3] for b in boxes if b[3] > 0)

    def q(xs):
        if len(xs) == 1:
            return xs[0], xs[0], xs[0]
        qs = statistics.quantiles(xs, n=4)
        return qs[0], qs[1], qs[2]

    return {"n": len(boxes), "area": q(area), "aspect": q(aspect)}


def top_alternatives(rows: Iterable[Mapping], item: str, min_score: float = 0.25) -> tuple[dict, int]:
    """
    예측 CSV 행에서, 품목 item 이 있는데 점수가 0 인 이미지마다 점수가 가장 높은 다른 품목을 센다.
    그 점수가 min_score 미만이면 '없음'으로 센다. (결과, 미검출 이미지 수)
    """
    per: dict[str, dict] = {}
    for r in rows:
        e = per.setdefault(r["image_id"], {"target": None, "others": {}})
        if r["item"] == item:
            e["target"] = (r["y_true"], r["score"])
        else:
            e["others"][r["item"]] = r["score"]
    counts: dict[str, int] = {}
    missed = 0
    for e in per.values():
        if e["target"] is None or e["target"][0] != 1 or e["target"][1] != 0.0:
            continue
        missed += 1
        best, score = max(e["others"].items(), key=lambda kv: kv[1]) if e["others"] else ("없음", 0.0)
        key = best if score >= min_score else "없음"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])), missed
