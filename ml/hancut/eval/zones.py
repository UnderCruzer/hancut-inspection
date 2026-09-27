from __future__ import annotations

"""
판정 3구간과 운영점 계산 (E2).

이 프로젝트의 성능은 정확도 하나로 말하지 않는다. 두 실수의 값이 다르기 때문이다.

- 미준수를 준수로 놓치면 → 불이 났을 때 그 설비가 작동하지 않는다 (놓침)
- 준수를 미준수로 잡으면 → 점검원이 한 번 더 보면 된다 (오경보)

그래서 판정을 세 구간으로 나누고, **놓침률 상한을 먼저 정한 뒤** 점검원이
확인해야 하는 비율을 최소화한다.

    p < low          자동 · 준수      (여기 섞인 미준수가 곧 '놓침')
    low <= p < high  확인 필요        (점검원이 직접 판단)
    p >= high        자동 · 미준수    (사유와 함께 반영)

규약: y_true 는 1 = 미준수, 0 = 준수. score 는 p(미준수).
"""

from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable, Sequence

AUTO_CLEAR = "auto_clear"
REVIEW = "review"
AUTO_ALARM = "auto_alarm"


@dataclass(frozen=True)
class Thresholds:
    low: float
    high: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.low <= self.high <= 1.0:
            raise ValueError(f"0 <= low <= high <= 1 이어야 한다: low={self.low}, high={self.high}")


@dataclass(frozen=True)
class OperatingPoint:
    thresholds: Thresholds
    miss_rate: float          # 미준수 중 '자동 · 준수'로 빠진 비율 — 안전 지표
    review_rate: float        # 전체 중 점검원이 확인해야 하는 비율 — 업무량 지표
    false_alarm_rate: float   # 준수 중 '자동 · 미준수'로 잡힌 비율
    auto_rate: float          # 자동 처리 비율 (1 - review_rate)
    n: int
    n_threat: int

    def as_row(self) -> dict:
        return {
            "low": round(self.thresholds.low, 4),
            "high": round(self.thresholds.high, 4),
            "miss_rate": round(self.miss_rate, 4),
            "review_rate": round(self.review_rate, 4),
            "false_alarm_rate": round(self.false_alarm_rate, 4),
            "auto_rate": round(self.auto_rate, 4),
            "n": self.n,
            "n_threat": self.n_threat,
        }


def zone_of(score: float, thresholds: Thresholds) -> str:
    if score < thresholds.low:
        return AUTO_CLEAR
    if score >= thresholds.high:
        return AUTO_ALARM
    return REVIEW


def assign_zones(scores: Iterable[float], thresholds: Thresholds) -> list[str]:
    return [zone_of(s, thresholds) for s in scores]


def _validate(y_true: Sequence[int], scores: Sequence[float]) -> None:
    if len(y_true) != len(scores):
        raise ValueError("y_true 와 scores 의 길이가 다르다")
    if not y_true:
        raise ValueError("빈 입력으로는 운영점을 계산할 수 없다")
    if any(y not in (0, 1) for y in y_true):
        raise ValueError("y_true 는 1 = 미준수, 0 = 준수 만 허용한다")
    if any(not 0.0 <= s <= 1.0 for s in scores):
        raise ValueError("score 는 p(미준수) 로 0..1 범위여야 한다")


def evaluate(y_true: Sequence[int], scores: Sequence[float], thresholds: Thresholds) -> OperatingPoint:
    """주어진 임계값에서의 놓침률·확인 필요 비율·오경보율."""
    _validate(y_true, scores)
    n = len(y_true)
    n_non = sum(y_true)
    n_com = n - n_non

    misses = sum(1 for y, s in zip(y_true, scores) if y == 1 and s < thresholds.low)
    reviews = sum(1 for s in scores if thresholds.low <= s < thresholds.high)
    false_alarms = sum(1 for y, s in zip(y_true, scores) if y == 0 and s >= thresholds.high)

    return OperatingPoint(
        thresholds=thresholds,
        miss_rate=misses / n_non if n_non else 0.0,
        review_rate=reviews / n,
        false_alarm_rate=false_alarms / n_com if n_com else 0.0,
        auto_rate=1.0 - reviews / n,
        n=n,
        n_threat=n_non,
    )


def _candidates(scores: Sequence[float]) -> list[float]:
    """임계값 후보 — 관측된 점수들과 양 끝."""
    return sorted({0.0, *scores, 1.0 + 1e-9})


def fit_thresholds(
    y_true: Sequence[int],
    scores: Sequence[float],
    max_miss_rate: float,
    max_false_alarm_rate: float = 0.05,
) -> Thresholds:
    """
    놓침률 상한을 지키는 가장 큰 low, 오경보 상한을 지키는 가장 작은 high.

    low 가 클수록 자동 처리되는 '준수'가 늘고 놓침도 늘어난다 (놓침률은 low 에 대해 단조 증가).
    high 가 작을수록 자동 처리되는 '미준수'가 늘고 오경보도 늘어난다.
    상한을 지킬 수 없으면 해당 구간을 비운다 (low=0 또는 high=1).

    high를 높이면 오경보는 감소한다. 다만 high<=1 규약 때문에 준수 점수가
    정확히 1이면 오경보 상한을 만족하지 못할 수 있다. 내보내기 전 evaluate()로
    실제 상한 충족 여부를 확인한다. 후보별 누적 개수는 이진 탐색으로 계산한다.
    """
    _validate(y_true, scores)
    if not 0.0 <= max_miss_rate <= 1.0 or not 0.0 <= max_false_alarm_rate <= 1.0:
        raise ValueError("상한은 0..1 범위여야 한다")

    n_non = sum(y_true)
    n_com = len(y_true) - n_non

    non_scores = sorted(s for y, s in zip(y_true, scores) if y == 1)
    com_scores = sorted(s for y, s in zip(y_true, scores) if y == 0)
    candidates = _candidates(scores)
    low = 0.0
    for candidate in candidates:
        misses = bisect_left(non_scores, candidate)
        miss_rate = misses / n_non if n_non else 0.0
        if miss_rate <= max_miss_rate:
            low = candidate
        else:
            break

    high = 1.0 + 1e-9
    for candidate in reversed(candidates):
        if candidate < low:
            break
        false_alarms = n_com - bisect_left(com_scores, candidate)
        rate = false_alarms / n_com if n_com else 0.0
        if rate <= max_false_alarm_rate:
            high = candidate
        else:
            break

    return Thresholds(low=min(low, 1.0), high=min(high, 1.0))


def sweep(
    y_true: Sequence[int],
    scores: Sequence[float],
    miss_rate_caps: Sequence[float] = (0.01, 0.03, 0.05),
    max_false_alarm_rate: float = 0.05,
) -> list[dict]:
    """놓침률 상한별 운영점 — E2 결과 표의 한 행씩."""
    rows = []
    for cap in miss_rate_caps:
        thresholds = fit_thresholds(y_true, scores, cap, max_false_alarm_rate)
        row = {"miss_rate_cap": cap, **evaluate(y_true, scores, thresholds).as_row()}
        rows.append(row)
    return rows


def sweep_by_item(
    items: Sequence[str],
    y_true: Sequence[int],
    scores: Sequence[float],
    miss_rate_caps: Sequence[float] = (0.01, 0.03, 0.05),
    max_false_alarm_rate: float = 0.05,
) -> dict[str, list[dict]]:
    """시설 종류별로 따로 계산한다 — 방화문과 유도등은 같은 임계값을 쓰지 않는다."""
    if not (len(items) == len(y_true) == len(scores)):
        raise ValueError("items, y_true, scores 의 길이가 모두 같아야 한다")

    grouped: dict[str, tuple[list[int], list[float]]] = {}
    for item, y, s in zip(items, y_true, scores):
        ys, ss = grouped.setdefault(item, ([], []))
        ys.append(y)
        ss.append(s)

    return {
        item: sweep(ys, ss, miss_rate_caps, max_false_alarm_rate)
        for item, (ys, ss) in sorted(grouped.items())
    }


def format_table(rows: Sequence[dict]) -> str:
    """E2 결과를 마크다운 표로 — 보고서에 그대로 붙인다."""
    header = "| 놓침률 상한 | low | high | 놓침률 | 확인 필요 | 오경보율 | 자동 처리 |"
    divider = "|---|---|---|---|---|---|---|"
    lines = [header, divider]
    for r in rows:
        lines.append(
            f"| {r['miss_rate_cap']:.0%} | {r['low']:.3f} | {r['high']:.3f} | "
            f"{r['miss_rate']:.1%} | {r['review_rate']:.1%} | {r['false_alarm_rate']:.1%} | {r['auto_rate']:.1%} |"
        )
    return "\n".join(lines)
