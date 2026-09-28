"""
판정 3구간과 운영점 계산 (E2).

이 프로젝트의 성능은 정확도나 mAP 하나로 말하지 않는다. 두 실수의 값이 다르기 때문이다.

- 위해물품이 든 가방을 통과시키면 → 그 물건이 검색대를 지나간다 (놓침)
- 빈 가방을 적발하면 → 판독관이 한 번 더 보거나 개봉하면 된다 (오경보)

그래서 판정을 세 구간으로 나누고, **놓침률 상한을 먼저 정한 뒤** 판독관이
직접 봐야 하는 비율(재검률)을 최소화한다.

    p < low          자동 · 통과      (여기 섞인 위해물품이 곧 '놓침')
    low <= p < high  재검             (판독관이 직접 확인)
    p >= high        자동 · 적발      (개봉 검사 대상)

규약: 품목 단위로 본다. y_true 는 1 = 해당 품목 있음, 0 = 없음. score 는 p(품목 있음).
"""

from __future__ import annotations

import math
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
    miss_rate: float          # 품목이 있는데 '자동 · 통과'로 빠진 비율 — 안전 지표
    review_rate: float        # 전체 중 판독관이 직접 봐야 하는 비율(재검률) — 업무량 지표
    false_alarm_rate: float   # 품목이 없는데 '자동 · 적발'로 잡힌 비율
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
        raise ValueError("y_true 는 1 = 품목 있음, 0 = 없음 만 허용한다")
    if any(not 0.0 <= s <= 1.0 for s in scores):
        raise ValueError("score 는 p(품목 있음) 으로 0..1 범위여야 한다")


def evaluate(y_true: Sequence[int], scores: Sequence[float], thresholds: Thresholds) -> OperatingPoint:
    """주어진 임계값에서의 놓침률·재검률·오경보율."""
    _validate(y_true, scores)
    n = len(y_true)
    n_pos = sum(y_true)
    n_neg = n - n_pos

    misses = sum(1 for y, s in zip(y_true, scores) if y == 1 and s < thresholds.low)
    reviews = sum(1 for s in scores if thresholds.low <= s < thresholds.high)
    false_alarms = sum(1 for y, s in zip(y_true, scores) if y == 0 and s >= thresholds.high)

    return OperatingPoint(
        thresholds=thresholds,
        miss_rate=misses / n_pos if n_pos else 0.0,
        review_rate=reviews / n,
        false_alarm_rate=false_alarms / n_neg if n_neg else 0.0,
        auto_rate=1.0 - reviews / n,
        n=n,
        n_threat=n_pos,
    )


def _candidates(scores: Sequence[float]) -> list[float]:
    """임계값 후보 — 관측된 점수들과 양 끝."""
    return sorted({0.0, *scores, 1.0 + 1e-9})


BOUNDS = ("empirical", "upper")


def allowed_errors(n: int, cap: float, delta: float = 0.05) -> int:
    """
    n 개 중 몇 개까지 틀려도 '참 비율이 cap 이하'라고 1-delta 로 말할 수 있나.

    Clopper–Pearson 한쪽 상한을 쓴다. k 개를 틀렸을 때의 상한이 cap 이하인 것은
    이항분포에서 P(X <= k | n, cap) <= delta 인 것과 같다. 그 최대 k 를 돌려준다.
    0 개를 틀려도 보장할 수 없을 만큼 n 이 작으면 -1 이다 — 그 구간은 비워야 한다.

    예: cap 1%, delta 5% 에서 0 개 틀림으로 보장하려면 n 이 299 이상이어야 한다.
    """
    if n < 0 or not 0.0 <= cap <= 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("n >= 0, 0 <= cap <= 1, 0 < delta < 1 이어야 한다")
    if n == 0 or cap >= 1.0:
        return n
    if cap <= 0.0:
        return -1
    log_delta = math.log(delta)
    log_p, log_q = math.log(cap), math.log1p(-cap)
    log_pmf = n * log_q                     # P(X = 0)
    log_cdf = log_pmf
    k = 0
    while log_cdf <= log_delta:
        if k == n:
            return n
        # P(X = k+1) = P(X = k) * (n - k) / (k + 1) * p / q
        log_pmf += math.log(n - k) - math.log(k + 1) + log_p - log_q
        k += 1
        log_cdf = log_cdf + math.log1p(math.exp(log_pmf - log_cdf)) if log_pmf <= log_cdf \
            else log_pmf + math.log1p(math.exp(log_cdf - log_pmf))
    return k - 1


def fit_thresholds(
    y_true: Sequence[int],
    scores: Sequence[float],
    max_miss_rate: float,
    max_false_alarm_rate: float = 0.05,
    *,
    bound: str = "empirical",
    delta: float = 0.05,
) -> Thresholds:
    """
    놓침률 상한을 지키는 가장 큰 low, 오경보 상한을 지키는 가장 작은 high.

    low 가 클수록 자동 통과가 늘고 놓침도 늘어난다 (놓침률은 low 에 대해 단조 증가).
    high 가 작을수록 자동 적발이 늘고 오경보도 늘어난다.
    상한을 지킬 수 없으면 해당 구간을 비운다 (low=0 또는 high=1).

    high를 높이면 오경보는 감소한다. 다만 high<=1 규약 때문에 품목 없는 사진의 점수가
    정확히 1이면 오경보 상한을 만족하지 못할 수 있다. 내보내기 전 evaluate()로
    실제 상한 충족 여부를 확인한다. 후보별 누적 개수는 이진 탐색으로 계산한다.

    bound="empirical" — 표본의 비율이 상한 이하면 된다 (기존 동작).
    bound="upper"     — 비율의 한쪽 신뢰 상한(1-delta)이 상한 이하여야 한다. 표본이 작을수록
                        보수적이 되고, 0 개 틀림으로도 보장할 수 없으면 그 구간을 비운다.
                        표본에서 딱 맞춘 기준은 다른 분포에서 넘치기 쉽다(#25).
    """
    _validate(y_true, scores)
    if not 0.0 <= max_miss_rate <= 1.0 or not 0.0 <= max_false_alarm_rate <= 1.0:
        raise ValueError("상한은 0..1 범위여야 한다")
    if bound not in BOUNDS:
        raise ValueError(f"bound 는 {BOUNDS} 중 하나여야 한다")

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if bound == "upper":
        miss_limit = allowed_errors(n_pos, max_miss_rate, delta)
        alarm_limit = allowed_errors(n_neg, max_false_alarm_rate, delta)
        miss_ok = lambda k: k <= miss_limit  # noqa: E731
        alarm_ok = lambda k: k <= alarm_limit  # noqa: E731
    else:
        miss_ok = lambda k: (k / n_pos if n_pos else 0.0) <= max_miss_rate  # noqa: E731
        alarm_ok = lambda k: (k / n_neg if n_neg else 0.0) <= max_false_alarm_rate  # noqa: E731

    non_scores = sorted(s for y, s in zip(y_true, scores) if y == 1)
    com_scores = sorted(s for y, s in zip(y_true, scores) if y == 0)
    candidates = _candidates(scores)
    low = 0.0
    for candidate in candidates:
        misses = bisect_left(non_scores, candidate)
        if miss_ok(misses):
            low = candidate
        else:
            break

    high = 1.0 + 1e-9
    for candidate in reversed(candidates):
        if candidate < low:
            break
        false_alarms = n_neg - bisect_left(com_scores, candidate)
        if alarm_ok(false_alarms):
            high = candidate
        else:
            break

    return Thresholds(low=min(low, 1.0), high=min(high, 1.0))


def sweep(
    y_true: Sequence[int],
    scores: Sequence[float],
    miss_rate_caps: Sequence[float] = (0.01, 0.02, 0.05),
    max_false_alarm_rate: float = 0.05,
    *,
    bound: str = "empirical",
    delta: float = 0.05,
) -> list[dict]:
    """놓침률 상한별 운영점 — E2 결과 표의 한 행씩."""
    rows = []
    for cap in miss_rate_caps:
        thresholds = fit_thresholds(y_true, scores, cap, max_false_alarm_rate, bound=bound, delta=delta)
        row = {"miss_rate_cap": cap, **evaluate(y_true, scores, thresholds).as_row()}
        rows.append(row)
    return rows


def sweep_by_item(
    items: Sequence[str],
    y_true: Sequence[int],
    scores: Sequence[float],
    miss_rate_caps: Sequence[float] = (0.01, 0.02, 0.05),
    max_false_alarm_rate: float = 0.05,
    *,
    bound: str = "empirical",
    delta: float = 0.05,
) -> dict[str, list[dict]]:
    """품목별로 따로 계산한다 — 총기와 라이터는 같은 임계값을 쓰지 않는다."""
    if not (len(items) == len(y_true) == len(scores)):
        raise ValueError("items, y_true, scores 의 길이가 모두 같아야 한다")

    grouped: dict[str, tuple[list[int], list[float]]] = {}
    for item, y, s in zip(items, y_true, scores):
        ys, ss = grouped.setdefault(item, ([], []))
        ys.append(y)
        ss.append(s)

    return {
        item: sweep(ys, ss, miss_rate_caps, max_false_alarm_rate, bound=bound, delta=delta)
        for item, (ys, ss) in sorted(grouped.items())
    }


def format_table(rows: Sequence[dict]) -> str:
    """E2 결과를 마크다운 표로 — 보고서에 그대로 붙인다."""
    header = "| 놓침률 상한 | low | high | 놓침률 | 재검률 | 오경보율 | 자동 처리 |"
    divider = "|---|---|---|---|---|---|---|"
    lines = [header, divider]
    for r in rows:
        lines.append(
            f"| {r['miss_rate_cap']:.0%} | {r['low']:.3f} | {r['high']:.3f} | "
            f"{r['miss_rate']:.1%} | {r['review_rate']:.1%} | {r['false_alarm_rate']:.1%} | {r['auto_rate']:.1%} |"
        )
    return "\n".join(lines)
