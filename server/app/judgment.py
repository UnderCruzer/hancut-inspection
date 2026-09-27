from __future__ import annotations

"""
판정 3구간 — ml/hancut/eval/zones.py 와 같은 규약을 서비스 쪽에서 쓴다.

임계값은 학습 쪽에서 E2로 구한 뒤 `models/thresholds.json` 으로 전달한다.
서버가 임의로 정하지 않는다.
"""

import json
from dataclasses import dataclass
from pathlib import Path

AUTO_CLEAR = "auto_clear"
REVIEW = "review"
AUTO_ALARM = "auto_alarm"


@dataclass(frozen=True)
class Thresholds:
    low: float
    high: float
    miss_rate_cap: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.low <= self.high <= 1.0:
            raise ValueError(f"0 <= low <= high <= 1 이어야 한다: low={self.low}, high={self.high}")


def zone_of(score: float, thresholds: Thresholds) -> str:
    if not 0.0 <= score <= 1.0:
        raise ValueError("score 는 p(미준수) 로 0..1 범위여야 한다")
    if score < thresholds.low:
        return AUTO_CLEAR
    if score >= thresholds.high:
        return AUTO_ALARM
    return REVIEW


def needs_review(zone: str) -> bool:
    return zone == REVIEW


def load_thresholds(path: Path) -> dict[str, Thresholds]:
    """
    시설별 임계값. 시설마다 다른 값을 쓴다 (방화문과 유도등은 같지 않다).

        {"default": {"low": 0.2, "high": 0.8},
         "방화문": {"low": 0.05, "high": 0.6, "miss_rate_cap": 0.01}}
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "default" not in raw:
        raise ValueError("thresholds.json 에 'default' 항목이 있어야 한다")
    return {
        item: Thresholds(
            low=float(values["low"]),
            high=float(values["high"]),
            miss_rate_cap=values.get("miss_rate_cap"),
        )
        for item, values in raw.items()
    }


def thresholds_for(table: dict[str, Thresholds], item: str | None) -> Thresholds:
    if item and item in table:
        return table[item]
    return table["default"]
