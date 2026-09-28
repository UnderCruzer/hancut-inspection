"""
판정 모델 인터페이스.

모델은 W3 이후에 붙는다. 그전까지 서버는 **가짜 결과를 만들지 않고** 503을 반환한다
(AGENTS.md — 모델이 없으면 없다고 말한다).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Prediction:
    item: str
    score: float                       # p(품목 있음)
    reasons: list[str] = field(default_factory=list)
    model_version: str = "unknown"


class Predictor(Protocol):
    """X-ray 이미지 바이트를 받아 품목과 그 품목이 있을 확률을 돌려준다."""

    model_version: str

    def predict(self, image_bytes: bytes) -> Prediction: ...
