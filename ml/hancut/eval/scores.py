"""
검출 결과 → 품목별 이미지 점수 (E1 기준선, #23).

검출기는 박스마다 (품목, 신뢰도) 를 낸다. 평가 CLI 는 사진 × 품목마다 점수 하나를 받는다.
기준선은 **품목별 박스 신뢰도의 최댓값**이고, 그 품목 박스가 없으면 0 이다.

주의: 추론할 때 신뢰도 하한을 높이면 낮은 점수가 전부 0 으로 뭉친다. 그러면 low 임계값을
그 아래로 내릴 수 없어 놓침률을 조절하지 못한다. 하한은 0.001 처럼 낮게 둔다.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .cli import FIELDS


def image_scores(detections: Iterable[tuple[int, float]], n_classes: int) -> list[float]:
    """한 이미지의 박스 [(클래스 번호, 신뢰도), ...] → 클래스 번호 순서의 점수 목록."""
    scores = [0.0] * n_classes
    for cls, conf in detections:
        if not 0 <= cls < n_classes:
            raise ValueError(f"클래스 번호 {cls} 가 범위 밖이다 (0..{n_classes - 1})")
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"신뢰도 {conf} 가 0..1 밖이다")
        scores[cls] = max(scores[cls], float(conf))
    return scores


def rows_for_image(
    image_id: str,
    detections: Iterable[tuple[int, float]],
    present: set[int],
    names: Sequence[str],
) -> list[dict]:
    """사진 한 장 → 품목 수만큼의 예측 행. present 는 정답 라벨에 있는 클래스 번호다."""
    scores = image_scores(detections, len(names))
    return [
        {"image_id": image_id, "item": name, "y_true": int(i in present), "score": scores[i]}
        for i, name in enumerate(names)
    ]


def read_yolo_classes(label_path: Path) -> set[int]:
    """YOLO 라벨 파일에 있는 클래스 번호. 파일이 없거나 비어 있으면 빈 집합."""
    if not label_path.is_file():
        return set()
    return {int(line.split()[0]) for line in label_path.read_text().splitlines() if line.strip()}


def write_predictions(rows: Iterable[Mapping], path: Path) -> int:
    """평가 CLI(`ml/scripts/evaluate.py`)가 읽는 형식으로 쓴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "score": f"{row['score']:.6f}"})
            n += 1
    return n
