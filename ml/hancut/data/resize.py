from __future__ import annotations

"""
640px 축소 전처리 (B안).

원본은 장당 평균 약 1.58MB인 고해상도지만 모델은 640px로 학습한다. 매 에폭마다
원본을 디코딩하면 GPU가 아니라 CPU가 병목이 되므로, **한 번만** 줄여서 저장한다.

    약 962GB (608,280장)  →  8종 서브셋 8만 장 × 약 70KB ≈ 6GB

박스 라벨도 같은 비율로 줄여야 하므로 `scale_points()` 를 함께 쓴다.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageOps

DEFAULT_MAX_SIDE = 640
DEFAULT_QUALITY = 90


@dataclass(frozen=True)
class ResizeResult:
    src: Path
    dst: Path
    scale: float
    src_bytes: int
    dst_bytes: int
    skipped: bool = False

    @property
    def saved_bytes(self) -> int:
        return self.src_bytes - self.dst_bytes


def scale_for(size: tuple[int, int], max_side: int = DEFAULT_MAX_SIDE) -> float:
    """긴 변이 max_side 가 되는 배율. 이미 작으면 1.0 — 확대하지 않는다."""
    longest = max(size)
    return 1.0 if longest <= max_side else max_side / longest


def scale_points(points: Sequence[float], scale: float) -> list[float]:
    """박스 좌표를 같은 배율로 줄인다. (x1, y1, x2, y2, ...) 평탄 목록."""
    return [round(p * scale, 2) for p in points]


def resize_image(
    src: Path,
    dst: Path,
    max_side: int = DEFAULT_MAX_SIDE,
    quality: int = DEFAULT_QUALITY,
    overwrite: bool = False,
) -> ResizeResult:
    """한 장을 줄여 JPEG로 저장한다. EXIF 회전을 먼저 적용하고 메타데이터는 버린다."""
    src_bytes = src.stat().st_size
    if dst.exists() and not overwrite:
        with Image.open(src) as probe:
            scale = scale_for(ImageOps.exif_transpose(probe).size, max_side)
        return ResizeResult(src, dst, scale, src_bytes, dst.stat().st_size, skipped=True)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image)
        scale = scale_for(image.size, max_side)
        if scale < 1.0:
            target = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(target, Image.LANCZOS)
        image.convert("RGB").save(dst, format="JPEG", quality=quality, optimize=True)

    return ResizeResult(src, dst, scale, src_bytes, dst.stat().st_size)


def resize_many(
    pairs: Iterable[tuple[Path, Path]],
    max_side: int = DEFAULT_MAX_SIDE,
    quality: int = DEFAULT_QUALITY,
    overwrite: bool = False,
) -> list[ResizeResult]:
    return [resize_image(src, dst, max_side, quality, overwrite) for src, dst in pairs]


def summarize(results: Sequence[ResizeResult]) -> dict:
    """전후 용량 — docs/data.md 의 추정치를 실측으로 바꾸는 근거."""
    src_total = sum(r.src_bytes for r in results)
    dst_total = sum(r.dst_bytes for r in results)
    return {
        "count": len(results),
        "skipped": sum(1 for r in results if r.skipped),
        "src_mb": round(src_total / 1_048_576, 2),
        "dst_mb": round(dst_total / 1_048_576, 2),
        "ratio": round(dst_total / src_total, 4) if src_total else 0.0,
        "mean_dst_kb": round(dst_total / len(results) / 1024, 1) if results else 0.0,
    }
