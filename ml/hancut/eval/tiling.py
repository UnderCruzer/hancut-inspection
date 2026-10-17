"""Deterministic overlapping image windows; coordinates are original-image pixels."""
from __future__ import annotations

import math


def tile_windows(width: int, height: int, fraction: float = 0.6) -> list[tuple[int, int, int, int]]:
    """Four corner windows covering the full image, with overlap and no padding."""
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    if not 0.5 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0.5, 1] to cover the image")
    tw, th = math.ceil(width * fraction), math.ceil(height * fraction)
    return list(dict.fromkeys(
        (x, y, x + tw, y + th)
        for y in (0, height - th) for x in (0, width - tw)
    ))


def restore_box(box, window):
    """Translate crop-pixel xyxy coordinates, clipping to the crop's bounds."""
    x, y, right, bottom = window
    x1, y1, x2, y2 = box
    if x2 < x1 or y2 < y1:
        raise ValueError("inverted box")
    return (min(right, max(x, x1 + x)), min(bottom, max(y, y1 + y)),
            min(right, max(x, x2 + x)), min(bottom, max(y, y2 + y)))
