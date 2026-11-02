"""
X-ray 합성 (threat image projection) — 물체를 다른 가방의 밀집 영역에 겹친다 (#52).

X-ray 는 물체를 통과하며 빛이 줄어 겹친 부분의 밝기가 곱으로 합쳐진다. 그래서 불투명하게 붙이지 않고
`배경 × 물체 투과율` 로 합성한다. 물체는 어노테이션 마스크로 잘라 원래 가방의 나머지를 가져오지 않는다.

MixUp(#38)과 다른 점: 사진 전체가 아니라 물체만 섞고, 위치를 어두운 영역 쪽으로 고른다.
물체와 배경 모두 학습 분할에서만 온다. 시험 · 보정 사진은 이 모듈에 들어오지 않는다.
"""

from __future__ import annotations

import random
from typing import Callable, Mapping, NamedTuple, Sequence

Rect = tuple[int, int, int, int]  # x0, y0, x1, y1 — 픽셀, x1 · y1 은 포함하지 않는다

DEFAULT_WEIGHTS = {"Lighter": 3.0, "Bullet": 2.0, "Knife": 2.0}
SCALE_RANGE = (0.75, 1.0)  # 숨김 라이터가 학습보다 작다 (#37: 박스 넓이 중앙값 1.48% vs 1.81%)
DENSITY_RANGE = (0.7, 1.0)
GRAY_RANGE = (0.15, 0.85)  # 더 밝으면 가방 밖, 더 어두우면 물체가 보이지 않아 정답을 배울 수 없다
OVERLAP_LIMIT = 0.5
CANDIDATES = 16
MIN_SIDE = 8
FEATHER_RADIUS = 1.0


class Cutout(NamedTuple):
    transmission: object  # (H, W, 3) 0..1
    alpha: object  # (H, W) 0..1, 가장자리는 부드럽다


def class_weights(names: Sequence[str], overrides: Mapping[str, float]) -> dict[int, float]:
    """품목 번호 → 뽑힐 가중치. 지정하지 않은 품목은 1 이다."""
    unknown = set(overrides) - set(names)
    if unknown:
        raise ValueError(f"모르는 품목: {sorted(unknown)}")
    if any(w <= 0 for w in overrides.values()):
        raise ValueError("가중치는 0 보다 커야 한다")
    return {i: float(overrides.get(name, 1.0)) for i, name in enumerate(names)}


def pick_class(rng: random.Random, weights: Mapping[int, float]) -> int:
    classes = sorted(weights)
    return rng.choices(classes, [weights[c] for c in classes])[0]


def overlap_fraction(a: Rect, b: Rect) -> float:
    """a 의 넓이 중 b 와 겹치는 비율."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = (a[2] - a[0]) * (a[3] - a[1])
    if w <= 0 or h <= 0 or area <= 0:
        return 0.0
    return w * h / area


def is_allowed(rect: Rect, existing: Sequence[Rect], limit: float = OVERLAP_LIMIT) -> bool:
    """어느 쪽에서 재도 겹침이 한도 이하여야 한다. 작은 물체가 큰 물체 안에 다 들어가는 경우도 막는다."""
    return all(overlap_fraction(rect, b) <= limit and overlap_fraction(b, rect) <= limit for b in existing)


def choose_position(
    rng: random.Random,
    image_size: tuple[int, int],
    size: tuple[int, int],
    existing: Sequence[Rect],
    brightness: Callable[[Rect], float],
    candidates: int = CANDIDATES,
    limit: float = OVERLAP_LIMIT,
    gray_range: tuple[float, float] = GRAY_RANGE,
) -> Rect | None:
    """
    조건에 맞는 후보 중 더 어두운 절반에서 무작위로 하나 고른다. 없으면 None.

    가장 어두운 곳만 고르면 같은 위치에 몰리고 물체가 안 보이는 사례만 늘어난다.
    """
    image_w, image_h = image_size
    w, h = size
    if w > image_w or h > image_h:
        return None
    valid = []
    for _ in range(candidates):
        x, y = rng.randint(0, image_w - w), rng.randint(0, image_h - h)
        rect = (x, y, x + w, y + h)
        if not is_allowed(rect, existing, limit):
            continue
        gray = brightness(rect)
        if gray_range[0] <= gray <= gray_range[1]:
            valid.append((gray, rect))
    if not valid:
        return None
    valid.sort()
    return rng.choice(valid[:max(1, len(valid) // 2)])[1]


def rasterize(segmentation, width: int, height: int):
    """COCO segmentation → (H, W) bool 마스크. 폴리곤은 직접 그리고, RLE 는 pycocotools 로 읽는다."""
    import numpy as np
    from PIL import Image, ImageDraw

    if isinstance(segmentation, list):
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for polygon in segmentation:
            if len(polygon) >= 6:
                draw.polygon([float(v) for v in polygon], fill=1)
        return np.asarray(canvas, dtype=bool)
    from pycocotools import mask as coco_mask

    rle = segmentation
    if isinstance(rle.get("counts"), list):
        rle = coco_mask.frPyObjects(rle, height, width)
    return coco_mask.decode(rle).astype(bool)


def cut_object(image, segmentation, pad: int = 2) -> Cutout | None:
    """(H, W, 3) uint8 사진에서 물체를 잘라낸다. 마스크가 비었거나 너무 작으면 None."""
    import numpy as np
    from PIL import Image, ImageFilter

    height, width = image.shape[:2]
    mask = rasterize(segmentation, width, height)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or xs.max() - xs.min() + 1 < MIN_SIDE or ys.max() - ys.min() + 1 < MIN_SIDE:
        return None
    x0, x1 = max(0, int(xs.min()) - pad), min(width, int(xs.max()) + 1 + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(height, int(ys.max()) + 1 + pad)
    alpha = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(FEATHER_RADIUS))
    return Cutout(image[y0:y1, x0:x1].astype(np.float32) / 255, np.asarray(alpha, dtype=np.float32) / 255)


def transform(cutout: Cutout, rng: random.Random, scale: float | None = None) -> Cutout:
    """뒤집기 · 90도 회전 · 축소. X-ray 는 방향에 민감하지 않고, 임의 각도는 보간으로 흐려진다."""
    import numpy as np
    from PIL import Image

    transmission, alpha = cutout
    if rng.random() < 0.5:
        transmission, alpha = transmission[:, ::-1], alpha[:, ::-1]
    if rng.random() < 0.5:
        transmission, alpha = transmission[::-1], alpha[::-1]
    turns = rng.randrange(4)
    transmission, alpha = np.rot90(transmission, turns), np.rot90(alpha, turns)
    scale = rng.uniform(*SCALE_RANGE) if scale is None else scale
    height, width = alpha.shape
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = Image.fromarray((np.clip(transmission, 0, 1) * 255).round().astype(np.uint8)).resize(size, Image.BILINEAR)
    soft = Image.fromarray((np.clip(alpha, 0, 1) * 255).round().astype(np.uint8)).resize(size, Image.BILINEAR)
    return Cutout(np.asarray(resized, dtype=np.float32) / 255, np.asarray(soft, dtype=np.float32) / 255)


def project(background, cutout: Cutout, x: int, y: int, density: float = 1.0):
    """배경에 물체를 곱으로 겹친 새 사진. 마스크 밖 픽셀은 원래 값 그대로다."""
    import numpy as np

    transmission, alpha = cutout
    height, width = alpha.shape
    out = background.copy()
    region = out[y:y + height, x:x + width].astype(np.float32)
    attenuation = 1 - alpha[..., None] * (1 - np.clip(transmission, 0, 1) ** density)
    out[y:y + height, x:x + width] = np.clip(region * attenuation, 0, 255).round().astype(np.uint8)
    return out


def object_box(cutout: Cutout, x: int, y: int) -> Rect | None:
    """붙인 위치 기준의 물체 박스. 흐림이 번진 가장자리는 넣지 않는다."""
    import numpy as np

    ys, xs = np.nonzero(cutout.alpha >= 0.5)
    if len(xs) == 0 or xs.max() - xs.min() + 1 < MIN_SIDE or ys.max() - ys.min() + 1 < MIN_SIDE:
        return None
    return (x + int(xs.min()), y + int(ys.min()), x + int(xs.max()) + 1, y + int(ys.max()) + 1)


def compose(background, existing: Sequence[Rect], objects: Sequence[tuple[int, Cutout]], rng: random.Random):
    """
    물체들을 차례로 합성한다. (합성한 사진, [(품목 번호, 박스), ...]).

    놓을 곳이 없는 물체는 건너뛴다 — 라벨 없는 물체가 사진에 남지 않게 박스를 먼저 정한다.
    """
    image = background
    height, width = background.shape[:2]
    occupied = list(existing)
    gray = background.astype("float32").mean(axis=2) / 255
    pasted = []
    for class_index, cutout in objects:
        size = (cutout.alpha.shape[1], cutout.alpha.shape[0])
        rect = choose_position(rng, (width, height), size, occupied,
                               lambda r: float(gray[r[1]:r[3], r[0]:r[2]].mean()))
        if rect is None:
            continue
        box = object_box(cutout, rect[0], rect[1])
        if box is None:
            continue
        image = project(image, cutout, rect[0], rect[1], rng.uniform(*DENSITY_RANGE))
        occupied.append(box)
        pasted.append((class_index, box))
    return image, pasted
