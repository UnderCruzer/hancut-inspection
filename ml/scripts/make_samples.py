#!/usr/bin/env python3
"""발표용 데이터셋 예시 사진 (#35).

    python3 ml/scripts/make_samples.py

data/yolo 의 정답 라벨로 두 가지를 만든다. 출력은 ml/runs/samples/ — 커밋하지 않는다 (PIDray 재배포 금지).

    items/<품목>.jpg       품목 12종마다 정답 박스 주변을 잘라낸 한 장 (시험 easy 에서 고른다)
    items_grid.jpg         그 12장을 6 × 2 로 붙인 것
    difficulty/<난이도>_<n>.jpg   easy · hard · hidden 각각 몇 장, 정답 박스와 품목 이름을 그린 전체 사진
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hancut.config import load_items  # noqa: E402
from hancut.data import yolo  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "yolo"
DIFFICULTIES = ("easy", "hard", "hidden")
COLORS = ["#C8412D", "#E8762C", "#D99A2B", "#2E8B57", "#2563EB", "#7C3AED",
          "#DB2777", "#0891B2", "#65A30D", "#B45309", "#475569", "#9333EA"]
FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf")


def read_labels(path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            c, *xywh = line.split()
            out.append((int(c), tuple(float(v) for v in xywh)))
    return out


def crop_region(box, width: int, height: int, pad: float = 0.35) -> tuple[int, int, int, int]:
    """정답 박스(YOLO 0..1) 주변을 pad 비율만큼 넓힌 정사각형에 가까운 영역, 픽셀 (x0, y0, x1, y1)."""
    cx, cy, w, h = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    side = max(w, h) * (1 + 2 * pad)
    x0, y0 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    return x0, y0, min(width, int(cx + side / 2)), min(height, int(cy + side / 2))


def pick_gallery(labels: dict[str, list], n_classes: int, seed: int = 0) -> dict[int, tuple[str, tuple]]:
    """품목마다 한 장: 그 품목 박스가 하나뿐이고, 박스 넓이가 그 품목의 중간쯤인 사진을 고른다."""
    by_class: dict[int, list[tuple[float, str, tuple]]] = {c: [] for c in range(n_classes)}
    for stem, boxes in sorted(labels.items()):
        for c in {b[0] for b in boxes}:
            own = [b for k, b in boxes if k == c]
            if len(own) == 1:
                by_class[c].append((own[0][2] * own[0][3], stem, own[0]))
    rng = random.Random(seed)
    out = {}
    for c, cands in by_class.items():
        if not cands:
            continue
        cands.sort()
        mid = cands[len(cands) // 4: max(len(cands) * 3 // 4, len(cands) // 4 + 1)]
        _, stem, box = rng.choice(mid)
        out[c] = (stem, box)
    return out


def pick_examples(stems: list[str], k: int, seed: int = 0) -> list[str]:
    return sorted(random.Random(seed).sample(sorted(stems), min(k, len(stems))))


def font(size: int):
    from PIL import ImageFont

    for p in FONT_PATHS:
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def draw_boxes(img, boxes, names):
    from PIL import ImageDraw

    d, W, H = ImageDraw.Draw(img), img.width, img.height
    f = font(max(14, W // 45))
    for c, (cx, cy, w, h) in boxes:
        x0, y0, x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
        col = COLORS[c % len(COLORS)]
        d.rectangle([x0, y0, x1, y1], outline=col, width=max(3, W // 250))
        tw, th = d.textbbox((0, 0), names[c], font=f)[2:]
        ty = y0 - th - 6 if y0 - th - 6 > 0 else y1 + 2
        d.rectangle([x0, ty, x0 + tw + 8, ty + th + 6], fill=col)
        d.text((x0 + 4, ty + 2), names[c], fill="white", font=f)
    return img


def main() -> int:
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-difficulty", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO / "ml" / "runs" / "samples")
    args = parser.parse_args()

    names = yolo.class_names(load_items())
    (args.out / "items").mkdir(parents=True, exist_ok=True)
    (args.out / "difficulty").mkdir(parents=True, exist_ok=True)

    easy = {p.stem: read_labels(p) for p in sorted((DATA / "labels" / "test_easy").glob("*.txt"))}
    if not easy:
        raise SystemExit(f"{DATA / 'labels' / 'test_easy'} 에 라벨이 없다. make_yolo_dataset.py 를 먼저 돌린다")

    tiles = []
    for c, (stem, box) in sorted(pick_gallery(easy, len(names), args.seed).items()):
        img = Image.open(DATA / "images" / "test_easy" / f"{stem}.png").convert("RGB")
        tile = img.crop(crop_region(box, img.width, img.height))
        tile.thumbnail((360, 360))
        canvas = Image.new("RGB", (360, 360), "white")
        canvas.paste(tile, ((360 - tile.width) // 2, (360 - tile.height) // 2))
        canvas.save(args.out / "items" / f"{names[c]}.jpg", quality=90)
        tiles.append(canvas)
        print(f"{names[c]:<10} {stem}")
    grid = Image.new("RGB", (360 * 6, 360 * 2), "white")
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % 6) * 360, (i // 6) * 360))
    grid.save(args.out / "items_grid.jpg", quality=90)

    for diff in DIFFICULTIES:
        lbl = DATA / "labels" / f"test_{diff}"
        stems = [p.stem for p in lbl.glob("*.txt")]
        for i, stem in enumerate(pick_examples(stems, args.per_difficulty, args.seed), 1):
            img = Image.open(DATA / "images" / f"test_{diff}" / f"{stem}.png").convert("RGB")
            img = draw_boxes(img, read_labels(lbl / f"{stem}.txt"), names)
            img.thumbnail((1000, 1000))
            img.save(args.out / "difficulty" / f"{diff}_{i}.jpg", quality=88)
            print(f"{diff}_{i:<6} {stem}")

    print(f"\n저장: {args.out}  (커밋하지 않는다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
