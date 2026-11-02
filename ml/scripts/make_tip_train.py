#!/usr/bin/env python3
"""학습 사진에 X-ray 합성 사진을 더한 학습셋 (#52).

    python3 ml/scripts/make_tip_train.py                      # 합성 12,500장
    python3 ml/scripts/make_tip_train.py --count 200 --name train_tipcheck --preview 24   # 먼저 눈으로 확인
    python3 ml/scripts/make_tip_train.py --overwrite          # 이미 있으면 다시 만든다

make_yolo_dataset.py 가 만든 data/yolo 의 train 옆에 train_tip 을 만든다. 원본은 링크로 걸고, 합성 사진만 새로 쓴다.
물체는 xray_train.json 의 마스크로 자르고, 물체 · 배경 모두 train 분할에서만 고른다.
val · calib · 시험셋은 건드리지 않고, pidray_tip.yaml 은 val 을 원래 것 그대로 가리킨다.

사진마다 난수를 seed 와 번호로 따로 만들어 프로세스 수가 달라도 결과가 같다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hancut.config import load_items  # noqa: E402
from hancut.data import tip, yolo  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "yolo"
CANDIDATES = (REPO / "data" / "pidray" / "pidray", REPO / "data" / "pidray")
OTHER_SPLITS = ("val", "calib", "test_easy", "test_hard", "test_hidden")
DEFAULT_WEIGHTS = tuple(f"{k}={v:g}" for k, v in tip.DEFAULT_WEIGHTS.items())

_state: dict = {}  # 프로세스마다 한 번 채운다. 사진마다 물체 목록을 넘기지 않으려고 둔다


def parse_weights(specs, names: list[str]) -> dict[int, float]:
    overrides = {}
    for spec in specs:
        name, sep, value = spec.partition("=")
        try:
            overrides[name] = float(value) if sep else float("nan")
        except ValueError:
            overrides[name] = float("nan")
        if overrides[name] != overrides[name]:
            raise SystemExit(f"'{spec}' 를 읽지 못했다. 형식은 품목=가중치")
    try:
        return tip.class_weights(names, overrides)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def find_root(explicit: Path | None) -> Path:
    for c in ((explicit,) if explicit else CANDIDATES):
        if (c / "annotations" / "xray_train.json").is_file():
            return c
    raise SystemExit("PIDray 어노테이션을 찾지 못했다. --root 로 annotations/ 가 있는 폴더를 지정한다")


def object_pool(annotation: dict, index_map: dict, train: set[str], min_side: int = tip.MIN_SIDE) -> dict[int, list]:
    """품목 번호 → [(사진 이름, segmentation)]. train 분할의, 마스크가 있고 충분히 큰 물체만."""
    images = {img["id"]: img for img in annotation["images"]}
    pool: dict[int, list] = {}
    for a in annotation["annotations"]:
        stem = Path(images[a["image_id"]]["file_name"]).stem
        if stem not in train or a.get("iscrowd") or not a.get("segmentation"):
            continue
        if min(a["bbox"][2], a["bbox"][3]) < min_side:
            continue
        pool.setdefault(index_map[a["category_id"]], []).append((stem, a["segmentation"]))
    return pool


def read_rects(path: Path, width: int, height: int) -> list[tip.Rect]:
    rects = []
    for line in path.read_text().splitlines():
        if line.strip():
            _, cx, cy, w, h = (float(v) for v in line.split())
            rects.append((round((cx - w / 2) * width), round((cy - h / 2) * height),
                          round((cx + w / 2) * width), round((cy + h / 2) * height)))
    return rects


def _init(state: dict) -> None:
    _state.update(state)


def _load(stem: str):
    import numpy as np
    from PIL import Image

    with Image.open(_state["images"] / f"{stem}.png") as image:
        return np.asarray(image.convert("RGB"))


def build_one(number: int) -> dict:
    from PIL import Image

    rng = random.Random(f"{_state['seed']}:{number}")
    stem = rng.choice(_state["stems"])
    background = _load(stem)
    height, width = background.shape[:2]
    objects = []
    for _ in range(rng.randint(*_state["objects"])):
        class_index = tip.pick_class(rng, _state["weights"])
        source, segmentation = rng.choice(_state["pool"][class_index])
        if source == stem:
            continue
        cutout = tip.cut_object(_load(source), segmentation)
        if cutout is not None:
            objects.append((class_index, tip.transform(cutout, rng)))
    labels = _state["labels"] / f"{stem}.txt"
    image, pasted = tip.compose(background, read_rects(labels, width, height), objects, rng)
    if not pasted:
        return {"name": None, "pasted": []}

    name = f"tip_{number:06d}"
    Image.fromarray(image).save(_state["out_images"] / f"{name}.png", compress_level=1)
    lines = [line for line in labels.read_text().splitlines() if line.strip()]
    for class_index, (x0, y0, x1, y1) in pasted:
        lines.append(yolo.to_yolo((x0, y0, x1 - x0, y1 - y0), width, height, class_index).line())
    (_state["out_labels"] / f"{name}.txt").write_text("\n".join(lines) + "\n")
    return {"name": name, "background": stem, "pasted": pasted}


def write_preview(folder: Path, images: Path, results: list[dict], n: int) -> None:
    """합성한 물체에 사각형을 그린 사진. data/ 아래라 커밋되지 않는다."""
    from PIL import Image, ImageDraw

    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for result in results[:n]:
        with Image.open(images / f"{result['name']}.png") as image:
            canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for _, rect in result["pasted"]:
            draw.rectangle(rect, outline=(255, 0, 0), width=2)
        canvas.save(folder / f"{result['name']}.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=12500, help="합성 시도 수")
    parser.add_argument("--weights", nargs="+", default=list(DEFAULT_WEIGHTS), help="품목=가중치, 나머지는 1")
    parser.add_argument("--objects", type=int, nargs=2, default=(1, 3), metavar=("MIN", "MAX"), help="사진당 물체 수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--root", type=Path, help="PIDray 폴더 (annotations/ 가 들어 있는 곳)")
    parser.add_argument("--name", default="train_tip")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preview", type=int, default=24, help="사각형을 그려 볼 사진 수")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.name.startswith("train_"):
        # 원래 split 을 지우지 않도록 이름으로 막는다
        raise SystemExit(f"--name 은 train_ 으로 시작해야 한다: {args.name}")
    if args.count < 1 or args.objects[0] < 1 or args.objects[1] < args.objects[0]:
        raise SystemExit("--count 와 --objects 를 확인한다")

    items = load_items()
    names, index_map = yolo.class_names(items), yolo.class_index_map(items)
    weights = parse_weights(args.weights, names)

    src_img, src_lbl = args.data / "images" / "train", args.data / "labels" / "train"
    labels = sorted(src_lbl.glob("*.txt"))
    if not labels:
        raise SystemExit(f"{src_lbl} 에 라벨이 없다. make_yolo_dataset.py 를 먼저 돌린다")
    stems = [p.stem for p in labels]
    train = set(stems)
    for split in OTHER_SPLITS:
        leaked = train & {p.stem for p in (args.data / "labels" / split).glob("*.txt")}
        if leaked:
            raise SystemExit(f"train 과 {split} 에 같은 사진이 있다: {sorted(leaked)[:3]}")

    annotation_path = find_root(args.root) / "annotations" / "xray_train.json"
    pool = object_pool(json.loads(annotation_path.read_text(encoding="utf-8")), index_map, train)
    missing = [names[i] for i in range(len(names)) if i not in pool]
    if missing:
        raise SystemExit(f"마스크가 있는 물체가 없는 품목: {missing}. 어노테이션을 확인한다")

    out_img, out_lbl = args.data / "images" / args.name, args.data / "labels" / args.name
    if out_img.exists() or out_lbl.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_img} 가 이미 있다. 다시 만들려면 --overwrite")
        shutil.rmtree(out_img, ignore_errors=True)
        shutil.rmtree(out_lbl, ignore_errors=True)
        (args.data / "labels" / f"{args.name}.cache").unlink(missing_ok=True)
    out_img.mkdir(parents=True)
    out_lbl.mkdir(parents=True)

    for path in labels:
        src = (src_img / f"{path.stem}.png").resolve()
        if not src.is_file():
            raise SystemExit(f"이미지가 없다: {src_img / path.stem}.png")
        os.symlink(src, out_img / f"{path.stem}.png")
        (out_lbl / path.name).write_text(path.read_text())

    state = {"seed": args.seed, "stems": stems, "pool": pool, "weights": weights, "objects": tuple(args.objects),
             "images": src_img, "labels": src_lbl, "out_images": out_img, "out_labels": out_lbl}
    numbers = range(args.count)
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(state,)) as executor:
            results = list(executor.map(build_one, numbers, chunksize=32))
    else:
        _init(state)
        results = [build_one(n) for n in numbers]

    made = [r for r in results if r["name"]]
    pasted = Counter(c for r in made for c, _ in r["pasted"])
    print(f"# X-ray 합성 학습셋 (#52)\n\n`{out_img}` — 원본 {len(stems):,} + 합성 {len(made):,}장 "
          f"(시도 {args.count:,}, 놓을 곳이 없어 빠짐 {args.count - len(made):,})\n")
    print("| 품목 | 가중치 | 물체 후보 | 합성한 물체 |\n|---|---:|---:|---:|")
    for i, n in enumerate(names):
        print(f"| {n} | {weights[i]:g} | {len(pool[i]):,} | {pasted[i]:,} |")

    yaml = [f"path: {args.data.resolve()}", f"train: images/{args.name}", "val: images/val",
            "test: images/test_hidden", "names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    out_yaml = args.data / f"pidray_{args.name.removeprefix('train_')}.yaml"
    out_yaml.write_text("\n".join(yaml) + "\n", encoding="utf-8")

    manifest = {"seed": args.seed, "count": args.count, "made": len(made), "originals": len(stems),
                "objects": list(args.objects), "weights": {names[i]: w for i, w in weights.items()},
                "pasted": {names[i]: pasted[i] for i in range(len(names))},
                "scale_range": tip.SCALE_RANGE, "density_range": tip.DENSITY_RANGE, "gray_range": tip.GRAY_RANGE,
                "overlap_limit": tip.OVERLAP_LIMIT,
                "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
                "train_stems_sha256": hashlib.sha256("\n".join(stems).encode()).hexdigest()}
    manifest_path = args.data / f"{args.name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    preview = args.data / f"{args.name}_preview"
    write_preview(preview, out_img, made, args.preview)
    print(f"\n`{out_yaml}`, `{manifest_path}` 를 썼다. 미리보기: `{preview}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
