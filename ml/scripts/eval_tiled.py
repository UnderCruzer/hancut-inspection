#!/usr/bin/env python3
"""Paired full-image vs full+target-class tiles, with box AP and E2 input CSVs.

No training, GT-guided crops, or test-set parameter selection. Both arms use the
same full-image detections (square letterboxing, rect=False). This controlled
baseline can differ from historical Ultralytics val's rectangular batches.
GPU packages are imported only when running on the inference instance.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hancut.config import load_items
from hancut.data.yolo import class_names
from hancut.eval.scores import rows_for_image, write_predictions
from hancut.eval.tiling import restore_box, tile_windows

REPO = Path(__file__).resolve().parents[2]


def merge_predictions(full, crops, windows, targets, iou):
    """Keep non-target classes verbatim; class-aware NMS only on target boxes."""
    import numpy as np
    import torch
    from torchvision.ops import batched_nms

    target_mask = np.isin(full[:, 5], targets)
    additions = [full[target_mask]]
    for crop, window in zip(crops, windows, strict=True):
        selected = crop[np.isin(crop[:, 5], targets)].copy()
        for row in selected:
            row[:4] = restore_box(row[:4], window)
        additions.append(selected)
    candidates = np.concatenate(additions)
    if len(candidates):
        t = torch.from_numpy(candidates.astype(np.float32))
        keep = batched_nms(t[:, :4], t[:, 4], t[:, 5], iou).numpy()
        candidates = candidates[keep]
    return np.concatenate([full[~target_mask], candidates])


def read_labels(path, width, height):
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(f"Missing ground truth: {path}")
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = map(float, line.split())
        rows.append([(cx-w/2)*width, (cy-h/2)*height,
                     (cx+w/2)*width, (cy+h/2)*height, cls])
    return np.array(rows, dtype=np.float32).reshape(-1, 5)


class Metrics:
    def __init__(self, names, fixed_conf):
        import numpy as np
        self.names, self.fixed_conf = names, fixed_conf
        self.stats, self.images = [], 0
        self.counts = np.zeros((len(names), 6), dtype=np.int64)

    @staticmethod
    def match(pred, gt):
        import torch
        from ultralytics.engine.validator import BaseValidator
        from ultralytics.utils.metrics import box_iou
        p, g = torch.from_numpy(pred), torch.from_numpy(gt)
        validator = SimpleNamespace(iouv=torch.linspace(.5, .95, 10))
        return BaseValidator.match_predictions(
            validator, p[:, 5], g[:, 4], box_iou(g[:, :4], p[:, :4])
        ).numpy()

    def add(self, pred, gt):
        import numpy as np
        tp = self.match(pred, gt)
        self.stats.append((tp, pred[:, 4], pred[:, 5], gt[:, 4]))
        fixed = pred[pred[:, 4] >= self.fixed_conf]
        fixed_tp = self.match(fixed, gt)[:, 0]
        self.images += 1
        for cls in range(len(self.names)):
            g, p = gt[:, 4] == cls, fixed[:, 5] == cls
            # GT boxes, predictions, matched boxes, positive images,
            # positive images with no score, negative images with an alarm.
            self.counts[cls] += [int(g.sum()), int(p.sum()), int((fixed_tp & p).sum()),
                                 int(g.any()), int(g.any() and not (pred[:, 5] == cls).any()),
                                 int(not g.any() and p.any())]
        return fixed_tp

    def summary(self):
        import numpy as np
        from ultralytics.utils.metrics import ap_per_class
        stats = [np.concatenate([row[i] for row in self.stats]) for i in range(4)]
        result = ap_per_class(*stats, plot=False)
        ap, classes = result[5], result[6]
        per = {}
        for i, name in enumerate(self.names):
            n_gt, n_pred, tp, positive, absent, fp_images = map(int, self.counts[i])
            where = np.flatnonzero(classes == i)
            a = ap[where[0]] if len(where) else np.zeros(10)
            per[name] = dict(ap50=float(a[0]), ap50_95=float(a.mean()),
                             gt_boxes=n_gt, pred_boxes=n_pred, tp=tp,
                             recall=tp/n_gt if n_gt else None,
                             precision=tp/n_pred if n_pred else None,
                             positive_images=positive, absent_images=absent,
                             negative_images=self.images-positive, false_alarm_images=fp_images)
        return dict(images=self.images, fixed_conf=self.fixed_conf, match_iou=.5,
                    map50=float(ap[:, 0].mean()) if len(ap) else 0., per_class=per)


def evaluate_split(model, split, args, names, target_ids):
    import cv2
    import numpy as np
    import torch

    images = sorted((args.data / "images" / split).glob("*.png"))
    if args.limit:
        import random
        images = sorted(random.Random(args.seed).sample(images, min(args.limit, len(images))))
    if not images:
        raise ValueError(f"No images in {split}")
    out = args.out_dir / split
    out.mkdir(parents=True, exist_ok=False)
    (out / "image_ids.json").write_text(json.dumps([p.stem for p in images]))
    metrics = {arm: Metrics(names, args.fixed_conf) for arm in ("baseline", "tiled")}
    rows = {arm: [] for arm in metrics}
    timing = dict(full_seconds=0., tile_seconds=0., merge_seconds=0.)
    started = time.perf_counter()
    predict_args = dict(imgsz=args.imgsz, device=args.device, conf=args.conf,
                        iou=args.nms_iou, rect=False, verbose=False, max_det=300)
    with gzip.open(out / "detections.jsonl.gz", "wt") as cache:
        for start in range(0, len(images), args.batch):
            paths = images[start:start+args.batch]
            originals = [cv2.imread(str(p)) for p in paths]
            if any(im is None for im in originals):
                raise ValueError("Unreadable input image")
            torch.cuda.synchronize()
            tick = time.perf_counter()
            full_results = model.predict(originals, **predict_args)
            torch.cuda.synchronize()
            timing["full_seconds"] += time.perf_counter()-tick
            windows = [tile_windows(im.shape[1], im.shape[0], args.fraction) for im in originals]
            crops = [im[y:b, x:r].copy() for im, ws in zip(originals, windows)
                     for x, y, r, b in ws]
            tick = time.perf_counter()
            tile_results = []
            for k in range(0, len(crops), args.batch * 4):
                tile_results.extend(model.predict(crops[k:k+args.batch*4], **predict_args))
            torch.cuda.synchronize()
            timing["tile_seconds"] += time.perf_counter()-tick
            offset = 0
            for path, im, full_result, ws in zip(paths, originals, full_results, windows, strict=True):
                full = full_result.boxes.data.cpu().numpy()
                crop_dets = [r.boxes.data.cpu().numpy() for r in tile_results[offset:offset+len(ws)]]
                offset += len(ws)
                tick = time.perf_counter()
                tiled = merge_predictions(full, crop_dets, ws, target_ids, args.nms_iou)
                timing["merge_seconds"] += time.perf_counter()-tick
                assert np.array_equal(full[~np.isin(full[:, 5], target_ids)],
                                      tiled[~np.isin(tiled[:, 5], target_ids)])
                gt = read_labels(args.data / "labels" / split / f"{path.stem}.txt", im.shape[1], im.shape[0])
                present = set(map(int, gt[:, 4]))
                for arm, pred in (("baseline", full), ("tiled", tiled)):
                    metrics[arm].add(pred, gt)
                    rows[arm].extend(rows_for_image(path.stem, [(int(p[5]), float(p[4])) for p in pred], present, names))
                cache.write(json.dumps(dict(image_id=path.stem, gt=gt.tolist(),
                                             baseline=full.tolist(), tiled=tiled.tolist()))+"\n")
            if start % (args.batch*20) == 0 or start+args.batch >= len(images):
                print(f"{split}: {min(start+args.batch, len(images))}/{len(images)}, {time.perf_counter()-started:.1f}s", flush=True)
    report = dict(split=split, arms={arm: metric.summary() for arm, metric in metrics.items()},
                  timing=timing, total_seconds=time.perf_counter()-started)
    for arm in rows:
        write_predictions(sorted(rows[arm], key=lambda r: (r["image_id"], r["item"])),
                          args.out_dir / arm / f"{split}.csv")
    (out / "metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({arm: {name: report["arms"][arm]["per_class"][name] for name in args.targets}
                      for arm in metrics}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=REPO / "data/yolo")
    parser.add_argument("--splits", nargs="+", choices=["calib", "val", "test_easy", "test_hard", "test_hidden"],
                        default=["calib", "test_hidden", "test_easy", "test_hard"])
    parser.add_argument("--targets", nargs="+", default=["Lighter", "Bullet"])
    parser.add_argument("--fraction", type=float, default=.6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=.001)
    parser.add_argument("--fixed-conf", type=float, default=.25)
    parser.add_argument("--nms-iou", type=float, default=.7)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    tile_windows(100, 100, args.fraction)
    if args.batch <= 0 or args.limit < 0 or args.imgsz <= 0:
        parser.error("invalid batch, limit or imgsz")
    if not 0 <= args.conf <= args.fixed_conf <= 1 or not 0 < args.nms_iou <= 1:
        parser.error("invalid confidence or IoU")
    import ultralytics
    from ultralytics import YOLO
    names = class_names(load_items())
    target_ids = [names.index(name) for name in args.targets]
    model = YOLO(str(args.weights))
    if [model.names[i] for i in range(len(names))] != names:
        raise ValueError("Model class order does not match items.json")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(ultralytics=ultralytics.__version__, rect=False,
                  weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                  comparison="paired same full-image detections; target-only crop additions; exploratory PIDray")
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    results = {}
    for split in args.splits:
        (args.out_dir / "stage.txt").write_text(split+"\n")
        results[split] = evaluate_split(model, split, args, names, target_ids)
    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2)+"\n")
    (args.out_dir / "stage.txt").write_text("complete\n")


if __name__ == "__main__":
    main()
