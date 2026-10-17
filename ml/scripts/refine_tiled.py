#!/usr/bin/env python3
"""Choose tile score scaling/NMS on calib only, then apply the frozen choices.

Input is eval_tiled's cache. It retains full-image detections and crop candidates
surviving the first merge. We can recover full boxes suppressed by a crop, but
cannot recover crop candidates removed by that first NMS. This is a separate,
explicitly cache-based experiment, not a claim of rerunning raw tile inference.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_tiled import Metrics
from hancut.config import load_items
from hancut.data.yolo import class_names
from hancut.eval.scores import rows_for_image, write_predictions

CANDIDATES = [(0., .7), (.25, .5), (.5, .3), (.5, .5), (.75, .5)]


def refine(full, tiled, choices, names):
    import numpy as np
    import torch
    from torchvision.ops import nms
    ids = [names.index(name) for name in choices]
    output = [full[~np.isin(full[:, 5], ids)]]
    for name, choice in choices.items():
        cls = names.index(name)
        base = full[full[:, 5] == cls]
        if choice["weight"] == 0:
            output.append(base)
            continue
        original = {tuple(row) for row in base}
        extra = np.array([row for row in tiled if int(row[5]) == cls and tuple(row) not in original],
                         dtype=np.float32).reshape(-1, 6)
        extra[:, 4] *= choice["weight"]
        extra = extra[extra[:, 4] >= .001]
        pred = np.concatenate([base, extra])
        if len(pred):
            t = torch.from_numpy(pred)
            pred = pred[nms(t[:, :4], t[:, 4], choice["nms_iou"]).numpy()]
        output.append(pred)
    return np.concatenate(output)


def cached_rows(path):
    import numpy as np
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            yield (row["image_id"], np.asarray(row["gt"], dtype=np.float32).reshape(-1, 5),
                   np.asarray(row["baseline"], dtype=np.float32).reshape(-1, 6),
                   np.asarray(row["tiled"], dtype=np.float32).reshape(-1, 6))


def choose(candidates, targets):
    """Select per class by calib AP50; baseline is always an eligible fallback."""
    result = {}
    for name in targets:
        baseline = candidates[0]["metrics"]["per_class"][name]
        eligible = [c for c in candidates if
                    c["metrics"]["per_class"][name]["ap50_95"] >= baseline["ap50_95"]-.005]
        best = max(eligible, key=lambda c: c["metrics"]["per_class"][name]["ap50"])
        result[name] = dict(weight=best["weight"], nms_iou=best["nms_iou"])
    return result


def select(source, out):
    names, targets = class_names(load_items()), ["Lighter", "Bullet"]
    cache = source / "calib/detections.jsonl.gz"
    records = list(cached_rows(cache))
    trials = []
    for weight, iou in CANDIDATES:
        metric = Metrics(names, .25)
        choices = {name: dict(weight=weight, nms_iou=iou) for name in targets}
        for _, gt, full, tiled in records:
            metric.add(refine(full, tiled, choices, names), gt)
        trial = dict(weight=weight, nms_iou=iou, metrics=metric.summary())
        trials.append(trial)
        print(json.dumps({"weight": weight, "nms_iou": iou,
                          "metrics": {n: trial["metrics"]["per_class"][n] for n in targets}}), flush=True)
    data = dict(choices=choose(trials, targets), candidates=trials,
                calibration_split="calib", calibration_images=len(records),
                calibration_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                selection_rule="max calib AP50 subject to AP50-95 >= baseline - 0.005; baseline fallback",
                caveat="surviving crop candidates only; exploratory existing PIDray test data")
    with out.open("x") as stream:
        json.dump(data, stream, indent=2)
    print("FROZEN_CHOICES", data["choices"], flush=True)


def evaluate(source, selection, out):
    names = class_names(load_items())
    selected = json.loads(selection.read_text())
    choices = selected["choices"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "selection.json").write_text(selection.read_text())
    results = {}
    for split in ("calib", "test_hidden", "test_easy", "test_hard"):
        metric, rows = Metrics(names, .25), []
        for image_id, gt, full, tiled in cached_rows(source / split / "detections.jsonl.gz"):
            pred = refine(full, tiled, choices, names)
            metric.add(pred, gt)
            rows.extend(rows_for_image(image_id, [(int(p[5]), float(p[4])) for p in pred],
                                       set(map(int, gt[:, 4])), names))
        results[split] = metric.summary()
        write_predictions(sorted(rows, key=lambda r: (r["image_id"], r["item"])), out / f"{split}.csv")
        print(split, json.dumps({n: results[split]["per_class"][n] for n in choices}), flush=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["select", "evaluate"])
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--selection", type=Path)
    a = p.parse_args()
    if a.mode == "select":
        select(a.source, a.out)
    else:
        if a.selection is None:
            p.error("evaluate requires --selection")
        evaluate(a.source, a.selection, a.out)
