#!/usr/bin/env python3
"""Prepare, smoke-test and run a paired MixUp fine-tuning experiment on AWS.

prepare -> train --arm smoke -> run (control, mixup, all evaluations).
Every training subprocess reloads the original E1 weights, never the other arm.
No checkpoint's saved run arguments are copied into a new run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hancut.config import load_items
from hancut.data.yolo import class_names

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "ml/configs/overlap_mixup_20261017.json"
DEFAULT_OUT = REPO / "ml/runs/overlap_mixup_20261017"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    # Monitoring must never see a partially written JSON document.
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+"\n")
    temp.replace(path)


def build_train_args(config, out, arm):
    if arm not in ("smoke", "control", "mixup"):
        raise ValueError("unknown arm")
    args = dict(config["train"])
    for key in ("save_dir", "project", "name", "model", "data", "cfg"):
        if key in args:
            raise ValueError(f"unsafe inherited training key: {key}")
    if args.get("resume") or args.get("exist_ok"):
        raise ValueError("this experiment must start fresh in new output directories")
    args.update(data=str((REPO / config["data"]).resolve()),
                project=str(out.resolve() / "train"), name=arm,
                mixup=config["arms"]["mixup" if arm == "smoke" else arm])
    if arm == "smoke":
        args.update(epochs=1, close_mosaic=0, save_period=-1)
    return args


def check_disjoint(paths_by_split):
    owner = {}
    for split, paths in paths_by_split.items():
        for path in paths:
            key = str(path.resolve())
            if key in owner:
                raise ValueError(f"source image occurs twice: {key} ({owner[key]}, {split})")
            owner[key] = split


def prepare(config_path, out):
    import ultralytics
    import yaml
    config = json.loads(config_path.read_text())
    if ultralytics.__version__ != config["ultralytics"]:
        raise ValueError("Ultralytics version changed")
    weights = REPO / config["weights"]
    if digest(weights) != config["weights_sha256"]:
        raise ValueError("initial weights changed")
    dataset = REPO / "data/yolo"
    data_config = yaml.safe_load((REPO / config["data"]).read_text())
    root = Path(data_config["path"])
    if not root.is_absolute():
        raise ValueError("dataset YAML path must be absolute")
    for split in ("train", "val"):
        if (root / data_config[split]).resolve() != (dataset / "images" / split).resolve():
            raise ValueError(f"unexpected dataset {split} directory")
    paths, metadata = {}, {}
    for split, expected in config["expected_counts"].items():
        images = sorted((dataset / "images" / split).glob("*.png"))
        if len(images) != expected:
            raise ValueError(f"{split}: expected {expected} images, got {len(images)}")
        paths[split] = images
        manifest = hashlib.sha256()
        for image in images:
            label = dataset / "labels" / split / f"{image.stem}.txt"
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(image if not image.is_file() else label)
            manifest.update(f"{image.name}\t{image.resolve()}\t".encode())
            manifest.update(label.read_bytes())
            manifest.update(b"\n")
        metadata[split] = dict(images=len(images), path_and_label_sha256=manifest.hexdigest())
    check_disjoint(paths)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", config)
    write_json(out / "preflight.json", dict(weights_sha256=digest(weights), splits=metadata,
               data_yaml_sha256=digest(REPO / config["data"]),
               distinct_source_paths=True, bag_or_near_duplicate_groups_checked=False,
               software=dict(ultralytics=ultralytics.__version__, python=sys.version),
               prepared_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    write_json(out / "status.json", dict(stage="prepared"))
    print("PREPARED", out, flush=True)


def train(out, arm):
    from ultralytics import YOLO
    config = json.loads((out / "config.json").read_text())
    weights = REPO / config["weights"]
    if digest(weights) != config["weights_sha256"]:
        raise ValueError("initial weights changed")
    args = build_train_args(config, out, arm)
    expected = out.resolve() / "train" / arm
    if expected.exists():
        raise FileExistsError(f"refusing to overwrite or implicitly resume {expected}")
    model = YOLO(str(weights))
    names = class_names(load_items())
    if [model.names[i] for i in range(len(names))] != names:
        raise ValueError("class order mismatch")
    # Model.train requires overrides['model']; discard every other inherited key.
    model.overrides = {"model": str(weights), "task": "detect"}
    started = time.monotonic()

    def verify(trainer):
        if Path(trainer.save_dir).resolve() != expected:
            raise RuntimeError(f"unexpected save_dir: {trainer.save_dir}")
        write_json(out / f"{arm}_effective_args.json", vars(trainer.args))

    def progress(trainer):
        write_json(out / "status.json", dict(stage="training", arm=arm,
                   epoch=trainer.epoch+1, epochs=trainer.epochs,
                   elapsed_seconds=time.monotonic()-started,
                   metrics=trainer.metrics, save_dir=str(trainer.save_dir)))

    model.add_callback("on_pretrain_routine_start", verify)
    model.add_callback("on_fit_epoch_end", progress)
    write_json(out / "status.json", dict(stage="starting_training", arm=arm, epochs=args["epochs"]))
    model.train(**args)
    if digest(weights) != config["weights_sha256"]:
        raise RuntimeError("initial weights changed during training")
    with (expected / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != args["epochs"]:
        raise RuntimeError("training did not complete the planned epoch count")
    result = dict(arm=arm, elapsed_seconds=time.monotonic()-started,
                  epochs=len(rows), best=str(expected / "weights/best.pt"),
                  best_sha256=digest(expected / "weights/best.pt"),
                  initial_weights_sha256=config["weights_sha256"])
    write_json(out / f"{arm}_result.json", result)
    write_json(out / "status.json", dict(stage="trained", **result))
    print("TRAIN_COMPLETE", json.dumps(result), flush=True)


def command(out, label, argv):
    print("BEGIN", label, " ".join(map(str, argv)), flush=True)
    with (out / f"{label}.log").open("x") as log:
        subprocess.run(list(map(str, argv)), cwd=REPO, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    print("END", label, flush=True)


def evaluate(out, arm):
    from ultralytics import YOLO
    from eval_detector import evaluate as eval_detector, split_yaml, build_report
    config = json.loads((out / "config.json").read_text())
    weights = (REPO / config["weights"]) if arm == "baseline" else out / "train" / arm / "weights/best.pt"
    names = class_names(load_items())
    dest = out / f"detector_{arm}"
    dest.mkdir(exist_ok=False)
    model = YOLO(str(weights))
    results = {}
    for split in ("val", "test_easy", "test_hard", "test_hidden"):
        write_json(out / "status.json", dict(stage="detector_evaluation", arm=arm, split=split))
        results[split] = eval_detector(model, split_yaml(dest / f"{split}.yaml", split, names),
                                      "0", config["eval"]["batch"], config["eval"]["imgsz"])
        results[split]["images"] = config["expected_counts"][split]
    write_json(dest / "metrics.json", results)
    (dest / "summary.md").write_text(build_report(str(weights), results, names, config["eval"]["imgsz"]))


def run(out):
    config = json.loads((out / "config.json").read_text())
    if not (out / "smoke_result.json").exists():
        raise RuntimeError("complete and inspect the smoke test before running")
    if any((out / f"train/{arm}").exists() for arm in config["arms"]):
        raise FileExistsError("main training already started; explicit recovery is required")
    py = sys.executable
    for arm in ("control", "mixup"):
        command(out, f"train_{arm}", [py, __file__, "train", "--out", out, "--arm", arm])
    for arm in ("baseline", "control", "mixup"):
        command(out, f"eval_{arm}", [py, __file__, "evaluate", "--out", out, "--arm", arm])
        weights = (REPO / config["weights"]) if arm == "baseline" else out / f"train/{arm}/weights/best.pt"
        for split in ("calib", "test_easy", "test_hard", "test_hidden"):
            write_json(out / "status.json", dict(stage="score_extraction", arm=arm, split=split))
            command(out, f"predict_{arm}_{split}", [py, REPO / "ml/scripts/predict_scores.py", weights, split,
                    "--out", out / f"predictions_{arm}/{split}.csv", "--imgsz", config["eval"]["imgsz"],
                    "--batch", config["eval"]["batch"], "--conf", config["eval"]["conf"]])
        for cap in config["eval"]["caps"]:
            write_json(out / "status.json", dict(stage="operating_evaluation", arm=arm, cap=cap))
            command(out, f"e2_{arm}_{cap}", [py, REPO / "ml/scripts/e2_compare.py", "--pred-dir",
                    out / f"predictions_{arm}", "--out-dir", out / f"e2_{arm}_{cap}",
                    "--cap", cap, "--seed", config["eval"]["seed"], "--fraction", .5])
    write_json(out / "status.json", dict(stage="complete", ended_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "evaluate", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--arm", choices=("smoke", "baseline", "control", "mixup"))
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((REPO / "ml/runs").resolve()):
        parser.error("output must be a new directory inside ml/runs")
    try:
        if args.mode == "prepare":
            prepare(args.config, out)
        elif args.mode == "train":
            train(out, args.arm)
        elif args.mode == "evaluate":
            evaluate(out, args.arm)
        else:
            run(out)
            (out / "pipeline.exit").write_text("0\n")
    except Exception as exc:
        if out.exists() and args.mode != "prepare":
            write_json(out / "status.json", dict(stage="failed", mode=args.mode, arm=args.arm, error=str(exc)))
            if args.mode == "run":
                (out / "pipeline.exit").write_text("1\n")
        raise


if __name__ == "__main__":
    main()
