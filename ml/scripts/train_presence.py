#!/usr/bin/env python3
"""사진 전체를 보고 품목마다 있을 확률을 내는 분류기를 학습하고 점수를 뽑는다 (#56).

    python3 ml/scripts/train_presence.py --config ml/configs/presence_20261116.json \\
        --out ml/runs/presence_20261116/cls

학습은 train, 에폭 선택은 val (12종 평균 이미지 AP) 만 쓴다. calib · 시험셋은 학습이 끝난 뒤 점수만 뽑는다.
출력:
    <out>/best.pt                 val 평균 AP 가 가장 높은 에폭의 가중치 (state_dict)
    <out>/epochs.csv              에폭별 학습 손실 · val 평균 AP
    <out>/scores/<split>.csv      calib · test_easy · test_hard · test_hidden — e2_compare.py · fuse_scores.py 형식
    <out>/summary.md, manifest.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hancut.config import load_items  # noqa: E402
from hancut.data import presence, yolo  # noqa: E402
from hancut.eval import scores  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "yolo"
SCORE_SPLITS = ("calib", "test_easy", "test_hard", "test_hidden")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet


def digest(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def load_image(path: Path, size: int, flip: bool):
    import cv2
    import numpy as np
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"읽지 못했다: {path}")
    h, w = img.shape[:2]
    nh, nw, top, left = presence.letterbox_shape(h, w, size)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if flip:
        canvas = canvas[:, ::-1]
    x = canvas[..., ::-1].astype(np.float32) / 255  # BGR → RGB
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def build_model(config: dict, n_classes: int):
    import torch
    from torchvision import models
    if config["model"] != "efficientnet_b0":
        raise SystemExit(f"모르는 모델: {config['model']}")
    net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights[config["pretrained"]])
    net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, n_classes)
    return net


def predict(net, samples, config, device, loader_kwargs):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class Images(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            return torch.from_numpy(load_image(samples[i][0], config["imgsz"], False))

    net.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        for x in DataLoader(Images(), shuffle=False, **loader_kwargs):
            out += torch.sigmoid(net(x.to(device, non_blocking=True)).float()).cpu().tolist()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=0, help="분할마다 앞에서 이만큼만 (동작 확인용, 0 = 전부)")
    parser.add_argument("--epochs", type=int, default=0, help="설정의 에폭 수 대신 (동작 확인용, 0 = 설정대로)")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("기존 결과와 섞이지 않도록 새 --out 을 지정하세요")
    config = json.loads(args.config.read_text(encoding="utf-8"))["train"]
    if args.epochs:
        config = {**config, "epochs": args.epochs}

    import numpy as np
    import torch
    import torchvision
    from torch.utils.data import DataLoader, Dataset

    names = yolo.class_names(load_items())
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    def samples_of(split):
        s = presence.split_samples(args.data, split)
        return s[:args.limit] if args.limit else s

    train, val = samples_of("train"), samples_of("val")

    class Train(Dataset):
        def __len__(self):
            return len(train)

        def __getitem__(self, i):
            path, present = train[i]
            flip = random.random() < config["hflip"]
            return (torch.from_numpy(load_image(path, config["imgsz"], flip)),
                    torch.tensor(presence.multi_hot(present, len(names))))

    loader_kwargs = {"batch_size": config["batch"], "num_workers": config["workers"], "pin_memory": device.type == "cuda",
                     "persistent_workers": config["workers"] > 0}
    generator = torch.Generator().manual_seed(config["seed"])
    training = DataLoader(Train(), shuffle=True, generator=generator, drop_last=True,
                          worker_init_fn=lambda w: random.seed(config["seed"] * 1000 + w), **loader_kwargs)
    net = build_model(config, len(names)).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    val_labels = [presence.multi_hot(p, len(names)) for _, p in val]

    args.out.mkdir(parents=True)
    log, best, best_epoch = [], -1.0, 0
    start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        net.train()
        losses = []
        for x, y in training:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(net(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        aps = presence.item_ap(val_labels, predict(net, val, config, device, loader_kwargs), names)
        mean_ap = statistics.fmean(aps.values())
        log.append({"epoch": epoch, "seconds": round(time.time() - start), "train_loss": statistics.fmean(losses),
                    "val_mean_ap": mean_ap, **{f"val_ap_{k}": v for k, v in aps.items()}})
        print(f"에폭 {epoch}/{config['epochs']}: 손실 {log[-1]['train_loss']:.4f} · val 평균 AP {mean_ap:.4f} · {log[-1]['seconds']}초", flush=True)
        if mean_ap > best:
            best, best_epoch = mean_ap, epoch
            torch.save(net.state_dict(), args.out / "best.pt")
    with (args.out / "epochs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(log[0]))
        writer.writeheader()
        writer.writerows(log)

    net.load_state_dict(torch.load(args.out / "best.pt", map_location=device, weights_only=True))
    for split in SCORE_SPLITS:
        samples = samples_of(split)
        probs = predict(net, samples, config, device, loader_kwargs)
        rows = [r for (path, present), p in zip(samples, probs, strict=True)
                for r in scores.rows_for_image(path.stem, list(enumerate(p)), present, names)]
        scores.write_predictions(rows, args.out / "scores" / f"{split}.csv", fmt=".9g")
        print(f"{split}: {len(samples):,}장 점수", flush=True)

    best_row = log[best_epoch - 1]
    lines = ["# 사진 단위 품목 분류기 (#56)", "",
             f"{config['model']} ({config['pretrained']}) · 입력 {config['imgsz']} 정사각 letterbox · {config['epochs']}에폭 · "
             f"batch {config['batch']} · 학습 {len(train):,}장 · val {len(val):,}장 · {round(time.time() - start)}초", "",
             f"val 평균 이미지 AP 최고: **{best:.4f}** ({best_epoch}에폭)", "", "| 품목 | val AP |", "|---|---:|"]
    lines += [f"| {n} | {best_row[f'val_ap_{n}']:.3f} |" for n in names if f"val_ap_{n}" in best_row]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"config": str(args.config), "config_sha256": digest(args.config), "weights_sha256": digest(args.out / "best.pt"),
                "best_epoch": best_epoch, "val_mean_ap": best, "torch": torch.__version__, "torchvision": torchvision.__version__,
                "counts": {"train": len(train), "val": len(val)}, "limit": args.limit, "epochs": config["epochs"]}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
