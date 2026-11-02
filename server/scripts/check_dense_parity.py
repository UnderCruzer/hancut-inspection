#!/usr/bin/env python3
"""실제 PIDray 이미지의 오프라인 dense CSV와 API 응답/판정 일치를 검사한다."""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from pathlib import Path
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "server"), str(REPO / "ml")]

from app.dense import DenseBundle, DensePredictor, sha256
from app.judgment import zone_of
from hancut.eval import cli, resplit


def select_images(rows, names, per_cell, seed):
    if per_cell < 1:
        raise ValueError("per-cell은 1 이상이어야 한다")
    rng = random.Random(seed)
    selected = set()
    for name in names:
        for difficulty in ("easy", "hard", "hidden"):
            candidates = sorted({r["image_id"] for r in rows if r["item"] == name
                                 and r["y_true"] == 1 and resplit.difficulty_of(r["image_id"]) == difficulty})
            if not candidates:
                raise ValueError(f"양성 사진이 없다: {name}/{difficulty}")
            selected.update(rng.sample(candidates, min(per_cell, len(candidates))))
    return sorted(selected)


def compare_item(expected, actual, thresholds):
    return {"score_match": math.isclose(expected, actual["score"], rel_tol=1e-4, abs_tol=1e-12),
            "zone_match": actual["zone"] == zone_of(expected, thresholds),
            "expected_score": expected, "actual_score": actual["score"],
            "expected_zone": zone_of(expected, thresholds), "actual_zone": actual["zone"]}


def image_upload(path):
    """PIDray hidden은 .png 파일명이어도 JPEG일 수 있다. 재인코딩하지 않는다."""
    payload = path.read_bytes()
    with Image.open(io.BytesIO(payload)) as image:
        fmt = image.format
    formats = {"PNG": (".png", "image/png"), "JPEG": (".jpg", "image/jpeg")}
    if fmt not in formats:
        raise ValueError(f"지원하지 않는 실제 이미지 형식: {fmt}")
    suffix, mime = formats[fmt]
    return path.stem + suffix, payload, mime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("weights", "bundle", "pred-dir", "images", "out"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--per-cell", type=int, default=3)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("새 출력 경로를 지정하세요")
    bundle = DenseBundle.load(args.bundle)
    raw_bundle = json.loads(args.bundle.read_text())
    rows = []
    for difficulty in ("easy", "hard", "hidden"):
        path = args.pred_dir / f"test_{difficulty}.csv"
        if raw_bundle["calibration"]["prediction_sha256"][path.name] != sha256(path):
            raise ValueError("bundle의 보정 CSV와 비교 CSV가 다르다")
        rows.extend(cli.read_predictions(path))
    selected = select_images(rows, bundle.names, args.per_cell, args.seed)
    expected = {(r["image_id"], r["item"]): r["score"] for r in rows}

    from fastapi.testclient import TestClient
    from app.main import create_app

    predictor = DensePredictor(args.weights, bundle, args.device)
    differences, checked = [], 0
    with TestClient(create_app(dense_predictor=predictor)) as client:
        for image_id in selected:
            path = args.images / f"test_{resplit.difficulty_of(image_id)}" / f"{image_id}.png"
            response = client.post("/v2/inspections", files={"image": image_upload(path)})
            if response.status_code != 200:
                differences.append({"image_id": image_id, "status": response.status_code, "error": response.text})
                continue
            for item in response.json()["items"]:
                name = item["item"]
                comparison = compare_item(expected[image_id, name], item, bundle.thresholds[name])
                checked += 1
                if not comparison["score_match"] or not comparison["zone_match"]:
                    differences.append({"image_id": image_id, "item": name, **comparison})
    report = {"seed": args.seed, "images": selected, "checked_image_items": checked,
              "bundle_sha256": bundle.bundle_sha256, "differences": differences,
              "http_failure_count": sum("status" in d for d in differences),
              "score_difference_count": sum(d.get("score_match") is False for d in differences),
              "zone_difference_count": sum(d.get("zone_match") is False for d in differences),
              "passed": not differences and checked == len(selected) * len(bundle.names)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"{len(selected)}장 / {checked}개 사진×품목 검사, HTTP 실패 {report['http_failure_count']}장, "
          f"점수 차이 {report['score_difference_count']}개, 판정 차이 {report['zone_difference_count']}개: {args.out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
