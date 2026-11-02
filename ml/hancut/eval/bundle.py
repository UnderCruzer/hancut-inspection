"""Dense 설정 F 임계값과 겹치지 않는 평가 결과를 서버용으로 내보낸다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import cli, resplit
from .grouped import DIFFICULTIES, grouped_table

SCORE_METHOD = "yolo_class_max_pre_nms"
PREPROCESS = "letterbox_square_bgr_to_rgb_float32_v1"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prediction_manifest(folder: Path, weights: Path, names: list[str], imgsz: int,
                        version: str, splits: list[str], *, batch_size: int,
                        cudnn_allow_tf32: bool, cuda_matmul_allow_tf32: bool) -> dict:
    return {"schema_version": 1, "score_method": SCORE_METHOD, "preprocess": PREPROCESS,
            "imgsz": imgsz, "ultralytics_version": version, "class_names": names,
            "batch_size": batch_size, "cudnn_allow_tf32": cudnn_allow_tf32,
            "cuda_matmul_allow_tf32": cuda_matmul_allow_tf32,
            "weights_sha256": sha256(weights),
            "files": {f"{s}.csv": sha256(folder / f"{s}.csv") for s in splits}}


def export_bundle(folder: Path, names: list[str], seed: int) -> tuple[dict, dict]:
    manifest = json.loads((folder / "manifest.json").read_text())
    for key, value in {"schema_version": 1, "score_method": SCORE_METHOD, "preprocess": PREPROCESS,
                       "imgsz": 640, "ultralytics_version": "8.4.164", "class_names": names,
                       "batch_size": 1, "cudnn_allow_tf32": False, "cuda_matmul_allow_tf32": False}.items():
        if manifest.get(key) != value:
            raise ValueError(f"dense manifest {key} 불일치")
    rows = []
    for difficulty in DIFFICULTIES:
        path = folder / f"test_{difficulty}.csv"
        if manifest["files"].get(path.name) != sha256(path):
            raise ValueError(f"예측 파일 SHA256 불일치: {path.name}")
        split = cli.read_predictions(path)
        if any(resplit.difficulty_of(r["image_id"]) != difficulty for r in split):
            raise ValueError("시험 난이도 또는 이미지 ID 불일치")
        rows.extend(split)
    image_items = {}
    for row in rows:
        image_items.setdefault(row["image_id"], set()).add(row["item"])
    if any(items != set(names) for items in image_items.values()):
        raise ValueError("각 이미지에 모든 품목 점수가 필요하다")
    calib_ids, eval_ids = resplit.split_images(resplit.image_table(rows), 0.5, seed)
    calib, evaluation = resplit.partition(rows, calib_ids)
    # 각 품목×난이도에서 양성/음성이 양쪽 모두 존재해야 결과를 해석할 수 있다.
    for subset in (calib, evaluation):
        for name in names:
            for difficulty in DIFFICULTIES:
                labels = {r["y_true"] for r in subset if r["item"] == name
                          and resplit.difficulty_of(r["image_id"]) == difficulty}
                if labels != {0, 1}:
                    raise ValueError(f"보정/평가 표본 부족: {name} {difficulty}")
    delta = 0.05 / (len(names) * len(DIFFICULTIES))

    def points(subset, table):
        def point(rs):
            p = cli.point_for(rs, table)
            return {**p, "n_non_threat": p["n"] - p["n_threat"]}
        return {"overall": point(subset),
                "by_item": {n: point([r for r in subset if r["item"] == n]) for n in names},
                "by_cell": {f"{n}/{d}": point([r for r in subset if r["item"] == n
                                               and resplit.difficulty_of(r["image_id"]) == d])
                            for n in names for d in DIFFICULTIES}}

    tables = {str(cap): grouped_table(calib, cap, "upper", delta) for cap in (0.05, 0.02, 0.01)}
    report = {"purpose": "exploratory; reused PIDray test distribution, not a new independent holdout",
              "seed": seed, "fraction": 0.5, "delta": delta,
              "calibration_images": sorted(calib_ids), "evaluation_images": sorted(eval_ids),
              "e2": {cap: {"calibration": points(calib, t), "evaluation": points(evaluation, t)}
                     for cap, t in tables.items()}}
    bundle = {k: v for k, v in manifest.items() if k != "files"}
    bundle.update(purpose="exploratory", thresholds=tables["0.05"],
                  calibration={"seed": seed, "fraction": 0.5, "delta": delta, "miss_rate_cap": 0.05,
                               "prediction_sha256": manifest["files"],
                               "n_calibration_images": len(calib_ids), "n_evaluation_images": len(eval_ids)})
    return bundle, report
