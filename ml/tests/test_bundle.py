import json
from pathlib import Path

import pytest

from hancut.config import load_items
from hancut.data.yolo import class_names
from hancut.eval.bundle import export_bundle, prediction_manifest
from hancut.eval import scores, cli
from hancut.eval.grouped import grouped_table


@pytest.fixture
def predictions(tmp_path):
    names = class_names(load_items())
    weights = tmp_path / "test.pt"
    weights.write_bytes(b"fixture; not a model")
    splits = [f"test_{d}" for d in ("easy", "hard", "hidden")]
    for split in splits:
        rows = []
        for i in range(96):
            for j, name in enumerate(names):
                positive = i % len(names) == j
                rows.append(dict(image_id=f"xray_{split[5:]}{i:05}", item=name,
                                 y_true=int(positive), score=0.8 if positive else 1e-8))
        scores.write_predictions(rows, tmp_path / f"{split}.csv", fmt=".9g")
    manifest = prediction_manifest(tmp_path, weights, names, 640, "8.4.164", splits,
                                   batch_size=1, cudnn_allow_tf32=False, cuda_matmul_allow_tf32=False)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, names


def test_export_matches_f_and_records_disjoint_e2(predictions):
    folder, names = predictions
    bundle, report = export_bundle(folder, names, 0)
    assert (bundle, report) == export_bundle(folder, names, 0)
    calib, evaluation = set(report["calibration_images"]), set(report["evaluation_images"])
    assert calib and evaluation and not calib & evaluation
    assert len(calib | evaluation) == 288
    rows = [r for d in ("easy", "hard", "hidden")
            for r in cli.read_predictions(folder / f"test_{d}.csv") if r["image_id"] in calib]
    assert bundle["thresholds"] == grouped_table(rows, 0.05, "upper", 0.05 / 36)
    assert set(report["e2"]) == {"0.05", "0.02", "0.01"}
    point = report["e2"]["0.05"]["evaluation"]["overall"]
    assert point["n"] == len(evaluation) * 12
    assert point["n_threat"] + point["n_non_threat"] == point["n"]
    assert point["review_rate"] + point["auto_rate"] == 1
    assert len(report["e2"]["0.05"]["evaluation"]["by_cell"]) == 36


def test_modified_csv_rejected(predictions):
    folder, names = predictions
    path = folder / "test_easy.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA256"):
        export_bundle(folder, names, 0)


def test_wrong_score_method_rejected(predictions):
    folder, names = predictions
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["score_method"] = "nms"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="score_method"):
        export_bundle(folder, names, 0)


def test_missing_image_item_rejected_even_with_new_hash(predictions):
    folder, names = predictions
    path = folder / "test_easy.csv"
    rows = cli.read_predictions(path)[1:]
    scores.write_predictions(rows, path, fmt=".9g")
    splits = [f"test_{d}" for d in ("easy", "hard", "hidden")]
    manifest = prediction_manifest(folder, folder / "test.pt", names, 640, "8.4.164", splits,
                                   batch_size=1, cudnn_allow_tf32=False, cuda_matmul_allow_tf32=False)
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="모든 품목"):
        export_bundle(folder, names, 0)


@pytest.mark.parametrize("key,value", [("batch_size", 32), ("cudnn_allow_tf32", True), ("cuda_matmul_allow_tf32", True)])
def test_export_rejects_different_inference_settings(predictions, key, value):
    folder, names = predictions
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=key):
        export_bundle(folder, names, 0)
