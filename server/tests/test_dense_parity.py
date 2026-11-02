import importlib.util
import json
from pathlib import Path

from app.judgment import Thresholds

spec = importlib.util.spec_from_file_location(
    "check_dense_parity", Path(__file__).resolve().parents[1] / "scripts/check_dense_parity.py")
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


def test_select_images_covers_each_cell_reproducibly():
    rows = [dict(image_id=f"xray_{d}{i:04}", item=n, y_true=int(i % 2 == j))
            for d in ("easy", "hard", "hidden") for i in range(10)
            for j, n in enumerate(("Gun", "Lighter"))]
    selected = parity.select_images(rows, ("Gun", "Lighter"), 2, 0)
    assert selected == parity.select_images(rows, ("Gun", "Lighter"), 2, 0)
    assert len(selected) == 12
    for d in ("easy", "hard", "hidden"):
        for n in ("Gun", "Lighter"):
            assert sum(r["image_id"] in selected and r["item"] == n and r["y_true"] == 1
                       and r["image_id"].startswith(f"xray_{d}") for r in rows) == 2


def test_small_score_error_does_not_hide_zone_change():
    t = Thresholds(1e-8, 1e-6)
    result = parity.compare_item(1e-8, {"score": 1e-8 - 1e-13, "zone": "auto_clear"}, t)
    assert result["score_match"] is True
    assert result["zone_match"] is False


def test_upload_uses_actual_jpeg_format_despite_png_extension(tmp_path):
    from PIL import Image
    path = tmp_path / "xray_hidden00001.png"
    Image.new("RGB", (16, 16)).save(path, format="JPEG")
    name, payload, mime = parity.image_upload(path)
    assert name == "xray_hidden00001.jpg"
    assert mime == "image/jpeg"
    assert payload == path.read_bytes()


def test_upload_keeps_png_pixels_and_mime(tmp_path):
    from PIL import Image
    path = tmp_path / "xray_easy00001.png"
    Image.new("RGB", (16, 16)).save(path, format="PNG")
    assert parity.image_upload(path) == (path.name, path.read_bytes(), "image/png")


def test_ml_export_loads_in_server_without_changing_thresholds(tmp_path):
    from app.dense import DenseBundle
    from hancut.config import load_items
    from hancut.data.yolo import class_names
    from hancut.eval.bundle import export_bundle, prediction_manifest
    from hancut.eval.scores import write_predictions
    from hancut.eval import zones

    names = class_names(load_items())
    weights = tmp_path / "fixture.pt"
    weights.write_bytes(b"synthetic fixture; not a trained checkpoint")
    splits = [f"test_{d}" for d in ("easy", "hard", "hidden")]
    for split in splits:
        rows = [dict(image_id=f"xray_{split[5:]}{i:05}", item=name,
                     y_true=int(i % 12 == j), score=0.8 if i % 12 == j else 2e-8)
                for i in range(96) for j, name in enumerate(names)]
        write_predictions(rows, tmp_path / f"{split}.csv", fmt=".9g")
    manifest = prediction_manifest(tmp_path, weights, names, 640, "8.4.164", splits,
                                   batch_size=1, cudnn_allow_tf32=False, cuda_matmul_allow_tf32=False)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    exported, _ = export_bundle(tmp_path, names, seed=0)
    bundle_path = tmp_path / "dense-bundle.json"
    bundle_path.write_text(json.dumps(exported, allow_nan=False))
    loaded = DenseBundle.load(bundle_path)
    assert loaded.names == tuple(names)
    assert loaded.weights_sha256 == manifest["weights_sha256"]
    for name in names:
        table = exported["thresholds"][name]
        server_table = loaded.thresholds[name]
        assert (server_table.low, server_table.high) == (table["low"], table["high"])
        ml_table = zones.Thresholds(low=table["low"], high=table["high"])
        for score in (0.0, 2e-8, table["low"], table["high"], 1.0):
            assert parity.zone_of(score, server_table) == zones.zone_of(score, ml_table)
