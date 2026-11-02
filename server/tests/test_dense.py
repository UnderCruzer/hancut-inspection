import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.dense import DenseBundle, DensePredictor, PREPROCESS, SCORE_METHOD, ULTRALYTICS_VERSION
from app.main import create_app

NAMES = tuple(x["dataset_name"] for x in json.loads(
    (Path(__file__).resolve().parents[2] / "ml/configs/items.json").read_text())["items"])


def bundle_dict():
    return {"schema_version": 1, "score_method": SCORE_METHOD, "preprocess": PREPROCESS,
            "ultralytics_version": ULTRALYTICS_VERSION, "imgsz": 640, "purpose": "exploratory",
            "batch_size": 1, "cudnn_allow_tf32": False, "cuda_matmul_allow_tf32": False,
            "class_names": list(NAMES), "weights_sha256": "a" * 64,
            "thresholds": {n: {"low": 1e-8, "high": 1e-6} for n in (*NAMES, "default")}}


def load_bundle(tmp_path, raw=None):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(raw if raw is not None else bundle_dict()))
    return DenseBundle.load(path)


class FakeDense:
    def __init__(self, bundle):
        self.bundle = bundle
        self.scores = {n: 2e-8 for n in bundle.names}
        self.calls = 0

    def predict(self, payload):
        self.calls += 1
        return self.scores


def png():
    stream = io.BytesIO()
    Image.new("RGB", (16, 8)).save(stream, "PNG")
    return stream.getvalue()


def upload(client, data=None, mime="image/png"):
    return client.post("/v2/inspections", files={"image": ("x.png", png() if data is None else data, mime)})


@pytest.fixture
def service(tmp_path):
    predictor = FakeDense(load_bundle(tmp_path))
    app = create_app(dense_predictor=predictor, thresholds_path=tmp_path / "absent.json")
    with TestClient(app) as client:
        yield predictor, app, client


def test_returns_all_items_and_preserves_tiny_scores_and_boundaries(service):
    predictor, _, client = service
    predictor.scores[NAMES[0]] = 0.0
    predictor.scores[NAMES[1]] = 1e-8
    predictor.scores[NAMES[2]] = 1e-6
    response = upload(client)
    assert response.status_code == 200
    body = response.json()
    assert [i["item"] for i in body["items"]] == list(NAMES)
    assert [i["zone"] for i in body["items"][:3]] == ["auto_clear", "review", "auto_alarm"]
    assert body["items"][-1]["score"] == 2e-8
    assert body["experimental"] is True
    assert body["model_version"] == "sha256:" + "a" * 64
    assert body["bundle_sha256"] == predictor.bundle.bundle_sha256
    assert client.get("/v2/health").json()["model_loaded"] is True


@pytest.mark.parametrize("data,mime,status", [
    (b"", "image/png", 400), (b"not an image", "image/png", 400),
    (None, "image/jpeg", 415), (None, "application/octet-stream", 415),
])
def test_invalid_images_never_reach_model(service, data, mime, status):
    predictor, _, client = service
    assert upload(client, data, mime).status_code == status
    assert predictor.calls == 0


def test_rejects_truncated_png(service):
    predictor, _, client = service
    assert upload(client, png()[:40]).status_code == 400
    assert predictor.calls == 0


def test_byte_and_pixel_limits(service, monkeypatch):
    predictor, app, client = service
    app.state.max_upload_bytes = 5
    assert upload(client).status_code == 413
    app.state.max_upload_bytes = 10000
    monkeypatch.setattr("app.dense_api.MAX_IMAGE_PIXELS", 32)
    assert upload(client).status_code == 413
    assert predictor.calls == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, "bad"])
def test_invalid_model_scores_fail_closed(service, bad):
    predictor, _, client = service
    predictor.scores[NAMES[0]] = bad
    assert upload(client).status_code == 503


def test_missing_item_fails_closed(service):
    predictor, _, client = service
    predictor.scores.pop(NAMES[-1])
    assert upload(client).status_code == 503


def test_inference_failure_returns_503(service, monkeypatch):
    predictor, _, client = service
    def fail(_):
        raise RuntimeError("CUDA failure")
    monkeypatch.setattr(predictor, "predict", fail)
    assert upload(client).status_code == 503


@pytest.mark.parametrize("key,value", [
    ("score_method", "nms"), ("imgsz", 960), ("ultralytics_version", "other"),
    ("preprocess", "stretch"), ("weights_sha256", "bad"),
    ("class_names", list(NAMES[:-1])), ("class_names", ["Gun"] * 12),
    ("purpose", "production"),
    ("batch_size", 32), ("cudnn_allow_tf32", True), ("cuda_matmul_allow_tf32", True),
])
def test_bundle_rejects_mismatch(tmp_path, key, value):
    raw = bundle_dict()
    raw[key] = value
    with pytest.raises(ValueError):
        load_bundle(tmp_path, raw)


def test_bundle_rejects_missing_item_or_nan(tmp_path):
    raw = bundle_dict()
    raw["thresholds"].pop(NAMES[0])
    with pytest.raises(ValueError):
        load_bundle(tmp_path, raw)
    raw = bundle_dict()
    raw["thresholds"][NAMES[0]]["low"] = float("nan")
    with pytest.raises(ValueError):
        load_bundle(tmp_path, raw)


def test_checkpoint_hash_checked_before_importing_gpu_libraries(tmp_path):
    weights = tmp_path / "bad.pt"
    weights.write_bytes(b"wrong weights")
    with pytest.raises(ValueError, match="SHA256"):
        DensePredictor(weights, load_bundle(tmp_path))


def test_failed_startup_serves_503(tmp_path, monkeypatch):
    monkeypatch.setenv("DENSE_WEIGHTS_PATH", str(tmp_path / "missing.pt"))
    monkeypatch.delenv("DENSE_BUNDLE_PATH", raising=False)
    with TestClient(create_app(thresholds_path=tmp_path / "absent.json")) as client:
        assert upload(client).status_code == 503
        assert client.get("/v2/health").json()["model_loaded"] is False
