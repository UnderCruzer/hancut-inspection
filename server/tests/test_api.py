import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import judgment
from app.main import create_app
from app.predictor import Prediction


class FakePredictor:
    model_version = "fake-0.1"

    def __init__(self, score=0.9, item="Gun", reasons=("칼날 윤곽",)):
        self._prediction = Prediction(
            item=item, score=score, reasons=list(reasons), model_version=self.model_version
        )

    def predict(self, image_bytes: bytes) -> Prediction:
        assert image_bytes, "빈 바이트가 모델까지 내려오면 안 된다"
        return self._prediction


def _thresholds_file(tmp_path, table=None):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(table or {
        "default": {"low": 0.2, "high": 0.8},
        "Knife": {"low": 0.05, "high": 0.6, "miss_rate_cap": 0.01},
    }), encoding="utf-8")
    return path


def _image_bytes(size=(64, 64), fmt="JPEG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, (180, 40, 30)).save(buffer, format=fmt)
    return buffer.getvalue()


def _client(tmp_path, predictor=None, table=None):
    return TestClient(create_app(predictor=predictor, thresholds_path=_thresholds_file(tmp_path, table)))


def _upload(client, content_type="image/jpeg", data=None):
    return client.post(
        "/v1/inspections",
        files={"image": ("photo.jpg", data if data is not None else _image_bytes(), content_type)},
    )


class TestHealth:
    def test_reports_that_no_model_is_loaded(self, tmp_path):
        body = _client(tmp_path).get("/health").json()
        assert body == {"status": "ok", "model_loaded": False, "thresholds_loaded": True}

    def test_reports_loaded_model(self, tmp_path):
        body = _client(tmp_path, FakePredictor()).get("/health").json()
        assert body["model_loaded"] is True


class TestJudge:
    def test_returns_503_when_model_is_missing(self, tmp_path):
        response = _upload(_client(tmp_path))
        assert response.status_code == 503

    def test_high_score_is_auto_alarm(self, tmp_path):
        response = _upload(_client(tmp_path, FakePredictor(score=0.95)))
        body = response.json()
        assert response.status_code == 200
        assert body["zone"] == judgment.AUTO_ALARM
        assert body["needs_review"] is False
        assert body["reasons"] == ["칼날 윤곽"]
        assert body["model_version"] == "fake-0.1"

    def test_middle_score_goes_to_a_human(self, tmp_path):
        body = _upload(_client(tmp_path, FakePredictor(score=0.5))).json()
        assert body["zone"] == judgment.REVIEW
        assert body["needs_review"] is True

    def test_low_score_is_auto_clear(self, tmp_path):
        body = _upload(_client(tmp_path, FakePredictor(score=0.01))).json()
        assert body["zone"] == judgment.AUTO_CLEAR

    def test_item_specific_thresholds_are_used(self, tmp_path):
        # Knife은 low=0.05 — 같은 점수라도 기본 임계값(0.2)과 결과가 다르다
        body = _upload(_client(tmp_path, FakePredictor(score=0.1, item="Knife"))).json()
        assert body["zone"] == judgment.REVIEW
        assert body["thresholds"] == {"low": 0.05, "high": 0.6}

    def test_rejects_unsupported_content_type(self, tmp_path):
        response = _upload(_client(tmp_path, FakePredictor()), content_type="application/pdf")
        assert response.status_code == 415

    def test_rejects_empty_file(self, tmp_path):
        response = _upload(_client(tmp_path, FakePredictor()), data=b"")
        assert response.status_code == 400

    def test_rejects_file_over_the_size_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.main.MAX_UPLOAD_BYTES", 128)
        response = _upload(_client(tmp_path, FakePredictor()), data=_image_bytes((512, 512)))
        assert response.status_code == 413


class TestThresholds:
    def test_requires_a_default_entry(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text(json.dumps({"Knife": {"low": 0.1, "high": 0.9}}), encoding="utf-8")
        with pytest.raises(ValueError, match="default"):
            judgment.load_thresholds(path)

    def test_rejects_reversed_thresholds(self):
        with pytest.raises(ValueError):
            judgment.Thresholds(low=0.9, high=0.1)

    def test_zone_boundaries_match_the_ml_convention(self):
        t = judgment.Thresholds(low=0.2, high=0.8)
        assert judgment.zone_of(0.199, t) == judgment.AUTO_CLEAR
        assert judgment.zone_of(0.2, t) == judgment.REVIEW
        assert judgment.zone_of(0.8, t) == judgment.AUTO_ALARM

    def test_rejects_score_outside_range(self):
        with pytest.raises(ValueError):
            judgment.zone_of(1.2, judgment.Thresholds(low=0.2, high=0.8))
