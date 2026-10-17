import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("refine_tiled", Path(__file__).resolve().parents[1] / "scripts/refine_tiled.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)


def test_selector_rejects_localization_regression_and_can_choose_baseline():
    def candidate(w, ap, loc):
        return dict(weight=w, nms_iou=.5, metrics={"per_class": {"Lighter": {"ap50": ap, "ap50_95": loc}}})
    trials = [candidate(0, .8, .7), candidate(.5, .9, .6), candidate(.25, .79, .7)]
    assert rt.choose(trials, ["Lighter"])["Lighter"]["weight"] == 0
    trials.append(candidate(.75, .85, .71))
    assert rt.choose(trials, ["Lighter"])["Lighter"]["weight"] == .75


def test_zero_weight_recovers_full_boxes_suppressed_by_original_tile_merge():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torchvision")
    full = np.array([[0, 0, 10, 10, .7, 0], [20, 20, 30, 30, .8, 1]], dtype=np.float32)
    tiled = np.array([[0, 0, 9, 9, .9, 0]], dtype=np.float32)
    result = rt.refine(full, tiled, {"Lighter": dict(weight=0, nms_iou=.5)}, ["Lighter", "Bullet"])
    np.testing.assert_array_equal(result[result[:, 5] == 0], full[full[:, 5] == 0])
    np.testing.assert_array_equal(result[result[:, 5] == 1], full[full[:, 5] == 1])


def test_new_tile_scores_scaled_without_scaling_full_image_scores():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torchvision")
    full = np.array([[0, 0, 10, 10, .8, 0]], dtype=np.float32)
    tiled = np.array([[0, 0, 10, 10, .8, 0], [20, 20, 30, 30, .9, 0]], dtype=np.float32)
    result = rt.refine(full, tiled, {"Lighter": dict(weight=.5, nms_iou=.5)}, ["Lighter"])
    assert len(result) == 2
    np.testing.assert_allclose(sorted(result[:, 4]), [.45, .8])
