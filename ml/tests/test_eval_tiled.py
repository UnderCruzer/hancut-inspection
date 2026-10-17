"""GPU runtime dependencies are available on the inference instance."""
import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torchvision")
pytest.importorskip("ultralytics")
spec = importlib.util.spec_from_file_location("eval_tiled", Path(__file__).resolve().parents[1] / "scripts/eval_tiled.py")
et = importlib.util.module_from_spec(spec)
spec.loader.exec_module(et)


def test_cross_tile_duplicates_suppressed_without_touching_other_classes():
    full = np.array([[45, 42, 55, 52, .8, 0], [45, 42, 55, 52, .9, 1]], dtype=np.float32)
    crop = np.array([[5, 10, 15, 20, .95, 0], [1, 1, 2, 2, .99, 1]], dtype=np.float32)
    merged = et.merge_predictions(full, [crop], [(40, 32, 100, 80)], [0], .7)
    assert len(merged) == 2
    np.testing.assert_array_equal(merged[merged[:, 5] == 1], full[full[:, 5] == 1])
    np.testing.assert_allclose(merged[merged[:, 5] == 0][0], [45, 42, 55, 52, .95, 0])


def test_empty_detections_still_count_ground_truth():
    m = et.Metrics(["Lighter"], .25)
    m.add(np.empty((0, 6), dtype=np.float32), np.array([[0, 0, 10, 10, 0]], dtype=np.float32))
    s = m.summary()["per_class"]["Lighter"]
    assert s["gt_boxes"] == 1 and s["absent_images"] == 1
    assert s["ap50"] == 0 and s["recall"] == 0


def test_duplicate_and_wrong_class_boxes_are_false_positives():
    gt = np.array([[0, 0, 10, 10, 0]], dtype=np.float32)
    pred = np.array([[0, 0, 10, 10, .9, 0], [0, 0, 10, 10, .8, 0],
                     [0, 0, 10, 10, .7, 1]], dtype=np.float32)
    m = et.Metrics(["Lighter", "Bullet"], .25)
    m.add(pred, gt)
    s = m.summary()["per_class"]
    assert s["Lighter"]["tp"] == 1
    assert s["Lighter"]["precision"] == .5
    assert s["Bullet"]["false_alarm_images"] == 1


def test_fixed_threshold_matching_ignores_filtered_boxes():
    gt = np.array([[0, 0, 10, 10, 0]], dtype=np.float32)
    pred = np.array([[0, 0, 10, 10, .1, 0], [0, 0, 9, 9, .9, 0]], dtype=np.float32)
    m = et.Metrics(["Lighter"], .25)
    m.add(pred, gt)
    assert m.summary()["per_class"]["Lighter"]["tp"] == 1


def test_missing_label_is_not_silently_a_negative(tmp_path):
    with pytest.raises(FileNotFoundError):
        et.read_labels(tmp_path / "missing.txt", 100, 100)
