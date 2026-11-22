import importlib.util
from pathlib import Path

import pytest

from hancut.data import presence

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "train_presence.py"


def test_multi_hot_marks_present_classes_and_rejects_out_of_range():
    assert presence.multi_hot({0, 2}, 3) == [1.0, 0.0, 1.0]
    assert presence.multi_hot(set(), 2) == [0.0, 0.0]
    with pytest.raises(ValueError):
        presence.multi_hot({3}, 3)


def test_letterbox_shape_fits_the_long_side_and_centers():
    assert presence.letterbox_shape(448, 896, 640) == (320, 640, 160, 0)
    assert presence.letterbox_shape(448, 448, 640) == (640, 640, 0, 0)
    assert presence.letterbox_shape(600, 300, 640) == (640, 320, 0, 160)
    with pytest.raises(ValueError):
        presence.letterbox_shape(0, 10, 640)


def test_split_samples_reads_labels_and_treats_missing_files_as_empty(tmp_path):
    for name in ("b", "a"):
        (tmp_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        (tmp_path / "images" / "val" / f"{name}.png").write_bytes(b"")
    (tmp_path / "labels" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val" / "a.txt").write_text("3 0.5 0.5 0.1 0.1\n11 0.2 0.2 0.1 0.1\n3 0.6 0.6 0.1 0.1\n")
    samples = presence.split_samples(tmp_path, "val")
    assert [(p.name, c) for p, c in samples] == [("a.png", {3, 11}), ("b.png", set())]
    with pytest.raises(ValueError, match="사진이 없다"):
        presence.split_samples(tmp_path, "train")


def test_item_ap_skips_items_without_positives():
    labels = [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    probs = [[0.9, 0.1], [0.2, 0.5], [0.4, 0.3]]
    assert presence.item_ap(labels, probs, ["Gun", "Knife"]) == {"Gun": pytest.approx(1.0)}
    with pytest.raises(ValueError):
        presence.item_ap(labels, probs[:2], ["Gun", "Knife"])


def test_load_image_letterboxes_flips_and_normalizes(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    spec = importlib.util.spec_from_file_location("train_presence", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    img = np.zeros((448, 896, 3), dtype=np.uint8)
    img[:, :448] = (0, 0, 255)  # 왼쪽 절반 빨강 (BGR)
    path = tmp_path / "a.png"
    cv2.imwrite(str(path), img)
    x = script.load_image(path, 64, flip=False)
    assert x.shape == (3, 64, 64) and x.dtype == np.float32
    red = (1 - script.MEAN[0]) / script.STD[0]
    pad = (114 / 255 - script.MEAN[0]) / script.STD[0]
    assert x[0, 32, 2] == pytest.approx(red, abs=1e-5) and x[0, 0, 2] == pytest.approx(pad, abs=1e-5)
    assert script.load_image(path, 64, flip=True)[0, 32, 61] == pytest.approx(red, abs=1e-5)
