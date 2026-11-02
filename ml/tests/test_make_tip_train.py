import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hancut.config import load_items

SCRIPT =Path(__file__).resolve().parent.parent / "scripts" / "make_tip_train.py"
spec = importlib.util.spec_from_file_location("make_tip_train", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

NAMES = ["Baton", "Knife"]


def test_parse_weights_reads_specs_and_exits_on_bad_ones():
    assert script.parse_weights(["Knife=2.5"], NAMES) == {0: 1.0, 1: 2.5}
    for bad in (["Knife"], ["Knife=x"], ["Sword=2"], ["Knife=0"]):
        with pytest.raises(SystemExit):
            script.parse_weights(bad, NAMES)


def test_object_pool_keeps_only_train_objects_with_usable_masks():
    annotation = {
        "images": [{"id": 0, "file_name": "a.png"}, {"id": 1, "file_name": "held_out.png"}],
        "annotations": [
            {"image_id": 0, "category_id": 7, "bbox": [0, 0, 20, 20], "segmentation": [[0, 0, 20, 0, 20, 20]]},
            {"image_id": 1, "category_id": 7, "bbox": [0, 0, 20, 20], "segmentation": [[0, 0, 20, 0, 20, 20]]},
            {"image_id": 0, "category_id": 7, "bbox": [0, 0, 20, 20], "segmentation": []},
            {"image_id": 0, "category_id": 7, "bbox": [0, 0, 3, 20], "segmentation": [[0, 0, 3, 0, 3, 20]]},
            {"image_id": 0, "category_id": 7, "bbox": [0, 0, 20, 20], "segmentation": {"counts": "x"}, "iscrowd": 1},
        ],
    }
    pool = script.object_pool(annotation, {7: 1}, {"a"})
    assert list(pool) == [1]
    assert [stem for stem, _ in pool[1]] == ["a"]


def test_read_rects_converts_normalized_boxes(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("3 0.5 0.5 0.2 0.4\n\n")
    assert script.read_rects(path, 100, 50) == [(40, 15, 60, 35)]


def _dataset(root: Path, np, Image) -> tuple[Path, Path]:
    """make_yolo_dataset.py 와 같은 모양의 작은 데이터. 사진은 생성한 회색 가방이다."""
    data, pidray = root / "yolo", root / "pidray"
    (pidray / "annotations").mkdir(parents=True)
    images, annotations = [], []
    for split in ("train", "val"):
        (data / "images" / split).mkdir(parents=True)
        (data / "labels" / split).mkdir(parents=True)
    for number in range(6):
        stem = f"xray_{number:05d}"
        Image.fromarray(np.full((120, 160, 3), 128, dtype=np.uint8)).save(data / "images" / "train" / f"{stem}.png")
        (data / "labels" / "train" / f"{stem}.txt").write_text("0 0.1 0.1 0.1 0.1\n")
        images.append({"id": number, "file_name": f"{stem}.png"})
        for k, cid in enumerate(sorted(i["class_id"] for i in load_items())):
            x, y = 10 + (k % 4) * 30, 10 + (k // 4) * 30
            annotations.append({"image_id": number, "category_id": cid, "bbox": [x, y, 20, 20],
                                "segmentation": [[x, y, x + 20, y, x + 20, y + 20, x, y + 20]]})
    (pidray / "annotations" / "xray_train.json").write_text(json.dumps({"images": images, "annotations": annotations}))
    return data, pidray


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def _build(data, pidray, *extra, workers=1):
    return _run("--data", data, "--root", pidray, "--workers", workers, "--count", 8, "--preview", 2, *extra)


def test_builds_links_synthetic_images_labels_and_manifest(tmp_path):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    data, pidray = _dataset(tmp_path, np, Image)

    result = _build(data, pidray)
    assert result.returncode == 0, result.stderr

    out = data / "images" / "train_tip"
    originals = sorted(p for p in out.glob("xray_*.png"))
    synthetic = sorted(out.glob("tip_*.png"))
    assert len(originals) == 6 and all(p.is_symlink() for p in originals)
    assert synthetic and not any(p.is_symlink() for p in synthetic)

    for image in synthetic:
        lines = (data / "labels" / "train_tip" / f"{image.stem}.txt").read_text().splitlines()
        assert len(lines) >= 2 and lines[0] == "0 0.1 0.1 0.1 0.1"  # 배경 라벨을 지키고 더한다
        assert all(0 < float(v) <= 1 for line in lines for v in line.split()[1:])

    manifest = json.loads((data / "train_tip_manifest.json").read_text())
    assert manifest["made"] == len(synthetic) and manifest["originals"] == 6
    assert sum(manifest["pasted"].values()) >= len(synthetic)
    assert len(list((data / "train_tip_preview").glob("*.png"))) == 2

    yaml = (data / "pidray_tip.yaml").read_text()
    assert "train: images/train_tip" in yaml and "val: images/val" in yaml


def test_same_seed_gives_the_same_images_regardless_of_workers(tmp_path):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    data, pidray = _dataset(tmp_path, np, Image)

    assert _build(data, pidray, "--name", "train_a", workers=1).returncode == 0
    assert _build(data, pidray, "--name", "train_b", workers=2).returncode == 0
    a = {p.name: p.read_bytes() for p in (data / "labels" / "train_a").glob("tip_*.txt")}
    b = {p.name: p.read_bytes() for p in (data / "labels" / "train_b").glob("tip_*.txt")}
    assert a and a == b
    for name in a:
        stem = name.removesuffix(".txt")
        assert (data / "images" / "train_a" / f"{stem}.png").read_bytes() == (data / "images" / "train_b" / f"{stem}.png").read_bytes()


def test_refuses_leaks_overwrites_and_bad_names(tmp_path):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    data, pidray = _dataset(tmp_path, np, Image)

    assert _build(data, pidray, "--name", "train").returncode != 0
    assert _build(data, pidray).returncode == 0
    assert _build(data, pidray).returncode != 0
    assert _build(data, pidray, "--overwrite").returncode == 0

    (data / "labels" / "val" / "xray_00000.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    leaked = _build(data, pidray, "--overwrite")
    assert leaked.returncode != 0 and "같은 사진" in leaked.stderr
