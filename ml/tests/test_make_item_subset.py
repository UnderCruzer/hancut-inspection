import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from hancut.config import load_items
from hancut.data import yolo

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_item_subset.py"
spec = importlib.util.spec_from_file_location("make_item_subset", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

NAMES = yolo.class_names(load_items())
LIGHTER, BULLET, HAMMER = NAMES.index("Lighter"), NAMES.index("Bullet"), NAMES.index("Hammer")


def test_keep_lines_leaves_only_chosen_items_verbatim():
    text = f"{HAMMER} 0.5 0.5 0.2 0.2\n{LIGHTER} 0.1 0.1 0.05 0.05\n\n"
    assert script.keep_lines(text, {LIGHTER}) == [f"{LIGHTER} 0.1 0.1 0.05 0.05"]
    assert script.keep_lines(text, {BULLET}) == []


def _dataset(root: Path) -> Path:
    data = root / "yolo"
    labels = {"train": {"a": f"{HAMMER} 0.5 0.5 0.2 0.2\n{LIGHTER} 0.1 0.1 0.05 0.05\n", "b": f"{HAMMER} 0.5 0.5 0.2 0.2\n"},
              "val": {"c": f"{BULLET} 0.3 0.3 0.02 0.02\n"}}
    for split, files in labels.items():
        (data / "images" / split).mkdir(parents=True)
        (data / "labels" / split).mkdir(parents=True)
        for stem, text in files.items():
            (data / "images" / split / f"{stem}.png").write_bytes(b"png")
            (data / "labels" / split / f"{stem}.txt").write_text(text)
    return data


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_builds_linked_subsets_with_background_images_and_yaml(tmp_path):
    data = _dataset(tmp_path)
    result = _run("--data", data, "--items", "Lighter", "Bullet")
    assert result.returncode == 0, result.stderr

    images = sorted((data / "images" / "train_small").iterdir())
    assert [p.name for p in images] == ["a.png", "b.png"] and all(p.is_symlink() for p in images)
    assert (data / "labels" / "train_small" / "a.txt").read_text() == f"{LIGHTER} 0.1 0.1 0.05 0.05\n"
    assert (data / "labels" / "train_small" / "b.txt").read_text() == ""  # 망치만 든 사진은 배경으로 남는다
    assert (data / "labels" / "val_small" / "c.txt").read_text() == f"{BULLET} 0.3 0.3 0.02 0.02\n"

    yaml = (data / "pidray_small.yaml").read_text()
    assert "train: images/train_small" in yaml and "val: images/val_small" in yaml
    assert f"  {LIGHTER}: Lighter" in yaml and yaml.count("\n  ") == len(NAMES)  # 12종 번호 유지
    manifest = json.loads((data / "small_manifest.json").read_text())
    assert manifest["splits"]["train"] == {"images": 2, "with_items": 1, "boxes": {str(LIGHTER): 1}}


def test_refuses_unknown_items_bad_suffix_and_silent_overwrite(tmp_path):
    data = _dataset(tmp_path)
    assert "모르는 품목" in _run("--data", data, "--items", "Sword").stderr
    assert _run("--data", data, "--items", "Lighter", "--suffix", "../x").returncode != 0
    assert _run("--data", data, "--items", "Lighter").returncode == 0
    assert "--overwrite" in _run("--data", data, "--items", "Lighter").stderr
    assert _run("--data", data, "--items", "Lighter", "--overwrite").returncode == 0
