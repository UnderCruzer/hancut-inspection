import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_yolo_dataset.py"


def _pidray(root: Path, n_train=40):
    """PIDray 와 같은 폴더 구조의 작은 가짜 데이터셋."""
    cats = [{"id": i, "name": f"c{i}"} for i in range(1, 13)]
    ann = root / "annotations"
    ann.mkdir(parents=True)

    def make(folder, prefix, n, stem):
        (root / folder).mkdir()
        images, annotations = [], []
        for i in range(n):
            name = f"{prefix}{i:05d}.png"
            (root / folder / name).write_bytes(b"\x89PNG")
            images.append({"id": i, "file_name": name, "width": 600, "height": 448})
            annotations.append({"image_id": i, "category_id": (i % 12) + 1, "bbox": [10, 10, 50, 40]})
        (ann / f"{stem}.json").write_text(json.dumps(
            {"images": images, "annotations": annotations, "categories": cats}), encoding="utf-8")

    make("train", "xray_", n_train, "xray_train")
    make("easy", "xray_easy", 5, "xray_test_easy")
    make("hard", "xray_hard", 4, "xray_test_hard")
    make("hidden", "xray_hidden", 3, "xray_test_hidden")
    return root


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_builds_linked_images_labels_and_yaml(tmp_path):
    root = _pidray(tmp_path / "pidray")
    out = tmp_path / "yolo"
    result = _run("--root", root, "--out", out)
    assert result.returncode == 0, result.stdout + result.stderr

    for split, n in [("test_easy", 5), ("test_hard", 4), ("test_hidden", 3)]:
        assert len(list((out / "images" / split).iterdir())) == n
        assert len(list((out / "labels" / split).iterdir())) == n
    total = sum(len(list((out / "images" / s).iterdir())) for s in ("train", "val", "calib"))
    assert total == 40

    # 이미지는 복사가 아니라 원본으로의 링크다
    link = next((out / "images" / "test_easy").iterdir())
    assert link.is_symlink() and link.resolve().parent == (root / "easy").resolve()

    # 라벨 좌표는 이미지 크기로 정규화돼 있고, 클래스 번호는 0 부터다
    label = (out / "labels" / "test_easy" / "xray_easy00000.txt").read_text().split()
    assert label[0] == "0"
    assert all(0.0 <= float(v) <= 1.0 for v in label[1:])

    yaml = (out / "pidray.yaml").read_text()
    assert "train: images/train" in yaml and "val: images/val" in yaml
    assert "  0: Baton" in yaml and "  6: Gun" in yaml


def test_refuses_to_overwrite_without_the_flag(tmp_path):
    root = _pidray(tmp_path / "pidray")
    out = tmp_path / "yolo"
    assert _run("--root", root, "--out", out).returncode == 0
    again = _run("--root", root, "--out", out)
    assert again.returncode != 0 and "--overwrite" in again.stdout + again.stderr
    assert _run("--root", root, "--out", out, "--overwrite").returncode == 0


def test_overwrite_only_deletes_a_folder_named_yolo(tmp_path):
    # --overwrite 가 엉뚱한 폴더를 통째로 지우지 않게 막는다
    root = _pidray(tmp_path / "pidray")
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "keep.txt").write_text("keep")
    result = _run("--root", root, "--out", precious, "--overwrite")
    assert result.returncode != 0
    assert (precious / "keep.txt").read_text() == "keep"


def test_reports_images_missing_from_disk(tmp_path):
    root = _pidray(tmp_path / "pidray")
    (root / "hidden" / "xray_hidden00001.png").unlink()
    result = _run("--root", root, "--out", tmp_path / "yolo")
    assert result.returncode == 1
    assert "이미지 1장이 없다" in result.stdout
