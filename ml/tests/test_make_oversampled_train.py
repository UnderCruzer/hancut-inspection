import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_oversampled_train.py"


def _yolo(root: Path) -> Path:
    """make_yolo_dataset.py 가 만든 것과 같은 모양의 train 세 장: 총기 / 칼+가위 / 경봉."""
    img, lbl = root / "images" / "train", root / "labels" / "train"
    img.mkdir(parents=True)
    lbl.mkdir(parents=True)
    for stem, classes in [("xray_00000", [6]), ("xray_00001", [10, 4]), ("xray_00002", [0])]:
        src = root / "raw" / f"{stem}.png"
        src.parent.mkdir(exist_ok=True)
        src.write_bytes(b"\x89PNG")
        (img / f"{stem}.png").symlink_to(src)
        (lbl / f"{stem}.txt").write_text("".join(f"{c} 0.5 0.5 0.1 0.1\n" for c in classes))
    (root / "images" / "val").mkdir()
    return root


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_repeats_confusable_images_under_new_names(tmp_path):
    data = _yolo(tmp_path / "yolo")
    result = _run("--data", data)
    assert result.returncode == 0, result.stderr

    out = data / "images" / "train_os"
    stems = sorted(p.stem for p in out.glob("*.png"))
    assert stems == ["xray_00000", "xray_00000__r1", "xray_00000__r2",
                     "xray_00001", "xray_00001__r1", "xray_00001__r2", "xray_00002"]
    # 링크는 원본 파일을 가리키고, 라벨은 원본 그대로다
    assert (out / "xray_00001__r2.png").resolve() == (tmp_path / "yolo" / "raw" / "xray_00001.png").resolve()
    assert (data / "labels" / "train_os" / "xray_00001__r2.txt").read_text() == "10 0.5 0.5 0.1 0.1\n4 0.5 0.5 0.1 0.1\n"
    assert "사진 3 → 7장" in result.stdout

    yaml = (data / "pidray_os.yaml").read_text()
    assert "train: images/train_os" in yaml and "val: images/val" in yaml


def test_refuses_to_overwrite_without_flag_and_rejects_bad_specs(tmp_path):
    data = _yolo(tmp_path / "yolo")
    assert _run("--data", data).returncode == 0
    assert _run("--data", data).returncode != 0
    assert _run("--data", data, "--overwrite", "--repeat", "Knife=2").returncode == 0
    assert len(list((data / "images" / "train_os").glob("*.png"))) == 4

    assert "읽지 못했다" in _run("--data", data, "--overwrite", "--repeat", "Sword=2").stderr
    assert _run("--data", data, "--name", "train").returncode != 0
