import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "inspect_dataset.py"

CATEGORIES = ["gun", "knife", "wrench", "pliers", "scissors", "hammer",
              "handcuffs", "baton", "sprayer", "powerbank", "lighter", "bullet"]


def _dataset(root: Path, *, device_field: str | None = "device_id") -> Path:
    for split, n in [("train", 4), ("test/easy", 3), ("test/hard", 2), ("test/hidden", 2)]:
        d = root / split
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"xray_{split.split('/')[-1]}{i:05d}.png").write_bytes(b"\x89PNG" + b"0" * 64)
    images = []
    for i in range(3):
        img = {"id": i, "file_name": f"xray_easy{i:05d}.png", "width": 1200, "height": 900}
        if device_field:
            img[device_field] = "ABC"[i % 3]
        images.append(img)
    ann = root / "annotations"
    ann.mkdir()
    (ann / "xray_test_easy.json").write_text(json.dumps({
        "images": images,
        "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CATEGORIES)],
    }), encoding="utf-8")
    return root


def _run(path: Path) -> str:
    out = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                         capture_output=True, text=True, check=True)
    return out.stdout


def test_reports_class_ids_so_items_json_can_be_filled(tmp_path):
    report = _run(_dataset(tmp_path / "ds"))
    assert "| 1 | gun |" in report
    assert "| 12 | bullet |" in report


def test_finds_the_device_field_when_it_exists(tmp_path):
    report = _run(_dataset(tmp_path / "ds"))
    assert "장비 후보 필드: **device_id**" in report


def test_says_so_when_there_is_no_device_field(tmp_path):
    # E4(장비 교차)가 성립하지 않는 경우를 조용히 넘기지 않는다.
    report = _run(_dataset(tmp_path / "ds", device_field=None))
    assert "장비 후보 필드: **없음**" in report
    assert "E4(장비 교차)는 성립하지 않는다" in report


def test_counts_difficulty_split_files(tmp_path):
    report = _run(_dataset(tmp_path / "ds"))
    for level in ("easy", "hard", "hidden"):
        assert f"| {level} |" in report


def test_flags_an_image_count_that_matches_no_known_edition(tmp_path):
    report = _run(_dataset(tmp_path / "ds"))
    assert "이미지 **11장**" in report
    assert "알려진 수치와 다르다" in report


def test_exits_with_an_error_for_a_missing_directory(tmp_path):
    out = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "nope")],
                         capture_output=True, text=True)
    assert out.returncode == 1
