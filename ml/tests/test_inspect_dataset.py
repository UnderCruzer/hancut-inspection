import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "inspect_dataset.py"

CATEGORIES = ["Baton", "Pliers", "Hammer", "Powerbank", "Scissors", "Wrench",
              "Gun", "Bullet", "Sprayer", "HandCuffs", "Knife", "Lighter"]


def _dataset(root: Path, *, device_field=None, annotations=None) -> Path:
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
        "annotations": annotations or [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CATEGORIES)],
    }), encoding="utf-8")
    return root


def _run(path: Path) -> str:
    out = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                         capture_output=True, text=True, check=True)
    return out.stdout


def test_reports_class_ids_so_items_json_can_be_filled(tmp_path):
    report = _run(_dataset(tmp_path / "ds"))
    assert "| 1 | Baton |" in report
    assert "| 12 | Lighter |" in report


def test_finds_the_device_field_when_it_exists(tmp_path):
    report = _run(_dataset(tmp_path / "ds", device_field="device_id"))
    assert "장비 후보 필드: **device_id**" in report


def test_says_so_when_there_is_no_device_field(tmp_path):
    # PIDray 가 실제로 이 경우다. 조용히 넘기지 않는다.
    report = _run(_dataset(tmp_path / "ds"))
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


def test_ignores_macos_archive_junk(tmp_path):
    # 맥에서 압축하면 __MACOSX/ 와 ._ 파일이 섞인다. 판본 판정을 어긋나게 한다.
    root = _dataset(tmp_path / "ds")
    junk = root / "__MACOSX" / "pidray"
    junk.mkdir(parents=True)
    for i in range(50):
        (junk / f"._xray_easy{i:05d}.png").write_bytes(b"junk")
    (root / ".DS_Store").write_bytes(b"junk")
    report = _run(root)
    # 판본 판정은 부산물을 빼고 센다
    assert "이미지 **11장**" in report
    # 조용히 버리지 않고 몇 개를 뺐는지 밝힌다
    assert "맥 압축 부산물 **51개**를 세지 않았다" in report
    # 트리 집계에는 섞이지 않는다
    assert "| `__MACOSX/` |" not in report


def test_reports_per_item_positives_and_the_measurable_floor(tmp_path):
    root = _dataset(tmp_path / "ds", annotations=[
        {"image_id": 0, "category_id": 7},
        {"image_id": 1, "category_id": 7},
        {"image_id": 1, "category_id": 11},
    ])
    report = _run(root)
    assert "품목별 양성 장수" in report
    assert "| Gun |" in report
    # Gun 은 시험셋 양성 2장 → 1/2 = 50%
    assert "50.00%" in report
    # 양성이 하나도 없는 품목은 측정 불가
    assert "측정 불가" in report


def test_exits_with_an_error_for_a_missing_directory(tmp_path):
    out = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "nope")],
                         capture_output=True, text=True)
    assert out.returncode == 1
