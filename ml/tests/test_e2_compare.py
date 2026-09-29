import csv
import random
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "e2_compare.py"
ITEMS = ("Baton", "Gun", "Knife")


def _write(path, image_ids, rng, shift=0.0):
    """품목마다 양성 약 1/3. shift 가 크면 양성 점수가 낮아진다(분포 이동)."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "item", "y_true", "score"])
        for image_id in image_ids:
            present = rng.choice(ITEMS)
            for item in ITEMS:
                y = int(item == present)
                mean = (0.85 - shift) if y else 0.15
                s = min(1.0, max(0.0, rng.gauss(mean, 0.1)))
                w.writerow([image_id, item, y, f"{s:.6f}"])


def _pred_dir(root: Path) -> Path:
    rng = random.Random(0)
    root.mkdir()
    _write(root / "calib.csv", [f"xray_{i:05d}" for i in range(900)], rng)
    _write(root / "test_easy.csv", [f"xray_easy{i:05d}" for i in range(600)], rng)
    _write(root / "test_hard.csv", [f"xray_hard{i:05d}" for i in range(400)], rng)
    _write(root / "test_hidden.csv", [f"xray_hidden{i:05d}" for i in range(400)], rng, shift=0.35)
    return root


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_compares_four_configurations_on_the_same_evaluation_images(tmp_path):
    out = tmp_path / "out"
    result = _run("--pred-dir", _pred_dir(tmp_path / "pred"), "--out-dir", out, "--cap", "0.05")
    assert result.returncode == 0, result.stdout + result.stderr
    summary = (out / "summary.md").read_text()
    for key in "ABCDEF":
        assert f"| {key} |" in summary
    assert "가장 나쁜 칸" in summary
    assert "## 품목별 놓침 / 재검" in summary
    assert "박스 없음" in summary
    # 평가 사진은 시험셋의 절반, 보정 사진과 겹치지 않는다
    calib = {r["image_id"] for r in csv.DictReader((out / "data" / "test_calib.csv").open())}
    evaluation = {r["image_id"] for r in csv.DictReader((out / "data" / "test_eval.csv").open())}
    assert not calib & evaluation
    assert len(calib) + len(evaluation) == 1400


def test_stops_when_prediction_files_are_missing(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _run("--pred-dir", empty, "--out-dir", tmp_path / "out")
    assert result.returncode == 2
    assert "predict_scores.py" in result.stdout
