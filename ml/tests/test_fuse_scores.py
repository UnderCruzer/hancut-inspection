import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fuse_scores.py"
SPLITS = ("calib", "test_easy", "test_hard", "test_hidden")


def _write(folder: Path, gun: float, knife: float):
    folder.mkdir(parents=True)
    for split in SPLITS:
        (folder / f"{split}.csv").write_text(
            "image_id,item,y_true,score\n"
            f"{split}1,Gun,1,{gun}\n{split}1,Knife,0,{knife}\n{split}2,Gun,0,0.3\n{split}2,Knife,1,0.4\n")


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_fuses_every_split_with_max_and_writes_selection(tmp_path):
    _write(tmp_path / "a", 0.2, 0.9)
    _write(tmp_path / "b", 0.8, 0.1)
    out = tmp_path / "out"
    result = _run("--model", f"a={tmp_path / 'a'}", "--model", f"b={tmp_path / 'b'}", "--out", out)
    assert result.returncode == 0, result.stderr
    for split in SPLITS:
        lines = (out / f"{split}.csv").read_text().splitlines()
        assert lines[0] == "image_id,item,y_true,score"
        assert f"{split}1,Gun,1,0.8" in lines and f"{split}1,Knife,0,0.9" in lines
    sel = json.loads((out / "selection.json").read_text())
    assert sel["chosen"]["Gun"] == ["a", "b"]


def test_calib_ap_selection_and_bad_inputs(tmp_path):
    _write(tmp_path / "a", 0.9, 0.1)   # a 는 Gun·Knife 모두 완벽
    _write(tmp_path / "b", 0.1, 0.9)
    out = tmp_path / "out"
    result = _run("--model", f"a={tmp_path / 'a'}", "--model", f"b={tmp_path / 'b'}", "--select", "calib-ap", "--out", out)
    assert result.returncode == 0, result.stderr
    assert json.loads((out / "selection.json").read_text())["chosen"] == {"Gun": ["a"], "Knife": ["a"]}
    assert "추론에 필요한 모델: a (1개)" in (out / "summary.md").read_text()

    assert _run("--model", f"a={tmp_path / 'a'}", "--out", out).returncode != 0
    assert "형식은" in _run("--model", "a", "--model", "b=x", "--out", out).stderr


def test_tiny_dense_scores_are_not_rounded_to_zero(tmp_path):
    for name, gun in (("a", "3.2e-08"), ("b", "1.1e-09")):
        folder = tmp_path / name
        folder.mkdir()
        for split in SPLITS:
            (folder / f"{split}.csv").write_text(f"image_id,item,y_true,score\n{split}1,Gun,1,{gun}\n{split}2,Gun,0,1e-10\n")
    out = tmp_path / "out"
    assert _run("--model", f"a={tmp_path / 'a'}", "--model", f"b={tmp_path / 'b'}", "--out", out).returncode == 0
    assert "calib1,Gun,1,3.2e-08" in (out / "calib.csv").read_text().splitlines()
