import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "image_ap.py"


def _write(folder: Path, gun_scores):
    folder.mkdir(parents=True)
    rows = "".join(f"{i},Gun,{y},{s}\n" for i, (y, s) in enumerate(gun_scores))
    (folder / "test_hidden.csv").write_text("image_id,item,y_true,score\n" + rows)


def test_reports_image_ap_per_arm_and_stops_on_missing_items(tmp_path):
    _write(tmp_path / "a", [(1, 0.9), (0, 0.5), (1, 0.4)])   # AP = (1 + 2/3) / 2
    _write(tmp_path / "b", [(1, 0.9), (0, 0.1), (1, 0.4)])   # AP = 1
    out = tmp_path / "ap.md"
    run = lambda *a: subprocess.run([sys.executable, str(SCRIPT), "--arm", f"a={tmp_path / 'a'}", "--arm", f"b={tmp_path / 'b'}",  # noqa: E731
                                     "--splits", "test_hidden", "--out", out, *a], capture_output=True, text=True)
    result = run("--items", "Gun")
    assert result.returncode == 0, result.stderr
    assert "| Gun | 2 | 0.833 | 1.000 |" in out.read_text()
    assert "Knife 의 행이 없다" in run("--items", "Knife").stderr
