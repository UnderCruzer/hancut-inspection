import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "predict_dense_scores.py"
spec = importlib.util.spec_from_file_location("predict_dense_scores", SCRIPT)
pds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pds)


def test_help_runs_without_the_inference_stack():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "dense" in result.stdout


def test_p0_table_shows_both_rules():
    nms = [{"image_id": "a", "item": "Gun", "y_true": 1, "score": 0.0}, {"image_id": "b", "item": "Gun", "y_true": 1, "score": 0.4}]
    dense = [{"image_id": "a", "item": "Gun", "y_true": 1, "score": 0.02}, {"image_id": "b", "item": "Gun", "y_true": 1, "score": 0.5}]
    table = pds.p0_table(nms, dense, ["Gun", "Knife"])
    assert "| Gun | 2 | 1 (50.0%) | 0 (0.0%) |" in table
    assert "| Knife | 0 | - | - |" in table
