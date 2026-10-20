import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "f_seeds.py"
spec = importlib.util.spec_from_file_location("f_seeds", SCRIPT)
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)


def _run(seed, auto, miss, over):
    return {"seed": seed, "auto": auto, "miss": miss, "false_alarm": 0.01, "worst_cell": "Gun·hidden",
            "worst_miss": 0.06 if over else 0.02, "worst_n": 150, "over": over}


def test_summarize_mean_range_and_overs():
    s = fs.summarize([_run(0, 0.5, 0.002, False), _run(1, 0.7, 0.004, True)])
    assert s["auto"] == {"mean": pytest.approx(0.6), "min": 0.5, "max": 0.7}
    assert s["over"] == 1 and s["worst_miss_max"] == 0.06 and s["n"] == 2


def test_table_line_format():
    lines = fs.table_lines({"E1": fs.summarize([_run(0, 0.5, 0.002, False), _run(1, 0.7, 0.004, True)])})
    assert lines[2] == "| E1 | 60.0% (50.0%~70.0%) | 0.2%~0.4% | 1.0%~1.0% | 1/2 | 6.0% |"


def test_parse_arms_and_default_delta():
    assert fs.parse_arms(["a=x", "b=y/z"]) == {"a": Path("x"), "b": Path("y/z")}
    with pytest.raises(SystemExit):
        fs.parse_arms(["a"])
    assert fs.CELLS == 36
