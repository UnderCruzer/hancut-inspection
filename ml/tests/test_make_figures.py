import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py"
spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)


def test_committed_numbers_are_consistent():
    data = mf.load_results()
    assert data["method_progression"]["rows"][3]["label"].startswith("E1 dense")
    assert data["method_progression"]["rows"][3]["mean"] == 57.8


def test_committed_parity_has_no_zone_flips():
    s = mf.parity_summary(mf.load_parity())
    assert s == {**s, "rows": 432, "images": 36, "flips": 0}
    assert s["max_rel"] < 0.05


def test_rejects_auto_rate_that_disagrees_with_review_count(tmp_path):
    data = json.loads((mf.DATA / "results.json").read_text(encoding="utf-8"))
    data["e2_by_cap"]["rows"][0]["auto"] = 50.0
    path = tmp_path / "r.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="auto rate"):
        mf.load_results(path)


def test_parity_summary_counts_flips_and_ignores_tiny_scores():
    rows = [
        {"image": "a", "server_score": 0.5, "browser_score": 0.55, "server_zone": "review", "browser_zone": "review"},
        {"image": "a", "server_score": 1e-6, "browser_score": 9e-6, "server_zone": "auto_clear", "browser_zone": "review"},
    ]
    s = mf.parity_summary(rows)
    assert s["flips"] == 1 and s["images"] == 1
    assert s["max_rel"] == pytest.approx(0.1)


def test_draws_all_figures(tmp_path):
    pytest.importorskip("matplotlib")
    paths = mf.make_all(tmp_path)
    assert [p.name for p in paths] == [
        "fig1_methods.png", "fig2_calibration.png", "fig3_detector_map.png", "fig4_review_rate.png", "fig5_browser_parity.png",
    ]
    assert all(p.stat().st_size > 10_000 for p in paths)
