import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_detector.py"
spec = importlib.util.spec_from_file_location("eval_detector", SCRIPT)
ed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ed)

NAMES = ["Baton", "Pliers"]


def _result(images, p, per):
    return {"images": images, "all": {"p": p, "r": 0.9, "map50": 0.95, "map": 0.8}, "per_class": per}


def test_report_has_overall_and_per_class_tables():
    results = {
        "val": _result(1474, 0.98, {0: {"p": 0.99, "r": 0.97, "map50": 0.98, "map": 0.9}, 1: {"p": 0.9, "r": 0.8, "map50": 0.85, "map": 0.7}}),
        "test_hidden": _result(5005, 0.7, {0: {"p": 0.5, "r": 0.4, "map50": 0.45, "map": 0.3}}),
    }
    report = ed.build_report("w.pt", results, NAMES)
    assert "| val (학습 분포) | 1,474 | 0.980 | 0.900 | 0.950 | 0.800 |" in report
    assert "| 시험 · 숨김 | 5,005 | 0.700 |" in report
    # 품목별 mAP50 표: 숨김에 펜치가 없으면 '-'
    assert "| Pliers | 0.850 | - |" in report


def test_split_yaml_points_val_at_the_split(tmp_path):
    text = ed.split_yaml(tmp_path / "s.yaml", "test_hard", NAMES).read_text()
    assert "val: images/test_hard" in text and "train: images/train" in text and "  1: Pliers" in text
