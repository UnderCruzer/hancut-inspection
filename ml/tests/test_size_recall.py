import importlib.util
from pathlib import Path

import pytest

from hancut.eval.misses import Pred

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "size_recall.py"
spec = importlib.util.spec_from_file_location("size_recall", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

NAMES = ["Hammer", "Lighter"]


def test_collect_bins_each_kept_box_and_report_lists_both_scopes(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    data = tmp_path / "yolo"
    for split, text in (("test_hidden", "1 0.5 0.5 0.05 0.1\n0 0.5 0.5 0.5 0.5\n"), ("test_easy", "1 0.5 0.5 0.2 0.4\n")):
        (data / "images" / split).mkdir(parents=True)
        (data / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (640, 448)).save(data / "images" / split / "a.png")
        (data / "labels" / split / "a.txt").write_text(text)

    seen = []

    def preds_for(split, paths):
        seen.append((split, [p.name for p in paths]))
        return {"a": [Pred(1, 0.9, (0.5, 0.5, 0.05, 0.1))]} if split == "test_hidden" else {}

    records = script.collect(data, ["test_hidden", "test_easy"], {1}, NAMES, preds_for, 0.25, (32, 64, 96))
    # 숨김: 32×44.8 → 한 변 약 37.9px, 맞힘. 쉬움: 128×179 → ≥96, 놓침. 망치는 고르지 않아 빠진다
    assert records == [("test_hidden", "Lighter", "32~64", True), ("test_easy", "Lighter", "≥96", False)]
    assert seen == [("test_hidden", ["a.png"]), ("test_easy", ["a.png"])]

    text = script.report(records, ["Lighter"], "E1", Path("w.pt"), 640, 0.25, (32, 64, 96))
    assert "## 숨김 (test_hidden)" in text and "## 시험 전체" in text
    assert "| Lighter | - | 100.0% (1) | - | - | 100.0% (1) |" in text
    assert "| Lighter | - | 100.0% (1) | - | 0.0% (1) | 50.0% (2) |" in text
