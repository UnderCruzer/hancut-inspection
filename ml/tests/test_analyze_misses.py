import importlib.util
from pathlib import Path

from hancut.eval.misses import Pred

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "analyze_misses.py"
spec = importlib.util.spec_from_file_location("analyze_misses", SCRIPT)
am = importlib.util.module_from_spec(spec)
spec.loader.exec_module(am)

NAMES = ["Baton", "Pliers", "Hammer", "Powerbank", "Scissors", "Wrench", "Gun", "Bullet",
         "Sprayer", "HandCuffs", "Knife", "Lighter"]
GUN_BOX = (0.5, 0.5, 0.2, 0.1)


def _setup(tmp_path, monkeypatch):
    data = tmp_path / "yolo"
    for split in ("train", "test_easy", "test_hidden"):
        (data / "labels" / split).mkdir(parents=True)
    (data / "labels" / "train" / "xray_00000.txt").write_text("6 0.5 0.5 0.3 0.1\n")
    # 놓친 총기 3장: 렌치로 착각 / 점수 부족 / 아무것도 없음
    for stem in ("xray_easy00000", "xray_easy00001", "xray_hidden00000"):
        split = "test_hidden" if "hidden" in stem else "test_easy"
        (data / "labels" / split / f"{stem}.txt").write_text("6 0.5 0.5 0.2 0.1\n")
    (data / "labels" / "test_easy" / "xray_easy00009.txt").write_text("6 0.4 0.4 0.3 0.2\n")
    monkeypatch.setattr(am, "DATA", data)

    rows = []
    for stem, gun, wrench in [("xray_easy00000", 0.0, 0.9), ("xray_easy00001", 0.0, 0.1),
                              ("xray_hidden00000", 0.0, 0.0), ("xray_easy00009", 0.8, 0.0)]:
        rows += [{"image_id": stem, "item": "Gun", "y_true": 1, "score": gun},
                 {"image_id": stem, "item": "Wrench", "y_true": 0, "score": wrench}]
    missed = {"Gun": ["xray_easy00000", "xray_easy00001", "xray_hidden00000"]}
    detected = {"Gun": ["xray_easy00009"]}
    preds = {
        "xray_easy00000": [Pred(5, 0.9, GUN_BOX)],        # 렌치로 착각
        "xray_easy00001": [Pred(6, 0.0005, GUN_BOX)],     # 총기지만 하한 아래
        "xray_hidden00000": [],                           # 아무것도 없음
    }
    return data, rows, missed, detected, preds


def test_report_classifies_each_missed_box(tmp_path, monkeypatch):
    data, rows, missed, detected, preds = _setup(tmp_path, monkeypatch)
    report = am.build_report(["Gun"], NAMES, rows, missed, detected, {"Gun": 4}, preds, data / "labels" / "train")
    # 요약 행: 양성 4, 미검출 3 (75%), 네 분류가 1/3 씩 (위치 어긋남 0)
    assert "| Gun | 4 | 3 (75.0%) | 33.3% | 33.3% | 0.0% | 33.3% |" in report
    assert "착각한 품목: Wrench 1" in report
    assert "easy 2 · hard 0 · hidden 1" in report
    assert "| 학습 | 1 |" in report
    assert "| 시험 · 잡음 (score ≥ 0.5) | 1 |" in report
    assert "| 시험 · 놓침 | 3 |" in report


def test_label_path_routes_by_difficulty(monkeypatch, tmp_path):
    monkeypatch.setattr(am, "DATA", tmp_path)
    assert am.label_path("xray_hidden00001") == tmp_path / "labels" / "test_hidden" / "xray_hidden00001.txt"
    assert am.image_path("xray_easy00001") == tmp_path / "images" / "test_easy" / "xray_easy00001.png"
