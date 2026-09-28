import pytest

from hancut.eval import scores
from hancut.eval.cli import read_predictions

NAMES = ["Baton", "Gun", "Knife"]


class TestImageScores:
    def test_takes_the_maximum_confidence_per_item(self):
        assert scores.image_scores([(1, 0.3), (1, 0.8), (2, 0.1)], 3) == [0.0, 0.8, 0.1]

    def test_items_without_a_box_score_zero(self):
        assert scores.image_scores([], 3) == [0.0, 0.0, 0.0]

    def test_rejects_out_of_range_class(self):
        with pytest.raises(ValueError, match="범위 밖"):
            scores.image_scores([(3, 0.5)], 3)

    def test_rejects_confidence_outside_unit_interval(self):
        with pytest.raises(ValueError, match="0..1 밖"):
            scores.image_scores([(0, 1.2)], 3)


class TestRows:
    def test_one_row_per_item_with_truth_from_labels(self):
        rows = scores.rows_for_image("x", [(1, 0.9)], present={1, 2}, names=NAMES)
        assert [(r["item"], r["y_true"], r["score"]) for r in rows] == [
            ("Baton", 0, 0.0), ("Gun", 1, 0.9), ("Knife", 1, 0.0)]

    def test_missed_item_still_gets_a_row(self):
        # 검출기가 칼을 못 찾아도 칼 행은 있어야 한다. 없으면 그 놓침이 평가에서 사라진다.
        rows = scores.rows_for_image("x", [], present={2}, names=NAMES)
        knife = next(r for r in rows if r["item"] == "Knife")
        assert knife == {"image_id": "x", "item": "Knife", "y_true": 1, "score": 0.0}


class TestLabelFile:
    def test_reads_class_indices(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("1 0.5 0.5 0.1 0.1\n2 0.3 0.3 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")
        assert scores.read_yolo_classes(p) == {1, 2}

    def test_missing_or_empty_file_is_no_items(self, tmp_path):
        assert scores.read_yolo_classes(tmp_path / "none.txt") == set()
        (tmp_path / "e.txt").write_text("")
        assert scores.read_yolo_classes(tmp_path / "e.txt") == set()


def test_written_predictions_are_readable_by_the_evaluation_cli(tmp_path):
    # 예측 스크립트의 출력이 평가 CLI 에 그대로 들어가야 한다.
    rows = (scores.rows_for_image("a", [(1, 0.9)], {1}, NAMES)
            + scores.rows_for_image("b", [(0, 0.2)], {0}, NAMES))
    path = tmp_path / "pred.csv"
    assert scores.write_predictions(rows, path) == 6
    read = read_predictions(path)
    assert len(read) == 6
    assert {(r["image_id"], r["item"]) for r in read} == {(i, n) for i in "ab" for n in NAMES}
