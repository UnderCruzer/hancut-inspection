import pytest

from hancut.eval import fusion


def _model(rows):
    return {(img, item): (y, s) for img, item, y, s in rows}


def test_average_precision_perfect_and_ties():
    assert fusion.average_precision([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)
    # 양성 하나가 음성 하나와 같은 점수 → 그 덩어리의 정밀도 1/2
    assert fusion.average_precision([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    # 순서: 0.9(음) 0.8(양) → 정밀도 1/2
    assert fusion.average_precision([0, 1], [0.9, 0.8]) == pytest.approx(0.5)
    assert fusion.average_precision([0, 0], [0.3, 0.1]) == 0.0


def test_check_aligned_rejects_different_rows_or_labels():
    a = _model([("x", "Gun", 1, 0.5)])
    with pytest.raises(ValueError, match="행이 다르다"):
        fusion.check_aligned({"a": a, "b": _model([("y", "Gun", 1, 0.5)])})
    with pytest.raises(ValueError, match="정답이 다르다"):
        fusion.check_aligned({"a": a, "b": _model([("x", "Gun", 0, 0.5)])})
    fusion.check_aligned({"a": a, "b": _model([("x", "Gun", 1, 0.1)])})


def test_fuse_takes_the_max_of_chosen_models_per_item():
    a = _model([("x", "Gun", 1, 0.2), ("x", "Knife", 0, 0.7)])
    b = _model([("x", "Gun", 1, 0.6), ("x", "Knife", 0, 0.1)])
    rows = fusion.fuse({"a": a, "b": b}, {"Gun": ("a", "b"), "Knife": ("b",)})
    assert rows == [{"image_id": "x", "item": "Gun", "y_true": 1, "score": 0.6},
                    {"image_id": "x", "item": "Knife", "y_true": 0, "score": 0.1}]


def test_select_by_ap_prefers_fewer_models_unless_clearly_better():
    # Gun: a 는 완벽, b 는 뒤집힘 → a 하나. Knife: 둘을 합쳐야 완벽 → a+b
    a = _model([("1", "Gun", 1, 0.9), ("2", "Gun", 0, 0.1), ("1", "Knife", 1, 0.9), ("2", "Knife", 1, 0.0), ("3", "Knife", 0, 0.5)])
    b = _model([("1", "Gun", 1, 0.1), ("2", "Gun", 0, 0.9), ("1", "Knife", 1, 0.0), ("2", "Knife", 1, 0.9), ("3", "Knife", 0, 0.5)])
    chosen, table = fusion.select_by_ap({"a": a, "b": b}, ["Gun", "Knife"])
    assert chosen["Gun"] == ("a",)
    assert chosen["Knife"] == ("a", "b")
    assert table["Knife"][("a", "b")] == pytest.approx(1.0)


def test_select_by_ap_keeps_the_smaller_combo_on_a_tie():
    a = _model([("1", "Gun", 1, 0.9), ("2", "Gun", 0, 0.1)])
    b = _model([("1", "Gun", 1, 0.8), ("2", "Gun", 0, 0.2)])
    chosen, _ = fusion.select_by_ap({"a": a, "b": b}, ["Gun"])
    assert chosen["Gun"] == ("a",)
