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


def test_assign_swaps_only_the_named_items_and_rejects_unknown_names():
    assert fusion.assign(["Gun", "Lighter"], ["E1", "small"], {"Lighter": "small"}) == {"Gun": ("E1",), "Lighter": ("small",)}
    with pytest.raises(ValueError, match="모르는 품목"):
        fusion.assign(["Gun"], ["E1", "small"], {"Sword": "small"})
    with pytest.raises(ValueError, match="모르는 모델"):
        fusion.assign(["Gun"], ["E1", "small"], {"Gun": "big"})


def test_add_keeps_the_first_model_and_adds_the_named_one():
    assert fusion.add(["Gun", "Lighter"], ["E1", "cls"], {"Lighter": "cls"}) == {"Gun": ("E1",), "Lighter": ("E1", "cls")}
    assert fusion.add(["Gun"], ["E1", "cls"], {"Gun": "E1"}) == {"Gun": ("E1",)}
    with pytest.raises(ValueError, match="모르는 모델"):
        fusion.add(["Gun"], ["E1", "cls"], {"Gun": "big"})


def test_neg_z_puts_two_scales_on_the_calib_negatives_and_keeps_order():
    # 같은 순서지만 척도가 다른 두 모델: a 는 음성이 1e-6 근처, b 는 0.3 근처
    a = _model([("1", "Gun", 0, 1e-6), ("2", "Gun", 0, 4e-6), ("3", "Gun", 1, 0.5)])
    b = _model([("1", "Gun", 0, 0.2), ("2", "Gun", 0, 0.4), ("3", "Gun", 1, 0.9)])
    pa, pb = fusion.neg_z_params(a, ["Gun"])["Gun"], fusion.neg_z_params(b, ["Gun"])["Gun"]
    # 맞춘 뒤에는 두 모델의 음성 평균이 모두 0.5 (z = 0) 에 온다
    assert fusion.neg_z(2e-6, pa) == pytest.approx(0.5)
    assert fusion.neg_z(0.2, pb) + fusion.neg_z(0.4, pb) == pytest.approx(1.0)  # 음성 둘이 z = ±1
    vals = [fusion.neg_z(s, pa) for s in (1e-9, 1e-6, 0.5, 1.0)]
    assert vals == sorted(vals) and all(0 <= v <= 1 for v in vals)
    with pytest.raises(ValueError, match="모두 같아"):
        fusion.neg_z_params(_model([("1", "Gun", 0, 0.1), ("2", "Gun", 0, 0.1)]), ["Gun"])
    with pytest.raises(ValueError, match="둘 이상"):
        fusion.neg_z_params(_model([("1", "Gun", 0, 0.1), ("2", "Gun", 1, 0.9)]), ["Gun"])


def test_normalize_touches_only_items_that_mix_models():
    a = _model([("1", "Gun", 0, 0.1), ("2", "Gun", 0, 0.3), ("1", "Knife", 0, 0.2)])
    params = {"a": fusion.neg_z_params(a, ["Gun"])}
    out = fusion.normalize({"a": a}, params, {"Gun": ("a", "b"), "Knife": ("a",)})
    assert out["a"][("1", "Knife")] == (0, 0.2)
    assert out["a"][("1", "Gun")][1] < 0.5 < out["a"][("2", "Gun")][1]
