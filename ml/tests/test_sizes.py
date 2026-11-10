import pytest

from hancut.eval import sizes
from hancut.eval.misses import Pred


def test_side_px_scales_the_long_side_to_the_reference():
    # 1280×448 사진의 0.1×0.5 박스 → 640 기준 64×112 px → 한 변 √(64·112)
    assert sizes.side_px((0.5, 0.5, 0.1, 0.5), 1280, 448) == pytest.approx((64 * 112) ** 0.5)
    assert sizes.side_px((0.5, 0.5, 0.1, 0.1), 448, 448, reference=448) == pytest.approx(44.8)


def test_bins_put_the_edge_in_the_upper_bin_and_reject_bad_edges():
    assert sizes.bin_names() == ["<32", "32~64", "64~96", "≥96"]
    assert [sizes.bin_of(v) for v in (31.9, 32, 64, 200)] == ["<32", "32~64", "64~96", "≥96"]
    for bad in ([], [64, 32], [32, 32]):
        with pytest.raises(ValueError):
            sizes.bin_names(bad)


def test_match_uses_each_prediction_once_and_ignores_low_conf_or_other_items():
    gts = [(0, (0.5, 0.5, 0.2, 0.2)), (0, (0.52, 0.5, 0.2, 0.2)), (1, (0.2, 0.2, 0.1, 0.1))]
    preds = [Pred(0, 0.9, (0.51, 0.5, 0.2, 0.2)),  # 둘 다와 겹치지만 하나만 맞힌다
             Pred(1, 0.1, (0.2, 0.2, 0.1, 0.1)),   # 신뢰도 미달
             Pred(0, 0.8, (0.2, 0.2, 0.1, 0.1))]   # 다른 품목 자리
    assert sizes.match(gts, preds, conf=0.25) == [True, False, False]
    assert sum(sizes.match(gts, preds + [Pred(0, 0.7, (0.5, 0.5, 0.2, 0.2))], conf=0.25)) == 2
    assert sizes.match(gts, [], conf=0.25) == [False, False, False]


def test_recall_table_counts_hits_per_item_and_bin():
    table = sizes.recall_table([("Lighter", "<32", True), ("Lighter", "<32", False), ("Bullet", "≥96", True)])
    assert table == {"Lighter": {"<32": (1, 2)}, "Bullet": {"≥96": (1, 1)}}
