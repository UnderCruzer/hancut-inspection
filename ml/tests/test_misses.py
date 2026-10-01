import pytest

from hancut.eval import misses
from hancut.eval.misses import Pred

GT = (0.5, 0.5, 0.2, 0.2)
GUN = 6


def test_iou_identical_disjoint_and_half():
    assert misses.iou(GT, GT) == pytest.approx(1.0)
    assert misses.iou(GT, (0.1, 0.1, 0.1, 0.1)) == 0.0
    # 가로로 절반 겹침: 교집합 0.1x0.2, 합집합 0.04+0.04-0.02
    assert misses.iou(GT, (0.6, 0.5, 0.2, 0.2)) == pytest.approx(0.02 / 0.06)


def test_wrong_class_when_another_item_sits_on_the_gun():
    cat, p, v = misses.classify_miss(GT, GUN, [Pred(5, 0.8, GT)])
    assert cat == misses.WRONG_CLASS and p.cls == 5 and v == pytest.approx(1.0)


def test_low_score_when_the_same_item_is_below_the_floor():
    cat, p, _ = misses.classify_miss(GT, GUN, [Pred(GUN, 0.0004, GT), Pred(5, 0.3, (0.1, 0.1, 0.1, 0.1))])
    assert cat == misses.LOW_SCORE and p.conf == 0.0004


def test_misplaced_when_overlap_is_small():
    cat, _, v = misses.classify_miss(GT, GUN, [Pred(GUN, 0.0002, (0.62, 0.5, 0.2, 0.2))])
    assert cat == misses.MISPLACED and 0.1 <= v < 0.5


def test_nothing_when_no_box_is_near():
    assert misses.classify_miss(GT, GUN, [])[0] == misses.NOTHING
    assert misses.classify_miss(GT, GUN, [Pred(3, 0.9, (0.1, 0.1, 0.05, 0.05))])[0] == misses.NOTHING


def test_rejects_a_case_that_was_not_a_miss():
    with pytest.raises(ValueError, match="미검출이 아니다"):
        misses.classify_miss(GT, GUN, [Pred(GUN, 0.4, GT)])


def test_shape_stats_quartiles():
    boxes = [(0.5, 0.5, w, 0.1) for w in (0.1, 0.2, 0.3, 0.4, 0.5)]
    s = misses.shape_stats(boxes)
    assert s["n"] == 5
    assert s["area"][1] == pytest.approx(0.03)
    assert s["aspect"][1] == pytest.approx(3.0)
    assert misses.shape_stats([]) == {"n": 0}


def test_top_alternatives_counts_what_scored_instead():
    rows = []
    for i, (alt, score) in enumerate([("Wrench", 0.9), ("Wrench", 0.7), ("Knife", 0.3), ("Pliers", 0.1)]):
        img = f"x{i}"
        rows += [{"image_id": img, "item": "Gun", "y_true": 1, "score": 0.0},
                 {"image_id": img, "item": alt, "y_true": 0, "score": score}]
    # 잡힌 총기는 세지 않는다
    rows += [{"image_id": "hit", "item": "Gun", "y_true": 1, "score": 0.8},
             {"image_id": "hit", "item": "Wrench", "y_true": 0, "score": 0.9}]
    counts, missed = misses.top_alternatives(rows, "Gun")
    assert missed == 4
    assert counts == {"Wrench": 2, "Knife": 1, "없음": 1}
