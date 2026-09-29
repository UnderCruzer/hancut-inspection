import pytest

from hancut.eval import bags, zones

# 품목 A: low 0.2 / high 0.8, 품목 B: low 0.0 (자동 통과를 열 수 없음) / high 0.9
TABLE = {"default": {"low": 0.2, "high": 0.8}, "A": {"low": 0.2, "high": 0.8}, "B": {"low": 0.0, "high": 0.9}}


def _rows(image_id, a, b, present=()):
    return [{"image_id": image_id, "item": "A", "score": a, "y_true": int("A" in present)},
            {"image_id": image_id, "item": "B", "score": b, "y_true": int("B" in present)}]


def test_any_alarm_makes_the_bag_alarm():
    v = bags.bag_verdicts(_rows("x", 0.95, 0.5, present={"A"}), TABLE)["x"]
    assert v["verdict"] == zones.AUTO_ALARM


def test_review_when_no_alarm_but_some_item_is_uncertain():
    v = bags.bag_verdicts(_rows("x", 0.5, 0.1, present={"A"}), TABLE)["x"]
    assert v["verdict"] == zones.REVIEW
    assert v["flagged"] == {"A", "B"}   # B 는 low=0 이라 늘 재검


def test_an_item_with_low_zero_blocks_every_bag_from_clearing():
    # 품목 B 의 low 가 0 이면 어떤 가방도 모든 품목 auto_clear 가 될 수 없다
    rows = []
    for i in range(20):
        rows += _rows(f"x{i}", 0.05, 0.0)
    summary = bags.bag_summary(bags.bag_verdicts(rows, TABLE))
    assert summary["clear_rate"] == 0.0


def test_clear_only_when_every_item_clears():
    table = {**TABLE, "B": {"low": 0.2, "high": 0.9}}
    v = bags.bag_verdicts(_rows("x", 0.05, 0.1), table)["x"]
    assert v["verdict"] == zones.AUTO_CLEAR and v["flagged"] == set()


def test_summary_rates_and_targeting():
    table = {**TABLE, "B": {"low": 0.2, "high": 0.9}}
    rows = (_rows("alarm", 0.95, 0.1, present={"A"})       # 적발
            + _rows("rev_hit", 0.5, 0.1, present={"A"})    # 재검, A 를 짚고 실제로 A → 적중
            + _rows("rev_miss", 0.5, 0.1, present={"B"})   # 재검, A 만 짚었는데 실제는 B → 빗나감
            + _rows("cleared", 0.05, 0.05, present={"A"})) # 통과인데 A 가 있음 → 놓침
    s = bags.bag_summary(bags.bag_verdicts(rows, table))
    assert s["n"] == 4
    assert s["alarm_rate"] == 0.25 and s["review_rate"] == 0.5 and s["clear_rate"] == 0.25
    assert s["miss_rate"] == 0.25
    assert s["flagged_mean"] == 1.0
    assert s["hit_rate"] == 0.5


def test_rejects_empty():
    with pytest.raises(ValueError):
        bags.bag_summary({})
