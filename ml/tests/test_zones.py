import pytest

from hancut.eval import zones
from hancut.eval.zones import Thresholds


def _perfect(n=100):
    """양성은 높은 점수, 음성은 낮은 점수 — 완전히 분리된 이상적인 모델."""
    y = [1] * (n // 2) + [0] * (n // 2)
    s = [0.9] * (n // 2) + [0.1] * (n // 2)
    return y, s


class TestZoneOf:
    def test_three_zones_by_threshold(self):
        t = Thresholds(low=0.2, high=0.8)
        assert zones.zone_of(0.1, t) == zones.AUTO_CLEAR
        assert zones.zone_of(0.5, t) == zones.REVIEW
        assert zones.zone_of(0.9, t) == zones.AUTO_ALARM

    def test_boundaries_are_low_inclusive_high_inclusive(self):
        t = Thresholds(low=0.2, high=0.8)
        assert zones.zone_of(0.2, t) == zones.REVIEW
        assert zones.zone_of(0.8, t) == zones.AUTO_ALARM

    def test_thresholds_must_be_ordered(self):
        with pytest.raises(ValueError):
            Thresholds(low=0.9, high=0.1)


class TestEvaluate:
    def test_perfect_separation_has_no_miss_and_no_review(self):
        y, s = _perfect()
        point = zones.evaluate(y, s, Thresholds(low=0.5, high=0.5))
        assert point.miss_rate == 0.0
        assert point.review_rate == 0.0
        assert point.false_alarm_rate == 0.0
        assert point.auto_rate == 1.0

    def test_miss_rate_counts_threat_sent_to_auto_clear(self):
        y = [1, 1, 1, 1, 0]
        s = [0.1, 0.9, 0.9, 0.9, 0.1]  # 양성 1건이 낮은 점수
        point = zones.evaluate(y, s, Thresholds(low=0.5, high=0.8))
        assert point.miss_rate == 0.25
        assert point.n_threat == 4

    def test_false_alarm_rate_is_over_clear_items_only(self):
        y = [0, 0, 1]
        s = [0.95, 0.1, 0.95]
        point = zones.evaluate(y, s, Thresholds(low=0.2, high=0.9))
        assert point.false_alarm_rate == 0.5

    def test_review_rate_and_auto_rate_are_complementary(self):
        y = [1, 0, 1, 0]
        s = [0.5, 0.5, 0.95, 0.05]
        point = zones.evaluate(y, s, Thresholds(low=0.2, high=0.8))
        assert point.review_rate == 0.5
        assert point.auto_rate == 0.5

    def test_rejects_reversed_label_convention(self):
        with pytest.raises(ValueError):
            zones.evaluate([2, 0], [0.5, 0.5], Thresholds(low=0.2, high=0.8))

    def test_rejects_scores_outside_probability_range(self):
        with pytest.raises(ValueError):
            zones.evaluate([1, 0], [1.5, 0.5], Thresholds(low=0.2, high=0.8))

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            zones.evaluate([], [], Thresholds(low=0.2, high=0.8))


class TestFitThresholds:
    def test_respects_the_miss_rate_cap(self):
        # 양성 10건 중 1건만 0.3 으로 낮게 예측된 모델
        y = [1] * 10 + [0] * 10
        s = [0.3] + [0.9] * 9 + [0.1] * 10
        t = zones.fit_thresholds(y, s, max_miss_rate=0.0)
        assert zones.evaluate(y, s, t).miss_rate == 0.0
        assert t.low <= 0.3  # 0.3 을 자동 통과로 보내지 않는다

    def test_higher_cap_allows_more_automation(self):
        y = [1] * 10 + [0] * 10
        s = [0.3] + [0.9] * 9 + [0.1] * 10
        strict = zones.evaluate(y, s, zones.fit_thresholds(y, s, 0.0))
        loose = zones.evaluate(y, s, zones.fit_thresholds(y, s, 0.1))
        assert loose.review_rate <= strict.review_rate
        assert loose.miss_rate >= strict.miss_rate

    def test_false_alarm_cap_pushes_high_up(self):
        y = [1, 1, 0, 0]
        s = [0.5, 0.9, 0.7, 0.1]  # 음성 1건이 0.7 로 높게 예측됨
        allowed = zones.fit_thresholds(y, s, 0.0, max_false_alarm_rate=0.5)
        blocked = zones.fit_thresholds(y, s, 0.0, max_false_alarm_rate=0.0)
        assert allowed.high <= 0.7 < blocked.high

    def test_high_never_drops_below_low_even_if_it_costs_false_alarms(self):
        # 놓침 허용이 0이면 low 가 0.9 까지 올라가고, high 는 그 아래로 못 내려간다.
        # 오경보 상한이 사실상 적용되지 않는 경우 — 재검 구간이 비어 버린다.
        y = [1, 1, 0, 0]
        s = [0.9, 0.9, 0.7, 0.1]
        t = zones.fit_thresholds(y, s, 0.0, max_false_alarm_rate=0.0)
        assert t.low == t.high == 0.9
        assert zones.evaluate(y, s, t).review_rate == 0.0

    def test_unreachable_cap_leaves_the_auto_zone_empty(self):
        # 모든 양성이 0.0 — 어떤 low 를 써도 놓침률을 0 으로 만들 수 없다
        y = [1, 0]
        s = [0.0, 0.0]
        t = zones.fit_thresholds(y, s, max_miss_rate=0.0)
        assert t.low == 0.0
        assert zones.evaluate(y, s, t).miss_rate == 0.0

    def test_rejects_invalid_cap(self):
        y, s = _perfect(4)
        with pytest.raises(ValueError):
            zones.fit_thresholds(y, s, max_miss_rate=1.5)


class TestSweep:
    def test_one_row_per_cap_and_caps_are_met(self):
        y = [1] * 20 + [0] * 20
        s = [0.2, 0.3] + [0.9] * 18 + [0.1] * 20
        rows = zones.sweep(y, s, miss_rate_caps=(0.0, 0.05, 0.1))
        assert [r["miss_rate_cap"] for r in rows] == [0.0, 0.05, 0.1]
        for row in rows:
            assert row["miss_rate"] <= row["miss_rate_cap"] + 1e-9

    def test_by_item_groups_independently(self):
        items = ["Knife", "Knife", "Lighter", "Lighter"]
        y = [1, 0, 1, 0]
        s = [0.6, 0.4, 0.99, 0.01]
        result = zones.sweep_by_item(items, y, s, miss_rate_caps=(0.0,))
        assert set(result) == {"Knife", "Lighter"}
        # 잘 분리된 Lighter 가 Knife 보다 재검률이 낮다
        assert result["Lighter"][0]["review_rate"] <= result["Knife"][0]["review_rate"]

    def test_by_item_rejects_length_mismatch(self):
        with pytest.raises(ValueError):
            zones.sweep_by_item(["Knife"], [1, 0], [0.5, 0.5])

    def test_format_table_renders_every_row(self):
        y, s = _perfect(10)
        table = zones.format_table(zones.sweep(y, s))
        assert table.count("\n") == 4  # 헤더 + 구분선 + 상한 3개
        assert "놓침률 상한" in table
