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


def test_default_caps_match_the_grid_recorded_in_items_json():
    # D12: 측정 가능한 범위에서 정한 격자. 코드 기본값과 설정이 어긋나면 옵션 없이 돌렸을 때 다른 상한이 나온다.
    import inspect
    import json

    from hancut.config import CONFIG_DIR
    from hancut.eval import cli

    grid = sorted(json.loads((CONFIG_DIR / "items.json").read_text(encoding="utf-8"))["miss_rate_caps"]["primary"])
    assert sorted(inspect.signature(zones.sweep).parameters["miss_rate_caps"].default) == grid
    assert sorted(inspect.signature(zones.sweep_by_item).parameters["miss_rate_caps"].default) == grid
    assert sorted(cli.build_parser().get_default("miss_caps")) == grid


class TestAllowedErrors:
    def test_zero_errors_need_299_samples_for_one_percent(self):
        # 1 - 0.05 ** (1 / n) <= 0.01  ⇔  n >= 298.07
        assert zones.allowed_errors(298, 0.01) == -1
        assert zones.allowed_errors(299, 0.01) == 0

    def test_matches_the_binomial_tail(self):
        # n=852, p=0.01: P(X<=3) ≈ 0.030 <= 0.05, P(X<=4) ≈ 0.073 > 0.05
        assert zones.allowed_errors(852, 0.01) == 3

    def test_is_stricter_than_the_sample_rate(self):
        for n, cap in [(852, 0.01), (2778, 0.01), (10000, 0.05)]:
            assert zones.allowed_errors(n, cap) < int(n * cap)

    def test_grows_with_n(self):
        values = [zones.allowed_errors(n, 0.02) for n in (100, 500, 1000, 5000)]
        assert values == sorted(values)

    def test_edges(self):
        assert zones.allowed_errors(0, 0.01) == 0
        assert zones.allowed_errors(50, 1.0) == 50
        assert zones.allowed_errors(50, 0.0) == -1
        with pytest.raises(ValueError):
            zones.allowed_errors(10, 0.01, delta=0.0)


class TestUpperBoundFit:
    def _data(self, n_pos, n_neg, low_pos=0):
        """양성은 대부분 0.9, low_pos 개만 0.1. 음성은 0.2."""
        y = [1] * n_pos + [0] * n_neg
        s = [0.1] * low_pos + [0.9] * (n_pos - low_pos) + [0.2] * n_neg
        return y, s

    def test_default_behaviour_is_unchanged(self):
        y, s = self._data(100, 100, low_pos=1)
        assert zones.fit_thresholds(y, s, 0.01) == zones.fit_thresholds(y, s, 0.01, bound="empirical")

    def test_small_sample_cannot_auto_clear_under_the_upper_bound(self):
        # 양성 100장이면 0 개 틀려도 1% 를 95% 로 보장할 수 없다 → 자동 통과를 비운다
        y, s = self._data(100, 100)
        empirical = zones.fit_thresholds(y, s, 0.01)
        upper = zones.fit_thresholds(y, s, 0.01, bound="upper")
        assert empirical.low > 0.2
        assert upper.low == 0.0

    def test_large_sample_allows_auto_clear(self):
        y, s = self._data(400, 400)
        assert zones.fit_thresholds(y, s, 0.01, bound="upper").low > 0.2

    def test_upper_bound_is_never_more_aggressive(self):
        import random
        rng = random.Random(0)
        for _ in range(30):
            n = rng.randint(50, 800)
            y = [rng.random() < 0.3 for _ in range(n)]
            s = [min(1.0, max(0.0, rng.gauss(0.7 if t else 0.3, 0.2))) for t in y]
            y = [int(t) for t in y]
            if 0 < sum(y) < n:
                e = zones.fit_thresholds(y, s, 0.02)
                u = zones.fit_thresholds(y, s, 0.02, bound="upper")
                assert u.low <= e.low

    def test_rejects_unknown_bound(self):
        with pytest.raises(ValueError, match="bound"):
            zones.fit_thresholds([0, 1], [0.1, 0.9], 0.01, bound="loose")

    def test_sweep_passes_the_bound_through(self):
        y, s = self._data(100, 100)
        rows = zones.sweep(y, s, (0.01,), bound="upper")
        assert rows[0]["low"] == 0.0
