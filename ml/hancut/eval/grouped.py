"""
난이도 동시 보장 기준선(설정 E · F, #27)과 가장 나쁜 칸 — e2_compare.py 와 f_seeds.py 가 함께 쓴다.

품목마다 기준선 하나를 easy · hard · hidden 모두에서 동시에 상한을 지키도록 고른다.
"""

from __future__ import annotations

from . import cli, resplit, zones

DIFFICULTIES = ("easy", "hard", "hidden")
FALSE_ALARM_CAP = 0.05


def grouped_table(rows, cap: float, bound: str, delta: float) -> dict:
    """품목마다 기준 하나. easy·hard·hidden 모두에서 동시에 상한을 지키도록 고른다 (#27)."""
    def groups(subset):
        out = {}
        for r in subset:
            y, s = out.setdefault(resplit.difficulty_of(r["image_id"]), ([], []))
            y.append(r["y_true"])
            s.append(r["score"])
        return out

    fit = dict(bound=bound, delta=delta)
    table = {}
    for item in sorted({r["item"] for r in rows}):
        t = zones.fit_thresholds_grouped(groups([r for r in rows if r["item"] == item]), cap, FALSE_ALARM_CAP, **fit)
        table[item] = {"low": t.low, "high": t.high, "miss_rate_cap": cap}
    t = zones.fit_thresholds_grouped(groups(rows), cap, FALSE_ALARM_CAP, **fit)
    table["default"] = {"low": t.low, "high": t.high, "miss_rate_cap": cap}
    return table


def worst_cell(rows, table, items) -> tuple[str, float, int]:
    """품목 × 난이도 칸 가운데 놓침률이 가장 높은 칸. E5 의 판정은 모든 칸이 상한 이하인가다."""
    worst = ("", -1.0, 0)
    for item in items:
        for d in DIFFICULTIES:
            cell = [r for r in rows if r["item"] == item and resplit.difficulty_of(r["image_id"]) == d]
            point = cli.point_for(cell, table) if cell else None
            if point and point["miss_rate"] is not None and point["miss_rate"] > worst[1]:
                worst = (f"{item}·{d}", point["miss_rate"], point["n_threat"])
    return worst
