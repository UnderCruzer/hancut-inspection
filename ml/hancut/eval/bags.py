"""
품목 판정 → 가방 판정 (#29).

판독관은 품목이 아니라 가방을 본다. 가방 판정은 품목 판정을 합쳐서 정한다.

    적발  품목 하나라도 auto_alarm
    재검  적발이 아니고, 품목 하나라도 review
    통과  모든 품목이 auto_clear

PIDray v1 의 test 는 전부 위해물품이 든 가방이다. 그래서 가방 '통과'는 곧 놓침이고,
빈 가방을 얼마나 통과시키는지는 이 데이터로 잴 수 없다.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from . import zones
from .cli import thresholds_for


def bag_verdicts(rows: Iterable[Mapping], table: Mapping) -> dict[str, dict]:
    """이미지마다 {verdict, flagged(재검·적발로 짚은 품목), present(실제로 든 품목)}."""
    per: dict[str, dict] = defaultdict(lambda: {"zones": {}, "present": set()})
    for r in rows:
        entry = per[r["image_id"]]
        entry["zones"][r["item"]] = zones.zone_of(r["score"], thresholds_for(table, r["item"]))
        if r["y_true"] == 1:
            entry["present"].add(r["item"])

    out = {}
    for image_id, entry in per.items():
        decided = set(entry["zones"].values())
        if zones.AUTO_ALARM in decided:
            verdict = zones.AUTO_ALARM
        elif zones.REVIEW in decided:
            verdict = zones.REVIEW
        else:
            verdict = zones.AUTO_CLEAR
        flagged = {item for item, z in entry["zones"].items() if z != zones.AUTO_CLEAR}
        out[image_id] = {"verdict": verdict, "flagged": flagged, "present": entry["present"]}
    return out


def bag_summary(verdicts: Mapping[str, Mapping]) -> dict:
    """
    가방 단위 요약.

    miss_rate      위해물품이 든 가방 가운데 '통과'된 비율
    flagged_mean   재검 가방에서 판독관에게 짚어 준 품목 수의 평균 (적을수록 볼 곳이 좁혀진다)
    hit_rate       재검 가방에서 실제로 든 품목이 모두 짚은 품목 안에 있는 비율
    """
    n = len(verdicts)
    if n == 0:
        raise ValueError("가방이 없다")
    values = list(verdicts.values())
    count = {v: sum(b["verdict"] == v for b in values) for v in (zones.AUTO_ALARM, zones.REVIEW, zones.AUTO_CLEAR)}
    with_threat = [b for b in values if b["present"]]
    missed = sum(b["verdict"] == zones.AUTO_CLEAR for b in with_threat)
    reviewed = [b for b in values if b["verdict"] == zones.REVIEW]
    reviewed_with_threat = [b for b in reviewed if b["present"]]
    return {
        "n": n,
        "alarm_rate": count[zones.AUTO_ALARM] / n,
        "review_rate": count[zones.REVIEW] / n,
        "clear_rate": count[zones.AUTO_CLEAR] / n,
        "miss_rate": missed / len(with_threat) if with_threat else None,
        "n_review": len(reviewed),
        "flagged_mean": sum(len(b["flagged"]) for b in reviewed) / len(reviewed) if reviewed else None,
        "hit_rate": (sum(b["present"] <= b["flagged"] for b in reviewed_with_threat) / len(reviewed_with_threat)
                     if reviewed_with_threat else None),
    }
