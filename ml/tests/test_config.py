from hancut.config import items_in, load_items, phase_items

# 2026-09-27 실제 어노테이션(xray_train.json)에서 읽은 대응.
# 논문·README 표기 순서와 완전히 다르다. 순서를 가정했다면 12종 전부 틀렸다 (D5).
REAL_CLASS_IDS = {
    1: "Baton", 2: "Pliers", 3: "Hammer", 4: "Powerbank",
    5: "Scissors", 6: "Wrench", 7: "Gun", 8: "Bullet",
    9: "Sprayer", 10: "HandCuffs", 11: "Knife", 12: "Lighter",
}


def test_dataset_has_12_item_entries_with_unique_names_and_keys():
    items = load_items()
    assert len({i["name"] for i in items}) == len(items) == 12
    assert len({i["key"] for i in items}) == 12


def test_class_ids_match_the_annotation_file_not_the_paper_order():
    actual = {i["class_id"]: i["dataset_name"] for i in load_items()}
    assert actual == REAL_CLASS_IDS


def test_paper_order_would_have_been_wrong():
    # README 는 gun, knife, wrench, … 순으로 적는다. 그 순서였다면 id 1 은 Gun 이다.
    by_id = {i["class_id"]: i["dataset_name"] for i in load_items()}
    assert by_id[1] != "Gun"
    assert by_id[7] == "Gun"


def test_every_item_comes_from_pidray_only():
    # SIXray 는 확보 불가(#7). 다른 출처를 섞지 않는다.
    assert all(i["datasets"] == ["pidray"] for i in load_items())
    assert len(items_in("pidray")) == 12
    assert items_in("sixray") == []


def test_phase_one_now_covers_every_item():
    # phase 1/2 는 PIDray∩SIXray 를 뜻했으나 SIXray 확보 불가로 무의미해졌다.
    assert len(phase_items(1)) == 12


def test_split_counts_sum_to_the_v1_total():
    import json
    from pathlib import Path
    from hancut.config import CONFIG_DIR

    splits = json.loads((CONFIG_DIR / "items.json").read_text(encoding="utf-8"))["splits"]
    counts = {k: v for k, v in splits.items() if isinstance(v, int)}
    assert sum(counts.values()) == 47677
    assert counts["train"] == 29457


def test_device_metadata_is_recorded_as_absent():
    import json
    from pathlib import Path
    from hancut.config import CONFIG_DIR

    devices = json.loads((CONFIG_DIR / "items.json").read_text(encoding="utf-8"))["devices"]
    assert devices["available"] is False
import json
from hancut.config import CONFIG_DIR, load_items


def _cfg():
    return json.loads((CONFIG_DIR / "items.json").read_text(encoding="utf-8"))


def test_every_item_can_measure_the_one_percent_cap():
    # 1% 가 주 격자의 가장 낮은 상한이다. 한 품목이라도 못 재면 격자를 다시 짜야 한다.
    assert all(i["reliable_min_miss_rate"] <= 0.012 for i in load_items())


def test_the_tightest_item_is_baton():
    worst = max(load_items(), key=lambda i: i["reliable_min_miss_rate"])
    assert worst["dataset_name"] == "Baton"


def test_positives_sum_to_the_split_sizes():
    cfg = _cfg()
    items = cfg["items"]
    for split, per_image in [("test_easy", 1.0), ("test_hidden", 1.0)]:
        total = sum(i["positives"][split] for i in items)
        # easy 와 hidden 은 이미지당 정확히 1종이므로 양성 합계 = 이미지 수
        assert total == cfg["splits"][split]


def test_hard_is_the_only_split_with_multiple_items_per_image():
    cfg = _cfg()
    hard = sum(i["positives"]["test_hard"] for i in cfg["items"])
    assert hard > cfg["splits"]["test_hard"]
    assert round(hard / cfg["splits"]["test_hard"], 2) == 2.08


def test_primary_caps_do_not_include_an_unmeasurable_one():
    caps = _cfg()["miss_rate_caps"]["primary"]
    floor = max(i["reliable_min_miss_rate"] for i in load_items())
    assert min(caps) >= floor - 0.002  # Baton 1.17% 로 1% 는 빠듯하지만 성립
    assert 0.001 not in caps           # 0.1% 는 어느 품목도 불가
