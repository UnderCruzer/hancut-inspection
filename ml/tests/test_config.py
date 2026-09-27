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
