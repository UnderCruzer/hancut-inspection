from hancut.config import items_in, load_items, phase_items


def test_dataset_has_12_item_entries_with_unique_names_and_keys():
    items = load_items()
    assert len({i["name"] for i in items}) == len(items) == 12
    assert len({i["key"] for i in items}) == 12


def test_class_ids_are_unset_until_the_annotations_are_read():
    # D5: 논문 표기 순서를 class_id 로 가정하지 않는다.
    assert all(i["class_id"] is None for i in load_items())


def test_phase_one_is_the_five_items_shared_by_pidray_and_sixray():
    assert phase_items(1) == ["총기", "칼", "렌치", "펜치", "가위"]


def test_phase_two_includes_every_pidray_item():
    assert len(phase_items(2)) == 12


def test_sixray_covers_phase_one_plus_the_undersampled_hammer():
    assert items_in("sixray") == phase_items(1) + ["망치"]


def test_opixray_is_knives_only():
    assert items_in("opixray") == ["칼"]
