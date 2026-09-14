from hancut.config import load_facilities, phase_facilities


def test_dataset_has_24_facility_entries_with_unique_names():
    names = [f["name"] for f in load_facilities()]
    assert len(names) == len(set(names))
    # 23 facility types, with the hydrant cabinet split into closed/open
    assert len(names) == 24


def test_phase_one_is_the_eight_most_inspected_types():
    assert phase_facilities(1) == [
        "소형소화기", "스프링클러헤드", "열감지기", "연기감지기",
        "피난구유도등", "통로유도등", "비상조명등", "방화문",
    ]


def test_phase_two_includes_everything():
    assert len(phase_facilities(2)) == 24
