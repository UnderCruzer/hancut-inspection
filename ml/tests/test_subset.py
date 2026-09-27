import pytest

from hancut.data import subset
from hancut.data.subset import Record


def _records(item: str, clear: bool, n: int, prefix: str = "") -> list[Record]:
    return [
        Record(image_id=f"{prefix}{item}-{int(clear)}-{i:04d}", item=item, clear=clear)
        for i in range(n)
    ]


FACILITIES = ["Gun", "Knife"]


class TestStratifiedSubset:
    def test_takes_up_to_the_quota_from_each_stratum(self):
        records = _records("Gun", True, 50) + _records("Gun", False, 50)
        picked = subset.stratified_subset(records, ["Gun"], per_stratum=10, seed=1)
        summary = subset.summarize(picked)
        assert summary == [{"item": "Gun", "clear": 10, "threat": 10, "total": 20}]

    def test_drops_items_outside_phase_one(self):
        records = _records("Gun", True, 5) + _records("완강기", True, 5)
        picked = subset.stratified_subset(records, ["Gun"], per_stratum=5, seed=1)
        assert {r.item for r in picked} == {"Gun"}

    def test_takes_everything_when_a_stratum_is_short(self):
        records = _records("Knife", True, 10) + _records("Knife", False, 3)
        picked = subset.stratified_subset(records, ["Knife"], per_stratum=10, seed=1)
        summary = subset.summarize(picked)
        assert summary[0]["threat"] == 3
        assert subset.shortfalls(summary, 10) == ["Knife 적발 3/10"]

    def test_same_seed_gives_the_same_subset(self):
        records = _records("Gun", True, 100)
        a = subset.stratified_subset(records, FACILITIES, per_stratum=10, seed=42)
        b = subset.stratified_subset(records, FACILITIES, per_stratum=10, seed=42)
        assert [r.image_id for r in a] == [r.image_id for r in b]

    def test_different_seed_gives_a_different_subset(self):
        records = _records("Gun", True, 100)
        a = subset.stratified_subset(records, FACILITIES, per_stratum=10, seed=1)
        b = subset.stratified_subset(records, FACILITIES, per_stratum=10, seed=2)
        assert [r.image_id for r in a] != [r.image_id for r in b]

    def test_input_order_does_not_change_the_result(self):
        records = _records("Gun", True, 40)
        shuffled = list(reversed(records))
        a = subset.stratified_subset(records, FACILITIES, per_stratum=8, seed=5)
        b = subset.stratified_subset(shuffled, FACILITIES, per_stratum=8, seed=5)
        assert sorted(r.image_id for r in a) == sorted(r.image_id for r in b)

    def test_rejects_zero_quota(self):
        with pytest.raises(ValueError):
            subset.stratified_subset([], FACILITIES, per_stratum=0, seed=1)


class TestIndexIO:
    def test_round_trip_preserves_records(self, tmp_path):
        records = _records("Knife", True, 2) + _records("Knife", False, 1)
        path = tmp_path / "index.csv"
        subset.write_index(path, records)
        assert subset.read_index(path) == records

    def test_reads_clear_flag_as_boolean(self, tmp_path):
        path = tmp_path / "index.csv"
        path.write_text("image_id,item,clear,source\na,Knife,1,site\nb,Knife,0,video\n", encoding="utf-8")
        records = subset.read_index(path)
        assert [r.clear for r in records] == [True, False]
        assert records[1].source == "video"

    def test_rejects_index_missing_required_columns(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("image_id,item\na,Knife\n", encoding="utf-8")
        with pytest.raises(ValueError, match="clear"):
            subset.read_index(path)
