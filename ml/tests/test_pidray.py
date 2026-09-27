import json

import pytest

from hancut.data import pidray
from hancut.data.pidray import Row

CATS = [{"id": 1, "name": "Baton"}, {"id": 7, "name": "Gun"}, {"id": 11, "name": "Knife"}]


def _write(dirpath, stem, images, annotations, categories=CATS):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / f"{stem}.json").write_text(json.dumps({
        "info": {}, "license": {}, "categories": categories,
        "images": images, "annotations": annotations,
    }), encoding="utf-8")


def _image(i, name):
    return {"height": 448, "width": 620, "id": i, "file_name": name}


def _complete(tmp_path):
    """네 split 을 모두 갖춘 최소 데이터셋."""
    ann = tmp_path / "annotations"
    _write(ann, "xray_train",
           [_image(0, "xray_00000.png"), _image(1, "xray_00001.png")],
           [{"image_id": 0, "category_id": 7}])
    _write(ann, "xray_test_easy", [_image(0, "xray_easy00000.png")],
           [{"image_id": 0, "category_id": 1}])
    _write(ann, "xray_test_hard", [_image(0, "xray_hard00000.png")],
           [{"image_id": 0, "category_id": 1}, {"image_id": 0, "category_id": 11}])
    _write(ann, "xray_test_hidden", [_image(0, "xray_hidden00000.png")],
           [{"image_id": 0, "category_id": 11}])
    return ann


class TestRowShape:
    def test_every_image_gets_one_row_per_item(self, tmp_path):
        rows = pidray.build_index(_complete(tmp_path))
        # 이미지 5장 x 품목 3종
        assert len(rows) == 15

    def test_positive_only_for_the_items_actually_present(self, tmp_path):
        rows = pidray.build_index(_complete(tmp_path))
        hard = {r.item: r.y_true for r in rows if r.image_id == "xray_hard00000"}
        assert hard == {"Baton": 1, "Gun": 0, "Knife": 1}

    def test_an_image_with_no_annotation_is_negative_for_every_item(self, tmp_path):
        rows = pidray.build_index(_complete(tmp_path))
        empty = [r for r in rows if r.image_id == "xray_00001"]
        assert len(empty) == 3
        assert all(r.y_true == 0 for r in empty)

    def test_difficulty_is_empty_for_train_and_set_for_test(self, tmp_path):
        rows = pidray.build_index(_complete(tmp_path))
        by_id = {r.image_id: (r.split, r.difficulty) for r in rows}
        assert by_id["xray_00000"] == ("train", "")
        assert by_id["xray_hidden00000"] == ("test", "hidden")


class TestImageIdCollision:
    def test_numeric_ids_repeat_across_files_so_the_stem_is_the_key(self, tmp_path):
        # train 의 0 번과 test_easy 의 0 번은 다른 사진이다. 숫자 id 를 키로 쓰면 뭉개진다.
        rows = pidray.build_index(_complete(tmp_path))
        ids = {r.image_id for r in rows}
        assert "xray_00000" in ids and "xray_easy00000" in ids
        assert len(ids) == 5

    def test_leakage_is_empty_when_splits_are_disjoint(self, tmp_path):
        assert pidray.leakage(pidray.build_index(_complete(tmp_path))) == {}

    def test_leakage_reports_a_shared_image(self, tmp_path):
        ann = _complete(tmp_path)
        # 같은 파일명을 train 에도 넣어 누수를 만든다
        _write(ann, "xray_train",
               [_image(0, "xray_00000.png"), _image(1, "xray_easy00000.png")],
               [{"image_id": 0, "category_id": 7}])
        found = pidray.leakage(pidray.build_index(ann))
        assert found == {"xray_easy00000": {"train", "test/easy"}}


class TestStrictness:
    def test_missing_split_file_raises(self, tmp_path):
        ann = tmp_path / "annotations"
        _write(ann, "xray_train", [_image(0, "a.png")], [])
        with pytest.raises(FileNotFoundError, match="xray_test_easy"):
            pidray.annotation_files(ann)

    def test_unknown_annotation_file_raises(self, tmp_path):
        ann = _complete(tmp_path)
        _write(ann, "xray_test_impossible", [_image(0, "a.png")], [])
        with pytest.raises(ValueError, match="모르는 어노테이션 파일"):
            pidray.annotation_files(ann)

    def test_macos_junk_files_are_ignored(self, tmp_path):
        ann = _complete(tmp_path)
        (ann / "._xray_train.json").write_bytes(b"junk")
        assert [p.stem for p in pidray.annotation_files(ann)] == [
            "xray_test_easy", "xray_test_hard", "xray_test_hidden", "xray_train"]

    def test_category_id_outside_categories_raises(self, tmp_path):
        ann = _complete(tmp_path)
        _write(ann, "xray_test_easy", [_image(0, "xray_easy00000.png")],
               [{"image_id": 0, "category_id": 99}])
        with pytest.raises(ValueError, match="category_id 99"):
            list(pidray.rows_from_file(ann / "xray_test_easy.json"))


class TestIndexFile:
    def test_round_trip(self, tmp_path):
        rows = pidray.build_index(_complete(tmp_path))
        path = tmp_path / "index.csv"
        assert pidray.write_index(rows, path) == len(rows)
        assert pidray.read_index(path) == rows

    def test_read_rejects_a_different_header(self, tmp_path):
        path = tmp_path / "index.csv"
        path.write_text("image_id,item\na,Gun\n", encoding="utf-8")
        with pytest.raises(ValueError, match="열이 다르다"):
            pidray.read_index(path)


class TestSummary:
    def test_counts_positives_per_item_and_split(self, tmp_path):
        summary = pidray.summarize(pidray.build_index(_complete(tmp_path)))
        baton = next(r for r in summary if r["item"] == "Baton")
        assert baton["test_easy"] == 1 and baton["test_hard"] == 1
        assert baton["train"] == 0
        assert baton["test_total"] == 2

    def test_table_renders_the_item_column(self, tmp_path):
        table = pidray.format_table(pidray.summarize(pidray.build_index(_complete(tmp_path))))
        assert "| 품목 |" in table and "| Baton |" in table
