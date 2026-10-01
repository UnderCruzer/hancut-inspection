import pytest

from hancut.config import load_items
from hancut.data import yolo
from hancut.data.yolo import Box


class TestClassMapping:
    def test_indices_follow_the_class_ids_in_items_json(self):
        index_map = yolo.class_index_map(load_items())
        assert index_map == {cid: cid - 1 for cid in range(1, 13)}

    def test_names_are_in_class_id_order_not_paper_order(self):
        names = yolo.class_names(load_items())
        # 1 = Baton, 7 = Gun (논문 순서였다면 0 번이 Gun 이다)
        assert names[0] == "Baton" and names[6] == "Gun"
        assert len(names) == 12

    def test_rejects_missing_class_ids(self):
        with pytest.raises(ValueError, match="class_id 가 비어"):
            yolo.class_index_map([{"class_id": None}, {"class_id": 1}])


class TestToYolo:
    def test_normalizes_to_centre_and_size(self):
        box = yolo.to_yolo([100, 50, 200, 100], width=400, height=200, class_index=3)
        assert box == Box(3, 0.5, 0.5, 0.5, 0.5)

    def test_clips_boxes_that_leave_the_image(self):
        box = yolo.to_yolo([-50, -20, 100, 60], width=100, height=100, class_index=0)
        # 잘린 뒤 x 0..50, y 0..40
        assert box == Box(0, 0.25, 0.2, 0.5, 0.4)

    def test_drops_boxes_entirely_outside(self):
        assert yolo.to_yolo([150, 10, 20, 20], width=100, height=100, class_index=0) is None

    def test_drops_zero_area_boxes(self):
        assert yolo.to_yolo([10, 10, 0, 5], width=100, height=100, class_index=0) is None

    def test_line_format(self):
        assert Box(2, 0.5, 0.25, 0.1, 0.2).line() == "2 0.500000 0.250000 0.100000 0.200000"


def _coco(images, annotations):
    return {"images": images, "annotations": annotations,
            "categories": [{"id": i, "name": f"c{i}"} for i in range(1, 13)]}


class TestLabelsFromCoco:
    def test_keeps_images_without_boxes(self):
        data = _coco(
            [{"id": 0, "file_name": "a.png", "width": 100, "height": 100},
             {"id": 1, "file_name": "b.png", "width": 100, "height": 100}],
            [{"image_id": 0, "category_id": 7, "bbox": [10, 10, 20, 20]}],
        )
        labels, dropped = yolo.labels_from_coco(data, {c: c - 1 for c in range(1, 13)})
        assert labels["b"] == [] and len(labels["a"]) == 1
        assert labels["a"][0].class_index == 6
        assert dropped == 0

    def test_counts_dropped_boxes(self):
        data = _coco([{"id": 0, "file_name": "a.png", "width": 100, "height": 100}],
                     [{"image_id": 0, "category_id": 1, "bbox": [500, 500, 10, 10]}])
        labels, dropped = yolo.labels_from_coco(data, {c: c - 1 for c in range(1, 13)})
        assert labels["a"] == [] and dropped == 1

    def test_unknown_category_raises(self):
        data = _coco([{"id": 0, "file_name": "a.png", "width": 100, "height": 100}],
                     [{"image_id": 0, "category_id": 99, "bbox": [1, 1, 5, 5]}])
        with pytest.raises(ValueError, match="category_id 99"):
            yolo.labels_from_coco(data, {1: 0})


def _labels(n_common=900, n_rare=100):
    """흔한 품목 0 과 드문 품목 11 이 섞인 가짜 학습셋."""
    labels = {}
    for i in range(n_common):
        labels[f"c{i:04d}"] = [Box(0, 0.5, 0.5, 0.1, 0.1)]
    for i in range(n_rare):
        # 드문 품목은 흔한 품목과 같이 들어 있어도 드문 품목의 층으로 간다
        labels[f"r{i:04d}"] = [Box(0, 0.5, 0.5, 0.1, 0.1), Box(11, 0.3, 0.3, 0.1, 0.1)]
    return labels


class TestSplit:
    def test_splits_are_disjoint_and_cover_everything(self):
        labels = _labels()
        parts = yolo.split_train(labels)
        seen = [s for p in parts.values() for s in p]
        assert len(seen) == len(set(seen)) == len(labels)

    def test_sizes_follow_the_fractions(self):
        parts = yolo.split_train(_labels())
        assert len(parts["train"]) == 850
        assert len(parts["val"]) == 50
        assert len(parts["calib"]) == 100

    def test_rare_item_is_spread_across_splits(self):
        labels = _labels()
        parts = yolo.split_train(labels)
        rare = {name: yolo.class_counts(stems, labels)[11] for name, stems in parts.items()}
        # 층화하지 않으면 100장이 한쪽에 몰릴 수 있다. 비율대로 85 / 5 / 10
        assert rare == {"train": 85, "val": 5, "calib": 10}

    def test_same_seed_same_split_and_different_seed_differs(self):
        labels = _labels()
        assert yolo.split_train(labels, seed=0) == yolo.split_train(labels, seed=0)
        assert yolo.split_train(labels, seed=0) != yolo.split_train(labels, seed=1)

    def test_rejects_fractions_that_do_not_sum_to_one(self):
        with pytest.raises(ValueError, match="합이 1"):
            yolo.split_train(_labels(), {"train": 0.8, "calib": 0.1})


def test_repeat_counts_takes_the_largest_factor_per_image():
    classes = {"a": [6], "b": [10, 4], "c": [0], "d": []}
    assert yolo.repeat_counts(classes, {6: 3, 10: 3, 4: 2}) == {"a": 3, "b": 3, "c": 1, "d": 1}


def test_repeat_counts_rejects_zero():
    with pytest.raises(ValueError):
        yolo.repeat_counts({"a": [1]}, {1: 0})
