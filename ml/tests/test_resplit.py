import pytest

from hancut.eval import resplit


def _rows():
    """easy 200장(총기) · hidden 100장(총기) · easy 20장(경봉, 드문 품목), 품목 3종."""
    rows = []
    specs = ([("easy", i, "Gun") for i in range(200)]
             + [("hidden", i, "Gun") for i in range(100)]
             + [("easy", 200 + i, "Baton") for i in range(20)])
    for difficulty, i, present in specs:
        image_id = f"xray_{difficulty}{i:05d}"
        for item in ("Baton", "Gun", "Knife"):
            rows.append({"image_id": image_id, "item": item, "y_true": int(item == present), "score": 0.5})
    return rows


def test_difficulty_comes_from_the_image_id():
    assert resplit.difficulty_of("xray_hidden00012") == "hidden"
    assert resplit.difficulty_of("xray_easy00000") == "easy"
    assert resplit.difficulty_of("xray_00000") == ""


def test_every_image_goes_whole_to_one_side():
    rows = _rows()
    calib_ids, eval_ids = resplit.split_images(resplit.image_table(rows))
    assert not calib_ids & eval_ids
    calib, rest = resplit.partition(rows, calib_ids)
    # 한 사진의 3행이 갈라지지 않는다
    assert {r["image_id"] for r in calib}.isdisjoint({r["image_id"] for r in rest})
    assert len(calib) + len(rest) == len(rows)
    assert len(calib) % 3 == 0


def test_difficulty_and_rare_item_are_split_evenly():
    images = resplit.image_table(_rows())
    calib_ids, _ = resplit.split_images(images)
    in_calib = [images[i] for i in calib_ids]
    assert sum(d == "hidden" for d, _ in in_calib) == 50
    assert sum("Baton" in items for _, items in in_calib) == 10


def test_same_seed_same_split():
    images = resplit.image_table(_rows())
    assert resplit.split_images(images, seed=3) == resplit.split_images(images, seed=3)
    assert resplit.split_images(images, seed=3) != resplit.split_images(images, seed=4)


def test_rejects_degenerate_fraction():
    with pytest.raises(ValueError):
        resplit.split_images(resplit.image_table(_rows()), fraction=1.0)
