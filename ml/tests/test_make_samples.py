import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_samples.py"
spec = importlib.util.spec_from_file_location("make_samples", SCRIPT)
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)


def test_crop_region_pads_around_the_box_and_stays_inside():
    # 600x448 사진 가운데 60x40 박스 → 긴 변 60 에 양쪽 35% → 한 변 102
    x0, y0, x1, y1 = ms.crop_region((0.5, 0.5, 0.1, 40 / 448), 600, 448)
    assert (x1 - x0, y1 - y0) == (102, 102)
    # 모서리 박스는 사진 밖으로 나가지 않는다
    assert ms.crop_region((0.02, 0.02, 0.1, 0.1), 600, 448)[:2] == (0, 0)


def test_pick_gallery_skips_images_with_two_boxes_of_the_same_item():
    labels = {
        "a": [(0, (0.5, 0.5, 0.1, 0.1)), (0, (0.2, 0.2, 0.1, 0.1))],  # 경봉 둘 — 제외
        "b": [(0, (0.5, 0.5, 0.2, 0.2))],
        "c": [(6, (0.5, 0.5, 0.3, 0.3))],
    }
    picked = ms.pick_gallery(labels, 12)
    assert picked[0][0] == "b" and picked[6][0] == "c"
    assert 1 not in picked


def test_pick_examples_is_deterministic():
    stems = [f"x{i}" for i in range(20)]
    assert ms.pick_examples(stems, 3, seed=1) == ms.pick_examples(list(reversed(stems)), 3, seed=1)
    assert len(ms.pick_examples(stems[:2], 5)) == 2
