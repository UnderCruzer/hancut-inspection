import random

import pytest

from hancut.data import tip

NAMES = ["Baton", "Knife", "Lighter"]


def test_class_weights_default_to_one_and_reject_bad_input():
    assert tip.class_weights(NAMES, {"Lighter": 3}) == {0: 1.0, 1: 1.0, 2: 3.0}
    with pytest.raises(ValueError, match="모르는 품목"):
        tip.class_weights(NAMES, {"Sword": 2})
    with pytest.raises(ValueError, match="0 보다"):
        tip.class_weights(NAMES, {"Knife": 0})


def test_pick_class_follows_weights():
    rng = random.Random(0)
    picks = [tip.pick_class(rng, {0: 1.0, 1: 3.0}) for _ in range(4000)]
    assert 0.72 < picks.count(1) / len(picks) < 0.78


def test_overlap_fraction_is_measured_from_the_first_rect():
    small, large = (0, 0, 10, 10), (0, 0, 100, 100)
    assert tip.overlap_fraction(small, large) == 1.0
    assert tip.overlap_fraction(large, small) == 0.01
    assert tip.overlap_fraction(small, (50, 50, 60, 60)) == 0.0


def test_is_allowed_blocks_a_small_object_inside_a_large_one():
    assert not tip.is_allowed((10, 10, 20, 20), [(0, 0, 100, 100)])
    assert tip.is_allowed((95, 95, 105, 105), [(0, 0, 100, 100)])
    assert tip.is_allowed((0, 0, 10, 10), [])


def test_choose_position_prefers_darker_candidates_inside_the_bag():
    # 왼쪽 절반만 어둡다. 오른쪽은 가방 밖처럼 밝아 후보에서 빠진다.
    brightness = lambda r: 0.3 if r[2] <= 50 else 0.95  # noqa: E731
    for seed in range(30):
        rect = tip.choose_position(random.Random(seed), (100, 100), (10, 10), [], brightness, candidates=64)
        assert rect is not None and rect[2] <= 50


def test_choose_position_keeps_the_rect_inside_and_gives_up_when_nothing_fits():
    rng = random.Random(0)
    assert tip.choose_position(rng, (10, 10), (20, 5), [], lambda r: 0.5) is None
    assert tip.choose_position(rng, (100, 100), (10, 10), [], lambda r: 0.99) is None
    assert tip.choose_position(rng, (10, 10), (10, 10), [(0, 0, 10, 10)], lambda r: 0.5) is None
    assert tip.choose_position(rng, (10, 10), (10, 10), [], lambda r: 0.5) == (0, 0, 10, 10)


def test_rasterize_fills_polygons_and_ignores_degenerate_ones():
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    mask = tip.rasterize([[2, 2, 7, 2, 7, 7, 2, 7], [1, 1, 2, 2]], 10, 10)
    assert mask.shape == (10, 10) and mask[4, 4] and not mask[0, 0]
    assert not np.any(tip.rasterize([[1, 1, 2, 2]], 10, 10))


def _image(np, value=200):
    return np.full((60, 60, 3), value, dtype=np.uint8)


SQUARE = [[10, 10, 40, 10, 40, 40, 10, 40]]


def test_cut_object_keeps_the_mask_soft_and_rejects_tiny_objects():
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    cut = tip.cut_object(_image(np), SQUARE)
    assert cut.transmission.shape[:2] == cut.alpha.shape
    assert cut.alpha.max() > 0.99 and cut.alpha[0, 0] < 0.01
    assert ((cut.alpha > 0.01) & (cut.alpha < 0.99)).any()
    assert tip.cut_object(_image(np), [[1, 1, 4, 1, 4, 4, 1, 4]]) is None
    assert tip.cut_object(_image(np), []) is None


def test_transform_scales_and_keeps_shapes_aligned():
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    cut = tip.Cutout(np.full((20, 40, 3), 0.5, dtype=np.float32), np.ones((20, 40), dtype=np.float32))
    out = tip.transform(cut, random.Random(0), scale=0.5)
    assert sorted(out.alpha.shape) == [10, 20]
    assert out.transmission.shape[:2] == out.alpha.shape
    assert abs(float(out.transmission.mean()) - 0.5) < 0.01


def test_project_multiplies_inside_the_mask_and_leaves_the_rest_untouched():
    np = pytest.importorskip("numpy")
    bg = _image(np, 200)
    cut = tip.Cutout(np.full((4, 4, 3), 0.5, dtype=np.float32), np.array([[1, 1, 0, 0]] * 4, dtype=np.float32))
    out = tip.project(bg, cut, 10, 20)
    assert out[20, 10, 0] == 100  # 200 × 0.5
    assert out[20, 12, 0] == 200  # 마스크 밖
    assert out[0, 0, 0] == 200 and bg[20, 10, 0] == 200  # 원본은 그대로
    assert tip.project(bg, cut, 10, 20, density=0.5)[20, 10, 0] > out[20, 10, 0]  # 얇을수록 덜 어둡다


def test_project_never_brightens_the_background():
    np = pytest.importorskip("numpy")
    bg = np.array(np.random.default_rng(0).integers(0, 256, (30, 30, 3)), dtype=np.uint8)
    cut = tip.Cutout(np.array(np.random.default_rng(1).random((30, 30, 3)), dtype=np.float32),
                     np.array(np.random.default_rng(2).random((30, 30)), dtype=np.float32))
    assert (tip.project(bg, cut, 0, 0).astype(int) <= bg.astype(int)).all()


def test_object_box_excludes_the_feathered_edge():
    np = pytest.importorskip("numpy")
    alpha = np.zeros((20, 20), dtype=np.float32)
    alpha[5:15, 4:16] = 1
    alpha[4, 4:16] = 0.2
    cut = tip.Cutout(np.ones((20, 20, 3), dtype=np.float32), alpha)
    assert tip.object_box(cut, 100, 200) == (104, 205, 116, 215)
    assert tip.object_box(tip.Cutout(cut.transmission, np.zeros((20, 20), dtype=np.float32)), 0, 0) is None


def test_compose_labels_every_object_it_draws_and_skips_what_does_not_fit():
    np = pytest.importorskip("numpy")
    bg = np.full((100, 100, 3), 128, dtype=np.uint8)
    cut = tip.Cutout(np.full((20, 20, 3), 0.5, dtype=np.float32), np.ones((20, 20), dtype=np.float32))
    image, pasted = tip.compose(bg, [], [(2, cut)], random.Random(0))
    assert [c for c, _ in pasted] == [2]
    x0, y0, x1, y1 = pasted[0][1]
    assert (x1 - x0, y1 - y0) == (20, 20)
    assert (image[y0:y1, x0:x1, 0] < 128).all()
    assert (bg == 128).all()

    huge = tip.Cutout(np.ones((200, 200, 3), dtype=np.float32), np.ones((200, 200), dtype=np.float32))
    same, none = tip.compose(bg, [], [(1, huge)], random.Random(0))
    assert none == [] and (same == bg).all()


def test_compose_is_reproducible_with_the_same_seed():
    np = pytest.importorskip("numpy")
    bg = np.array(np.random.default_rng(0).integers(60, 180, (200, 200, 3)), dtype=np.uint8)
    cut = tip.Cutout(np.full((30, 30, 3), 0.4, dtype=np.float32), np.ones((30, 30), dtype=np.float32))
    a, pa = tip.compose(bg, [], [(0, cut), (1, cut)], random.Random(7))
    b, pb = tip.compose(bg, [], [(0, cut), (1, cut)], random.Random(7))
    assert pa == pb and (a == b).all()
