import pytest

from hancut.eval.tiling import restore_box, tile_windows


@pytest.mark.parametrize("width,height", [(100, 80), (103, 79), (1, 1), (2, 7)])
def test_windows_cover_every_pixel_without_out_of_bounds(width, height):
    windows = tile_windows(width, height)
    assert len(windows) == len(set(windows))
    assert all(0 <= x < r <= width and 0 <= y < b <= height for x, y, r, b in windows)
    for y in range(height):
        for x in range(width):
            assert any(l <= x < r and t <= y < b for l, t, r, b in windows)


def test_corner_translation_and_clipping():
    assert restore_box((5, 10, 15, 20), (40, 32, 100, 80)) == (45, 42, 55, 52)
    assert restore_box((-1, -2, 90, 70), (40, 32, 100, 80)) == (40, 32, 100, 80)
    assert tile_windows(100, 80, 1.0) == [(0, 0, 100, 80)]


@pytest.mark.parametrize("args", [(0, 80, .6), (80, 0, .6), (80, 80, .49), (80, 80, 1.1)])
def test_invalid_windows_fail(args):
    with pytest.raises(ValueError):
        tile_windows(*args)
