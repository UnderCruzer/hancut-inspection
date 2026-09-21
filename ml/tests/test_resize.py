import pytest
from PIL import Image

from hancut.data import resize


def _make_image(path, size=(1920, 1080), color=(200, 60, 40)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG", quality=95)
    return path


class TestScale:
    def test_long_side_becomes_max_side(self):
        assert resize.scale_for((1920, 1080), 640) == pytest.approx(640 / 1920)

    def test_small_images_are_not_upscaled(self):
        assert resize.scale_for((320, 240), 640) == 1.0

    def test_points_scale_with_the_image(self):
        assert resize.scale_points([100, 200, 300, 400], 0.5) == [50.0, 100.0, 150.0, 200.0]


class TestResizeImage:
    def test_long_side_is_capped_and_aspect_kept(self, tmp_path):
        src = _make_image(tmp_path / "src" / "a.jpg", (1920, 1080))
        result = resize.resize_image(src, tmp_path / "out" / "a.jpg")
        with Image.open(result.dst) as out:
            assert max(out.size) == 640
            assert out.size == (640, 360)

    def test_smaller_image_keeps_its_size(self, tmp_path):
        src = _make_image(tmp_path / "src" / "small.jpg", (320, 240))
        result = resize.resize_image(src, tmp_path / "out" / "small.jpg")
        with Image.open(result.dst) as out:
            assert out.size == (320, 240)
        assert result.scale == 1.0

    def test_output_is_much_smaller_than_the_original(self, tmp_path):
        src = _make_image(tmp_path / "src" / "b.jpg", (4000, 3000))
        result = resize.resize_image(src, tmp_path / "out" / "b.jpg")
        assert result.dst_bytes < result.src_bytes
        assert result.saved_bytes > 0

    def test_existing_output_is_skipped_unless_overwrite(self, tmp_path):
        src = _make_image(tmp_path / "src" / "c.jpg")
        dst = tmp_path / "out" / "c.jpg"
        first = resize.resize_image(src, dst)
        mtime = dst.stat().st_mtime_ns

        skipped = resize.resize_image(src, dst)
        assert skipped.skipped is True
        assert dst.stat().st_mtime_ns == mtime
        assert skipped.scale == first.scale

        forced = resize.resize_image(src, dst, overwrite=True)
        assert forced.skipped is False

    def test_exif_rotated_image_is_uprighted(self, tmp_path):
        src = tmp_path / "src" / "rot.jpg"
        src.parent.mkdir(parents=True)
        image = Image.new("RGB", (1000, 500), (10, 10, 10))
        exif = image.getexif()
        exif[274] = 6  # Orientation: 90도 회전 필요
        image.save(src, format="JPEG", exif=exif)

        result = resize.resize_image(src, tmp_path / "out" / "rot.jpg")
        with Image.open(result.dst) as out:
            assert out.height > out.width  # 회전이 실제로 적용됐다

    def test_png_input_is_written_as_jpeg(self, tmp_path):
        src = tmp_path / "src" / "d.png"
        src.parent.mkdir(parents=True)
        Image.new("RGBA", (800, 800), (0, 0, 0, 0)).save(src)
        result = resize.resize_image(src, tmp_path / "out" / "d.jpg")
        with Image.open(result.dst) as out:
            assert out.format == "JPEG"
            assert out.mode == "RGB"


class TestSummarize:
    def test_reports_before_and_after_totals(self, tmp_path):
        pairs = []
        for i in range(3):
            src = _make_image(tmp_path / "src" / f"{i}.jpg", (2000, 1500))
            pairs.append((src, tmp_path / "out" / f"{i}.jpg"))

        summary = resize.summarize(resize.resize_many(pairs))
        assert summary["count"] == 3
        assert summary["skipped"] == 0
        assert summary["dst_mb"] < summary["src_mb"]
        assert 0 < summary["ratio"] < 1
        assert summary["mean_dst_kb"] > 0

    def test_empty_input_does_not_divide_by_zero(self):
        assert resize.summarize([]) == {
            "count": 0, "skipped": 0, "src_mb": 0.0, "dst_mb": 0.0, "ratio": 0.0, "mean_dst_kb": 0.0,
        }
