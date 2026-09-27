import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_index.py"


def _run(ann, out):
    return subprocess.run([sys.executable, str(SCRIPT), str(ann), str(out)],
                          capture_output=True, text=True)


def test_refuses_to_write_the_index_when_splits_overlap(tmp_path):
    from tests.test_pidray import _complete, _write, _image
    ann = _complete(tmp_path)
    _write(ann, "xray_train",
           [_image(0, "xray_00000.png"), _image(1, "xray_easy00000.png")],
           [{"image_id": 0, "category_id": 7}])
    out = tmp_path / "index.csv"
    result = _run(ann, out)
    assert result.returncode == 1
    assert "같은 사진이 여러 split 에 있다" in result.stdout
    assert not out.exists()


def test_refuses_to_write_when_counts_disagree_with_items_json(tmp_path):
    from tests.test_pidray import _complete
    ann = _complete(tmp_path)
    out = tmp_path / "index.csv"
    result = _run(ann, out)
    # 장난감 데이터라 items.json 실측값과 맞을 리가 없다 — 그걸 잡아내야 한다
    assert result.returncode == 1
    assert "어긋난다" in result.stdout
    assert not out.exists()


def test_wrong_argument_count_says_what_it_received(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "only-one"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "받은 인자 1개" in result.stdout
