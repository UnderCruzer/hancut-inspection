import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("run_mixup_experiment", Path(__file__).resolve().parents[1] / "scripts/run_mixup_experiment.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


def config():
    return json.loads(ex.DEFAULT_CONFIG.read_text())


def test_main_arms_only_differ_in_mixup_and_output_name(tmp_path):
    c = config()
    a, b = [ex.build_train_args(c, tmp_path, arm) for arm in ("control", "mixup")]
    assert {k for k in a if a[k] != b[k]} == {"mixup", "name"}
    assert a["mixup"] == 0 and b["mixup"] == .3
    assert a["epochs"] == b["epochs"] == 20 and a["patience"] == 0
    assert Path(a["project"]).is_absolute()
    assert "save_dir" not in a and not a["resume"] and not a["exist_ok"]


@pytest.mark.parametrize("key,value", [("save_dir", "/old/run"), ("project", "/old/run"),
                                      ("resume", True), ("exist_ok", True)])
def test_inherited_run_settings_rejected(tmp_path, key, value):
    c = config()
    c["train"][key] = value
    with pytest.raises(ValueError):
        ex.build_train_args(c, tmp_path, "control")


def test_smoke_really_exercises_mixup_for_one_epoch(tmp_path):
    a = ex.build_train_args(config(), tmp_path, "smoke")
    assert a["epochs"] == 1 and a["close_mosaic"] == 0 and a["mixup"] == .3


def test_leakage_detected_even_with_different_symlink_names(tmp_path):
    source = tmp_path / "original.png"
    source.touch()
    alias = tmp_path / "renamed.png"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="occurs twice"):
        ex.check_disjoint({"train": [source], "val": [alias]})
    other = tmp_path / "other.png"
    other.touch()
    ex.check_disjoint({"train": [source], "val": [other]})


def test_ultralytics_mixup_preserves_both_classes_and_boxes():
    np = pytest.importorskip("numpy")
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import MixUp
    from ultralytics.utils.instance import Instances
    def labels(value, cls, box):
        return dict(img=np.full((4, 4, 3), value, dtype=np.uint8), cls=np.array([[cls]], dtype=np.float32),
                    instances=Instances(np.array([box], dtype=np.float32),
                                        bbox_format="xyxy", normalized=False))
    a = labels(20, 0, [0, 0, 2, 2]); b = labels(220, 1, [2, 2, 4, 4])
    a["mix_labels"] = [b]
    mix = MixUp(dataset=None, p=.3)
    result = mix.apply_instances(mix.apply_image(a, {"r": .5}), {"r": .5})
    assert (result["img"] == 120).all()
    np.testing.assert_array_equal(result["cls"].ravel(), [0, 1])
    np.testing.assert_array_equal(result["instances"].bboxes, [[0, 0, 2, 2], [2, 2, 4, 4]])
