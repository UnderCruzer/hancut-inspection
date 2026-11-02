import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_roi_refiner.py'
spec = importlib.util.spec_from_file_location('run_roi_refiner',SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_help_does_not_require_gpu_libraries():
    result = subprocess.run([sys.executable,str(SCRIPT),'--help'],capture_output=True,text=True)
    assert result.returncode == 0 and '--smoke' in result.stdout


def test_refinement_preserves_non_targets_and_zero_baseline():
    np = pytest.importorskip('numpy')
    pytest.importorskip('torchvision')
    full = np.array([[0,0,10,10,.9,0],[20,20,30,30,.8,1]],dtype=np.float32)
    probabilities = {0:[.05,.95,0.],1:[.9,.1,0.]}
    config = {'nms_iou':.7}
    assert np.array_equal(runner.refined(full,probabilities,[1],{1:0},config),full)
    result = runner.refined(full,probabilities,[1],{1:1},config)
    assert np.array_equal(result[result[:,5]==0],full[full[:,5]==0])
    assert sorted(result[result[:,5]==1,4]) == pytest.approx([.08,.855])


def test_patch_keeps_aspect_ratio_and_context():
    np = pytest.importorskip('numpy')
    pytest.importorskip('cv2')
    image = np.zeros((50,100,3),dtype=np.uint8)
    image[:,:] = (10,20,30)
    result = runner.patch(image,(0,0,100,50),{'crop_size':100,'context':0})
    assert result.shape == (100,100,3)
    assert (result[:25]==114).all()
    assert (result[25:75]==[10,20,30]).all()
