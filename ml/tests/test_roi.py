import pytest
from hancut.eval.roi import crop_window, covered_fraction, proposal_scores, choose_alpha, assert_disjoint


def test_context_is_clipped_without_losing_object():
    assert crop_window((5,10,25,30),30,40,.5) == (0,0,30,40)
    with pytest.raises(ValueError):
        crop_window((5,4,3,8),30,40)


def test_large_background_window_cannot_hide_a_tiny_object():
    assert covered_fraction((0,0,100,100),(40,40,45,45)) == 1
    assert covered_fraction((0,0,10,10),(40,40,45,45)) == 0


def test_zero_blend_preserves_existing_class_without_inventing_targets():
    assert proposal_scores(1,.9,[.8,.1,.1],[0,1],{0:0,1:0}) == {0:0,1:.9}
    scores = proposal_scores(1,.9,[.8,.1,.1],[0,1],{0:.5,1:.5})
    assert scores[0] == pytest.approx(.36)
    assert scores[1] == pytest.approx(.495)


def test_calibration_rejects_ap_gain_with_precision_or_recall_regression():
    base = dict(ap50=.6,ap50_95=.4,precision=.8,recall=.5)
    gate = dict(ap50_gain=.005,ap50_95_drop=.005,precision_drop=.02,recall_drop=.01)
    bad = {**base,'ap50':.7,'precision':.5}
    good = {**base,'ap50':.62}
    assert choose_alpha(base,[(.25,bad)],gate) == 0
    assert choose_alpha(base,[(.25,bad),(.5,good)],gate) == .5
    assert choose_alpha(base,[(.5,{**good,'recall':.4})],gate) == 0


def test_images_never_cross_partitions():
    assert_disjoint({'train':['a'],'val':['b'],'test_hidden':['c']})
    with pytest.raises(ValueError):
        assert_disjoint({'train':['a'],'val':['a']})
