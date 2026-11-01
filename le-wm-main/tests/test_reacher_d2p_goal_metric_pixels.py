"""Raw RGB numeric fixtures only; no neural inference or physical success."""
import numpy as np
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_goal_metric_pixels import pixel_cost


def test_pixel_cost_uses_all_channels_and_correct_uint8_normalization():
    dark=np.zeros((224,224,3),np.uint8)
    light=np.full_like(dark,255)
    assert pixel_cost(dark,dark)==0.
    assert pixel_cost(light,dark)==1.
    light[:,:,1:]=0
    assert pixel_cost(light,dark)==pytest.approx(1/3)


@pytest.mark.parametrize("bad",[np.zeros((224,224,3),np.float32),np.zeros((112,112,3),np.uint8)])
def test_scaled_or_partial_pixels_cannot_silently_replace_raw_rgb(bad):
    with pytest.raises(ValueError): pixel_cost(bad,np.zeros((224,224,3),np.uint8))
