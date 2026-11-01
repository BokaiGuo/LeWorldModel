import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_render_replay import pixel_difference
from experiments.spectral_reachability.reacher_d2p_context import pack_rgb_observation


def test_one_channel_conflict_is_not_erased_by_a_mean_pixel_tolerance():
    a = np.full((224,224,3), 111, np.uint8)
    b = a.copy()
    b[184,39,1] = 112
    result = pixel_difference(pack_rgb_observation(a), pack_rgb_observation(b))
    assert result == {"changed_components": 1, "changed_coordinates": [[184,39,1]], "max_pixel_error": 1}
    assert pixel_difference(pack_rgb_observation(a), pack_rgb_observation(a))["changed_components"] == 0


def test_changed_image_shape_is_not_treated_as_small_noise():
    with pytest.raises(ValueError):
        pixel_difference(pack_rgb_observation(np.zeros((4,4,3),np.uint8)),
            pack_rgb_observation(np.zeros((5,4,3),np.uint8)))
