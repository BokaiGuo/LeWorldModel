import numpy as np
import pytest
from experiments.spectral_reachability.analyze_pusht_observed_suffix_physical_fd import royal_blue_pixels, cosine


def test_default_agent_color_absence_and_presence():
    image = np.full((224, 224, 3), 255, np.uint8)
    assert royal_blue_pixels(image) == 0
    image[10:12, 20:22] = [65, 105, 225]
    assert royal_blue_pixels(image) == 4


def test_color_is_not_general_detector():
    image = np.full((224, 224, 3), 255, np.uint8)
    image[10, 10] = [66, 105, 225]
    assert royal_blue_pixels(image) == 0


def test_zero_derivative_is_not_fake_cosine_zero():
    assert cosine([1., 2.], [0., 0.]) is None
    assert np.isclose(cosine([1., 2.], [-1., -2.]), -1)


def test_bad_frame_rejected():
    with pytest.raises(ValueError): royal_blue_pixels(np.zeros((224, 224, 3)))
