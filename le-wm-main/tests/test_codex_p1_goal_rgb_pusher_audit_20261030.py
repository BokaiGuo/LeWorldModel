import numpy as np
import pytest

from experiments.spectral_reachability.codex_p1_goal_rgb_pusher_audit_20261030 import blue_summary


def test_pixel_centroid_and_empty():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    assert blue_summary(image)["centroid_xy"] is None
    image[2, 3] = [65, 105, 225]
    image[4, 5] = [65, 105, 225]
    result = blue_summary(image)
    assert result["count"] == 2
    assert result["centroid_xy"] == [4., 3.]
    assert result["bounds_xyxy"] == [3, 2, 5, 4]


def test_invalid_image_rejected():
    with pytest.raises(Exception):
        blue_summary(np.zeros((224, 224, 3), dtype=np.float32))
