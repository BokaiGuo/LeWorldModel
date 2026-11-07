import numpy as np
import pytest

from experiments.pc_rgb_goal_color_mask_dev_20261107_codex import mask_goal


def test_masks_blue_without_uint8_underflow_or_mutation():
    image = np.full((3, 3, 3), 255, dtype=np.uint8)
    image[1, 1] = [65, 105, 225]
    image[1, 2] = [250, 100, 20]
    before = image.copy()
    masked, mask, bbox = mask_goal(image)
    assert mask.sum() == 1
    assert bbox == [1, 1, 1, 1]
    np.testing.assert_array_equal(masked[1, 1], [255, 255, 255])
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(masked[1, 2], image[1, 2])


def test_missing_mask_and_bad_dtype_fail_closed():
    with pytest.raises(ValueError):
        mask_goal(np.full((2, 2, 3), 255, dtype=np.uint8))
    with pytest.raises(ValueError):
        mask_goal(np.zeros((2, 2, 3), dtype=np.float32))
