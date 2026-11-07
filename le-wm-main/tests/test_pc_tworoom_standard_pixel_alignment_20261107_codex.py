import numpy as np
import pytest

from experiments.pc_tworoom_standard_pixel_alignment_20261107_codex import centroid, compare


def test_compare_detects_single_channel_difference():
    first = np.zeros((3, 4, 3), dtype=np.uint8)
    second = first.copy()
    second[1, 2, 0] = 1
    result = compare(first, second)
    assert result["max_abs"] == 1
    assert result["changed_pixels"] == 1
    assert result["dataset_red_centroid"] is None
    assert result["render_red_centroid"] == [2.0, 1.0]


def test_compare_rejects_shape_and_dtype_mismatch():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        compare(rgb, np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        centroid(np.zeros((2, 2)))
