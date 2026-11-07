import numpy as np
import pytest

from experiments.pc_goal_rgb_canonicalization import goal_shift, translate


def scene(left):
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[55:197, left:left+142] = [222, 161, 100]
    return image


def test_nominal_and_shifted_goal_determine_translation():
    assert (goal_shift(scene(41))["dx"], goal_shift(scene(41))["dy"]) == (0, 0)
    assert (goal_shift(scene(27))["dx"], goal_shift(scene(27))["dy"]) == (14, 0)


def test_translation_recovers_shared_valid_region():
    corrected, valid = translate(scene(27), 14, 0)
    np.testing.assert_array_equal(corrected[valid], scene(41)[valid])
    assert valid.sum() == 224 * 210


def test_missing_boundary_rejected():
    with pytest.raises(ValueError):
        goal_shift(np.zeros((224, 224, 3), dtype=np.uint8))


def test_displacement_outside_contract_rejected():
    with pytest.raises(ValueError):
        goal_shift(scene(15))
