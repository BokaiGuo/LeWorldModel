import numpy as np
import pytest

from experiments.pc_shared_rgb_equal_tail_replan_20261107_codex import make_batch


def test_fixed_prefix_padding_incumbent_and_bounds():
    prefix = np.full((5, 2), .2)
    mean, std, incumbent = np.zeros((13, 2)), np.ones((13, 2)), np.full((13, 2), .3)
    batch = make_batch(prefix, mean, std, incumbent, np.full((16, 13, 2), 100))
    np.testing.assert_allclose(batch[:, :5], np.broadcast_to(prefix, (16, 5, 2)))
    np.testing.assert_allclose(batch[0, 5:18], incumbent)
    np.testing.assert_array_equal(batch[:, 18:], 0)
    assert np.max(np.abs(batch)) <= 1


def test_shape_and_nonfinite_rejected():
    with pytest.raises(ValueError):
        make_batch(np.zeros((4, 2)), np.zeros((2, 2)), np.ones((2, 2)),
                   np.zeros((2, 2)), np.zeros((16, 2, 2)))
