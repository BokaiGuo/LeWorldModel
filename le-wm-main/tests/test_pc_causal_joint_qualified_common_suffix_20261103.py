import numpy as np
import pytest

from experiments.pc_causal_joint_qualified_common_suffix_20261103 import build_suffixes


def test_common_suffixes_are_fixed_bounded_and_bound_to_plans():
    plans = np.zeros((13, 10, 15, 2), np.float32)
    plans[11, 3, 5:] = .2
    plans[8, 1, 5:] = -.1
    first = build_suffixes(plans, [3, 11], [1, 8])
    assert first.shape == (5, 10, 2)
    np.testing.assert_array_equal(first, build_suffixes(plans, [3, 11], [1, 8]))
    np.testing.assert_allclose(first[1], .2)
    np.testing.assert_allclose(first[2], -.1)
    assert np.max(np.abs(first)) <= 1
    with pytest.raises(ValueError):
        build_suffixes(plans[:, :, :-1], [3, 11], [1, 8])
