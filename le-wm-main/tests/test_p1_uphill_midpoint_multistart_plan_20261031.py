import numpy as np
import pytest

from experiments.spectral_reachability import p1_uphill_midpoint_multistart_plan_20261031 as plan


def test_starts_are_bounded_deterministic_and_include_original():
    original = np.full((5, 2), 0.2, dtype="<f4")
    starts = plan.initial_actions(3, original)
    assert starts.shape == (4, 5, 2)
    assert np.array_equal(starts[0], original)
    assert np.count_nonzero(starts[1]) == 0
    assert np.max(np.abs(starts)) <= 1
    assert np.array_equal(starts, plan.initial_actions(3, original))
    assert not np.array_equal(starts[2:], plan.initial_actions(4, original)[2:])
    with pytest.raises(ValueError):
        plan.initial_actions(8, original)
