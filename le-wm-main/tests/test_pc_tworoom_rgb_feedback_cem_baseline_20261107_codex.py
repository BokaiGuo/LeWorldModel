import numpy as np
import pytest

from experiments.pc_tworoom_rgb_feedback_cem_baseline_20261107_codex import candidates, old_or_hold


def test_candidate_bounds_and_old_tail():
    rng = np.random.default_rng(1)
    values = candidates(rng, np.zeros((5, 2)), np.ones((5, 2)), -1, 1, 4)
    assert values.shape == (4, 5, 2) and np.abs(values).max() <= 1
    old = np.ones((20, 2), np.float32)
    padded = old_or_hold(old)
    np.testing.assert_array_equal(padded[:20], old)
    np.testing.assert_array_equal(padded[20:], 0)
    with pytest.raises(ValueError):
        old_or_hold(np.full((20, 2), 2))
