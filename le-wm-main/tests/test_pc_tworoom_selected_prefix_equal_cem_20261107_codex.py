import numpy as np
import pytest

from experiments.pc_tworoom_selected_prefix_equal_cem_20261107_codex import tail_cem


def test_fixed_prefix_budget_initial_and_best_sample():
    prefix = np.full((10, 2), .2, np.float32)
    tails = np.full((4, 15, 2), .3, np.float32)
    seen = []

    def score(chains):
        seen.append(chains.copy())
        return np.sum(chains[:, 10:] ** 2, axis=(1, 2))

    result = tail_cem(prefix, tails, 42, score)
    assert len(seen) == 4
    for chains in seen:
        np.testing.assert_array_equal(chains[:, :10], np.broadcast_to(prefix, (64, 10, 2)))
        assert np.abs(chains).max() <= 1
    np.testing.assert_array_equal(seen[0][:4, 10:], tails)
    assert result["cost"] == 0
    assert np.asarray(result["actions"]).shape == (25, 2)


def test_reject_invalid_and_nonfinite_scores():
    with pytest.raises(ValueError):
        tail_cem(np.zeros((9, 2)), np.zeros((4, 15, 2)), 0, lambda x: np.zeros(len(x)))
    with pytest.raises(ValueError):
        tail_cem(np.zeros((10, 2)), np.zeros((4, 15, 2)), 0, lambda x: np.full(len(x), np.nan))
