import numpy as np
import pytest

from experiments.spectral_reachability.d2p_goal_mode_rank_20261102 import rank_goal_modes
from experiments.spectral_reachability import p1_dev304_mode_rank_saved_20261102 as saved


def test_rank_is_quadratic_improvement_not_update_magnitude():
    matrix = np.diag([10.0, 0.1])
    residual = np.array([1.0, 1.0])
    result = rank_goal_modes(matrix, residual, 0.01)
    assert np.argmax(np.abs(result["coefficients"])) == 1
    assert result["order"][0] == 0
    for i in range(2):
        action = result["coefficients"][i] * result["right_t"][i]
        cost = np.linalg.norm(residual - matrix @ action) ** 2 + 0.01 * np.linalg.norm(action) ** 2
        assert result["contributions"][i] == pytest.approx(np.linalg.norm(residual) ** 2 - cost)


def test_rejects_invalid_inputs_and_handles_zero_mode():
    for matrix, residual, damping in [
        (np.eye(2), [1, 2], 0), (np.eye(2), [1], 1),
        (np.eye(2), [1, np.nan], 1), (np.array([[np.inf]]), [1], 1),
    ]:
        with pytest.raises(ValueError):
            rank_goal_modes(matrix, residual, damping)
    result = rank_goal_modes(np.zeros((2, 2)), [1, 2], 0.1)
    assert np.all(result["contributions"] == 0)
    assert np.all(result["coefficients"] == 0)


def test_sealed_development_panel_is_bound_and_exploratory():
    result = saved.analyze()
    assert len(result["rows"]) == 2
    assert not result["independent_confirmation"] and not result["p1_admission"]
    for row in result["rows"]:
        assert row["goal_top2_contribution"] >= row["legacy_top2_contribution"]
