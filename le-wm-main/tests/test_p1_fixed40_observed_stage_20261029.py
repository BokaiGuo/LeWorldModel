import numpy as np
import pytest

from experiments.spectral_reachability.p1_fixed40_observed_stage_20261029 import stage_cost


def test_stage_cost_is_same_goal_and_signed():
    goal = np.zeros(192)
    result = stage_cost(np.ones(192), np.full(192, 2), goal)
    assert result == dict(start_Q=1.0, end_Q=4.0, delta_Q=3.0)


def test_stage_cost_rejects_bad_input():
    with pytest.raises(ValueError):
        stage_cost(np.zeros(191), np.zeros(192), np.zeros(192))
    with pytest.raises(ValueError):
        stage_cost(np.full(192, np.nan), np.zeros(192), np.zeros(192))
