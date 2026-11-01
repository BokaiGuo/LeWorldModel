import numpy as np
import pytest

from experiments.spectral_reachability.e3_cp10_goal_dls_20261031 import goal_bank


def test_dls_tracks_goal_and_stays_bounded():
    jac = np.zeros((192, 30))
    jac[0, 0], jac[1, 1] = 2., 1.
    goal = np.zeros(192)
    goal[1] = 1.
    bank, singular = goal_bank(jac, np.eye(192), goal, np.zeros(192))
    assert bank.shape == (8, 3, 5, 2)
    assert np.max(np.abs(bank)) <= 1
    assert bank[-1].reshape(-1)[1] > 0
    assert bank[-1].reshape(-1)[0] == 0
    assert singular[0] == 2.


def test_zero_and_invalid_inputs():
    jac = np.zeros((192, 30))
    bank, _ = goal_bank(jac, np.eye(192), np.ones(192), np.zeros(192))
    assert np.count_nonzero(bank) == 0
    jac[0, 0] = np.nan
    with pytest.raises(ValueError):
        goal_bank(jac, np.eye(192), np.zeros(192), np.zeros(192))
