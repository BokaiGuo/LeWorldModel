import numpy as np
import pytest

from experiments.spectral_reachability import p1_goal_conditioned_spectral_20261031 as method


def fixture():
    jac = np.zeros((192, 30))
    jac[:30] = np.diag(np.linspace(2., 1., 30))
    return jac, np.eye(192), np.zeros(192)


def test_goal_changes_candidate_and_matches_damped_solution():
    jac, projection, nominal = fixture()
    goal = np.zeros(192)
    goal[7] = .1
    bank, singular = method.goal_bank(jac, projection, nominal, goal)
    expected = np.linalg.solve(jac.T @ jac + .01 * singular[0] ** 2 * np.eye(30), jac.T @ goal)
    np.testing.assert_allclose(bank[-1].reshape(-1), expected, rtol=1e-6, atol=1e-7)
    assert bank.shape == (8, 3, 5, 2)
    assert np.max(np.abs(bank)) <= 1
    other = goal.copy()
    other[7], other[9] = 0, .1
    assert not np.array_equal(method.goal_bank(jac, projection, nominal, other)[0], bank)


def test_rejects_nonfinite_and_zero_jacobian():
    jac, projection, nominal = fixture()
    with pytest.raises(ValueError):
        method.goal_bank(np.zeros_like(jac), projection, nominal, nominal)
    nominal[0] = np.nan
    with pytest.raises(ValueError):
        method.goal_bank(jac, projection, nominal, nominal)
