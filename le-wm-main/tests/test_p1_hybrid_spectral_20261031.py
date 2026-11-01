import numpy as np
import pytest

from experiments.spectral_reachability import p1_hybrid_spectral_20261031 as hybrid
from experiments.spectral_reachability import codex_four_arm_spectral8_suffix_20261030 as old


def test_preserves_original_candidates_and_deduplicates():
    jac = np.zeros((192, 30))
    jac[:30] = np.diag(np.linspace(2., 1., 30))
    q = np.eye(192)
    goal = np.zeros(192)
    goal[7] = .1
    bank, _ = hybrid.goal_bank(jac, q, np.zeros(192), goal)
    original, _ = old.spectral_bank(jac, q)
    np.testing.assert_array_equal(bank[:4], original[:4])
    flat = bank.reshape(8, -1)
    assert all(np.max(np.abs(flat[i] - flat[j])) > 1e-7
               for i in range(8) for j in range(i))
    assert np.max(np.abs(bank)) <= 1
    changed = goal.copy()
    changed[7], changed[9] = 0, .1
    assert not np.array_equal(bank, hybrid.goal_bank(jac, q, np.zeros(192), changed)[0])


def test_zero_residual_and_invalid_inputs():
    jac = np.zeros((192, 30))
    jac[:30] = np.diag(np.linspace(2., 1., 30))
    bank, _ = hybrid.goal_bank(jac, np.eye(192), np.zeros(192), np.zeros(192))
    assert len(np.unique(bank.reshape(8, -1), axis=0)) == 8
    with pytest.raises(ValueError):
        hybrid.goal_bank(np.zeros_like(jac), np.eye(192), np.zeros(192), np.zeros(192))
    jac[0, 0] = np.nan
    with pytest.raises(ValueError):
        hybrid.goal_bank(jac, np.eye(192), np.zeros(192), np.zeros(192))


def test_damped_direction_matches_direct_solve_for_noncanonical_svd():
    # A negative diagonal makes numpy's SVD produce at least one right vector
    # whose deterministic spectral canonicalization requires a sign flip.
    jac = np.zeros((192, 30))
    diagonal = np.linspace(2., 1., 30)
    diagonal[7] *= -1
    jac[:30] = np.diag(diagonal)
    goal = np.zeros(192)
    goal[7], goal[9] = .1, .05
    bank, singular = hybrid.goal_bank(jac, np.eye(192), np.zeros(192), goal)
    expected = np.linalg.solve(jac.T @ jac + .01 * singular[0] ** 2 * np.eye(30),
                               jac.T @ goal)
    expected /= max(1., np.max(np.abs(expected)))
    np.testing.assert_allclose(bank[5].reshape(-1), .5 * expected, rtol=1e-6, atol=1e-7)
