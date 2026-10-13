import numpy as np

from experiments.empirical_spectral_audit.run_dynamics_subspace_probe import (
    _cross_covariance_basis,
    _residualize_target,
)


def test_cross_covariance_basis_is_orthonormal():
    rng = np.random.default_rng(7)
    z = rng.normal(size=(100, 8))
    y = z[:, :3] @ rng.normal(size=(3, 5)) + 0.01 * rng.normal(size=(100, 5))
    basis, singular_values = _cross_covariance_basis(z, y)
    assert basis.shape == (8, 5)
    assert np.allclose(basis.T @ basis, np.eye(5), atol=1e-6)
    assert np.all(np.diff(singular_values) <= 1e-8)


def test_action_residualization_removes_linear_action_signal():
    rng = np.random.default_rng(8)
    action = rng.normal(size=(120, 3))
    coef = rng.normal(size=(3, 4))
    dynamics = action @ coef + 0.01 * rng.normal(size=(120, 4))
    train_residual, test_residual = _residualize_target(
        action[:90], action[90:], dynamics[:90], dynamics[90:], ridge=1e-3
    )
    assert np.mean(np.square(train_residual)) < np.mean(np.square(dynamics[:90]))
    assert train_residual.shape == dynamics[:90].shape
    assert test_residual.shape == dynamics[90:].shape
