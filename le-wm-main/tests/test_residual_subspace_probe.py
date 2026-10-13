import numpy as np

from experiments.empirical_spectral_audit.run_residual_subspace_probe import _covariance_basis


def test_residual_covariance_basis_is_ordered_and_orthonormal():
    rng = np.random.default_rng(19)
    values = rng.normal(size=(200, 12))
    basis, eigenvalues = _covariance_basis(values)
    assert basis.shape == (12, 12)
    assert np.allclose(basis.T @ basis, np.eye(12), atol=1e-6)
    assert np.all(np.diff(eigenvalues) <= 1e-8)
    assert np.all(eigenvalues >= -1e-10)
