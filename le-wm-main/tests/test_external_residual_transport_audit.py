import numpy as np

from experiments.empirical_spectral_audit.run_external_residual_transport_audit import (
    _band_profile,
    _orthonormal_basis,
)


def test_band_profile_is_finite_and_uses_orthonormal_coordinates():
    rng = np.random.default_rng(23)
    error = rng.normal(size=(5, 192))
    basis = _orthonormal_basis(192, 20261003)
    profile = _band_profile(error, basis)
    assert all(np.isfinite(value) for value in profile.values())
    assert all(0.0 <= value <= 1.0 for value in profile.values())
    assert np.allclose(basis.T @ basis, np.eye(192), atol=2e-5)
