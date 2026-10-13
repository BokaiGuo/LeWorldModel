import numpy as np

from experiments.empirical_spectral_audit.run_frozen_subspace_probes import (
    _band_indices,
    _orthonormal_subspace,
    _ridge_probe,
)


def test_band_indices_cover_expected_ranges():
    assert np.array_equal(_band_indices(12, "top", 3), np.array([0, 1, 2]))
    assert np.array_equal(_band_indices(12, "middle", 4), np.array([4, 5, 6, 7]))
    assert np.array_equal(_band_indices(12, "tail", 3), np.array([9, 10, 11]))


def test_random_subspace_is_orthonormal_and_ridge_probe_is_finite():
    basis = _orthonormal_subspace(8, 3, np.random.default_rng(4))
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-6)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 8))
    y = x[:, :2] + 0.01 * rng.normal(size=(40, 2))
    report = _ridge_probe(x[:30], x[30:], y[:30], y[30:], ridge=1e-3)
    assert np.isfinite(report["r2"])
    assert report["feature_dim"] == 8
