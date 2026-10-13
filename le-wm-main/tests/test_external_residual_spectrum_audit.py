import numpy as np

from experiments.empirical_spectral_audit.run_external_residual_spectrum_audit import (
    _spectral_profile,
)


def test_spectral_profile_recovers_top_band_residual():
    rng = np.random.default_rng(17)
    target = rng.normal(size=(80, 192))
    target[:, :8] *= 10.0
    residual = np.zeros_like(target)
    residual[:, :8] = 2.0
    report = _spectral_profile(target, target + residual, random_trials=16, seed=3)
    assert report["residual_mse"] > 0.0
    assert report["bands"]["0:8"]["share"] > 0.95
    assert report["random_basis_null"]["observed_minus_null_mean"] > 0.5


def test_spectral_profile_is_finite_for_isotropic_residual():
    rng = np.random.default_rng(18)
    target = rng.normal(size=(80, 192))
    prediction = target + rng.normal(size=(80, 192))
    report = _spectral_profile(target, prediction, random_trials=8, seed=4)
    assert np.isfinite(report["residual_mse"])
    assert all(np.isfinite(item["share"]) for item in report["bands"].values())
