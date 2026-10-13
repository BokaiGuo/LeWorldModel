import numpy as np

from experiments.empirical_spectral_audit.run_horizon_operator_geometry_synthetic import (
    _operator_metrics,
)


def test_horizon_operator_metrics_are_finite_on_linear_fixture():
    rng = np.random.default_rng(13)
    source = rng.normal(size=(200, 4))
    target = source @ np.diag([0.9, 0.7, 0.5, 0.3]).T
    metrics = _operator_metrics(source, target, ridge=1e-3, seed=4, top_k=2)
    assert set(metrics) == {
        "commutator_sq",
        "heldout_residual_mse",
        "covariance_operator_overlap",
        "max_principal_angle_degrees",
        "stable_rank",
    }
    assert all(np.isfinite(value) for value in metrics.values())
