import numpy as np

from experiments.empirical_spectral_audit.run_frozen_local_residual_bootstrap import (
    _bootstrap_paired,
    run_report,
)


def test_paired_bootstrap_detects_better_local_errors():
    pooled = [np.full(40, 2.0), np.full(40, 2.0)]
    local = [np.full(40, 1.0), np.full(40, 1.0)]
    report = _bootstrap_paired(
        pooled, local, bootstrap_repeats=200, seed=11, group_weights=np.array([0.5, 0.5])
    )
    assert report["local_minus_pooled"] == -1.0
    assert report["local_minus_pooled_ci95"] == [-1.0, -1.0]
    assert report["probability_local_better"] == 1.0


def test_run_report_is_finite_on_small_fixture(tmp_path):
    rng = np.random.default_rng(9)
    n, d = 800, 5
    latents = rng.normal(size=(n, d)).astype(np.float32)
    predicted = (latents @ np.diag([0.8, 0.7, 0.6, 0.5, 0.4])).astype(np.float32)
    actions = rng.normal(size=(n, 2)).astype(np.float32)
    source = tmp_path / "latents.npz"
    np.savez(source, latents=latents, predicted_dynamics=predicted, actions=actions)
    report = run_report(source, 0, 3, [0, 1], [2, 4], 1e-3, 32, 50)
    assert len(report["records"]) == 4
    assert all(
        np.isfinite(record["local_minus_pooled"])
        and np.isfinite(record["relative_reduction_percent"])
        for record in report["records"]
    )
