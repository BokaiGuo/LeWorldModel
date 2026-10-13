import numpy as np

from experiments.empirical_spectral_audit.run_frozen_local_operator_permutation_null import (
    _partition_mse,
    run_report,
)


def test_partition_mse_is_finite_on_small_fixture():
    rng = np.random.default_rng(12)
    source = rng.normal(size=(400, 5))
    target = source @ np.diag([0.8, 0.7, 0.6, 0.5, 0.4])
    labels = np.repeat(np.arange(2), 200)
    train = np.arange(0, 400, 2)
    test = np.arange(1, 400, 2)
    result = _partition_mse(source, target, labels, train, test, ridge=1e-3, min_samples=32)
    assert np.isfinite(result["local_mse"])
    assert np.isfinite(result["pooled_mse"])


def test_permutation_report_is_finite(tmp_path):
    rng = np.random.default_rng(13)
    n, d = 800, 5
    latents = rng.normal(size=(n, d)).astype(np.float32)
    predicted = (latents @ np.diag([0.8, 0.7, 0.6, 0.5, 0.4])).astype(np.float32)
    actions = rng.normal(size=(n, 2)).astype(np.float32)
    source = tmp_path / "latents.npz"
    np.savez(source, latents=latents, predicted_dynamics=predicted, actions=actions)
    report = run_report(source, 0, 3, [0, 1], [2, 4], 1e-3, 32, 3)
    assert len(report["records"]) == 4
    assert all(
        np.isfinite(record["permutation_null"]["reduction_mean_percent"])
        for record in report["records"]
    )
