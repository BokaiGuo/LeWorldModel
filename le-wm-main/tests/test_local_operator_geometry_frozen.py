import numpy as np

from experiments.empirical_spectral_audit.run_local_operator_geometry_frozen import (
    _action_labels,
    _commutator_sq,
    _fit_operator,
    run_report,
)


def test_commutator_is_zero_for_diagonal_pair():
    covariance = np.diag([3.0, 2.0, 1.0])
    operator = np.diag([0.8, 0.6, 0.4])
    assert _commutator_sq(covariance, operator) < 1e-12


def test_action_labels_have_requested_bins():
    actions = np.arange(80, dtype=np.float64).reshape(20, 4)
    labels = _action_labels(actions, 4)
    assert set(labels.tolist()) == {0, 1, 2, 3}


def test_fit_operator_recovers_noiseless_map():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(300, 3))
    row_operator = np.diag([0.7, 0.5, 0.3])
    target = source @ row_operator
    fit = _fit_operator(source, target, ridge=1e-8)
    assert np.max(np.abs(fit["row_operator"] - row_operator)) < 1e-5


def test_run_report_is_finite_on_small_fixture(tmp_path):
    rng = np.random.default_rng(8)
    n, d = 800, 5
    latents = rng.normal(size=(n, d)).astype(np.float32)
    predicted = (latents @ np.diag([0.8, 0.7, 0.6, 0.5, 0.4])).astype(np.float32)
    actions = rng.normal(size=(n, 2)).astype(np.float32)
    source = tmp_path / "latents.npz"
    np.savez(source, latents=latents, predicted_dynamics=predicted, actions=actions)
    report = run_report(source, 0, 3, [0, 1], [2, 4], 1e-3, 32, 2)
    assert len(report["records"]) == 4
    assert all(np.isfinite(value) for record in report["records"] for value in [record["pooled_crossfit"]["commutator_sq_weighted"]])
