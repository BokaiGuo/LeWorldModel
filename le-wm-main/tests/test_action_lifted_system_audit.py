import numpy as np

from experiments.empirical_spectral_audit.run_action_lifted_system_audit import (
    action_excitation,
    build_report,
    counterfactual_transfer_error,
    finite_horizon_diagnostics,
    persistent_excitation,
    residualized_action_excitation,
)


def test_persistent_excitation_distinguishes_full_and_rank_deficient_support():
    rng = np.random.default_rng(4)
    eta = rng.normal(size=(128, 2))
    action = rng.normal(size=(128, 1))
    full = persistent_excitation(eta, action, threshold=1e-4)
    deficient = persistent_excitation(eta, np.zeros_like(action), threshold=1e-4)
    assert full["passes"]
    assert not deficient["passes"]
    assert deficient["rank"] < deficient["dimension"]


def test_action_only_excitation_is_not_rejected_by_high_dimensional_latent_rank():
    rng = np.random.default_rng(41)
    eta = rng.normal(size=(32, 64))
    action = rng.normal(size=(32, 3))
    input_gate = action_excitation(action, threshold=1e-4)
    residual_gate = residualized_action_excitation(eta, action, threshold=1e-8)
    assert input_gate["passes"]
    assert residual_gate["rank"] == 3


def test_gramians_and_transfer_are_finite_and_pbh_localizes_uncontrollable_mode():
    A = np.diag([0.9, 0.7])
    B = np.asarray([[1.0], [0.0]])
    D = np.eye(2)
    report = finite_horizon_diagnostics(A, B, D, horizon=4)
    assert np.allclose(report["transfer_operators"][0], np.asarray([[1.0], [0.0]]))
    assert np.asarray(report["controllability_eigenvalues"]).min() >= -1e-10
    assert np.asarray(report["observability_eigenvalues"]).min() > 0.0
    assert any(not row["controllable"] for row in report["pbh"])
    assert np.isfinite(report["max_transient_gain"])


def test_counterfactual_transfer_error_is_zero_for_identical_contrasts():
    rng = np.random.default_rng(5)
    real = rng.normal(size=(6, 4, 2))
    exact = counterfactual_transfer_error(real, real)
    shifted = counterfactual_transfer_error(real + 0.25, real)
    assert exact["rmse"] == 0.0
    assert np.isclose(shifted["rmse"], 0.0)  # common source shift cancels in contrasts
    changed = real.copy()
    changed[:, 2] += np.asarray([1.0, -0.5])
    assert counterfactual_transfer_error(changed, real)["rmse"] > 0.0


def test_build_report_has_protocol_boundary_and_finite_primary_fields():
    rng = np.random.default_rng(8)
    eta = rng.normal(size=(96, 3))
    action = rng.normal(size=(96, 2))
    A = np.diag([0.8, 0.7, 0.6])
    B = np.asarray([[1.0, 0.0], [0.0, 0.5], [0.2, 0.1]])
    D = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    next_eta = eta @ A.T + action @ B.T
    task = eta @ D.T
    report = build_report(eta, next_eta, action, task, horizons=(1, 3))
    assert report["protocol"]["study_scope"] == "covariance_predictor_unified"
    assert "not causal identification" in report["protocol"]["claim_boundary"]
    assert report["finite_horizon"]["1"]["horizon"] == 1
    assert report["finite_horizon"]["3"]["horizon"] == 3
