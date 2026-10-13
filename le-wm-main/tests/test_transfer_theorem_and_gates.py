from __future__ import annotations

import numpy as np

from experiments.theory_spectral_transfer.finite_horizon_transfer import (
    adversarial_matched_transfer_counterexample,
)
from experiments.theory_spectral_transfer.nonlinear_transfer_certificate import (
    adversarial_nonlinear_remainder_counterexample,
    check_nonlinear_transfer_applicability,
)
from experiments.empirical_spectral_audit.independent_planning_gain_gate import (
    audit_independent_planning_gain,
)
from experiments.empirical_spectral_audit.run_cross_environment_transfer_diagnostic import (
    diagnostic as cross_environment_diagnostic,
)


def _protocol(**overrides):
    value = {
        "metric_selection_frozen": True,
        "selector_weights_frozen": True,
        "test_planning_used_for_selection": False,
        "runtime_state_access": False,
        "privileged_successor_verification": False,
        "fresh_planning_episodes": True,
        "planner_frozen": True,
        "independent_outcome_split": True,
    }
    value.update(overrides)
    return value


def test_adversarial_family_matches_scalar_diagnostics_but_not_transfer():
    report = adversarial_matched_transfer_counterexample(horizon=2)
    assert report["same_spectrum"] is True
    assert report["same_covariance"] is True
    assert report["same_commutator"] is True
    assert report["same_prediction_mse"] is True
    assert report["same_action_inverse_risk"] is True
    assert report["different_transfer"] is True
    assert report["different_planning_risk"] is True
    assert report["left"]["G_h"] != report["right"]["G_h"]


def test_nonlinear_checker_certifies_only_with_all_independent_gates():
    kwargs = {
        "local_jacobians": [np.asarray([[0.8]]), np.asarray([[0.8]])],
        "task_readout_jacobian": np.asarray([[1.0]]),
        "jacobian_lipschitz": [0.1, 0.1],
        "tube_radius": 0.1,
        "residual_envelopes": {"closure": [0.01, 0.01], "off_manifold": [0.01, 0.01], "action": [0.01, 0.01]},
        "action_support": {"passed": True, "coverage": 1.0},
        "independent_transition_consistency": {"passed": True, "split": "planner_reachable"},
        "independent_readout_calibration": {"passed": True, "split": "readout_holdout"},
        "remainder_budget": 0.01,
        "readout_remainder_bound": 0.0,
        "initial_error": 0.0,
    }
    report = check_nonlinear_transfer_applicability(**kwargs)
    assert report["status"] == "certified_local"
    assert report["certified_local"] is True

    rejected = check_nonlinear_transfer_applicability(
        **{**kwargs, "independent_transition_consistency": {"passed": False}}
    )
    assert rejected["status"] == "rejected_missing_independent_transition"

    remainder_rejected = check_nonlinear_transfer_applicability(
        **{**kwargs, "jacobian_lipschitz": [10.0, 10.0], "remainder_budget": 0.001}
    )
    assert remainder_rejected["status"] == "rejected_remainder_bound"


def test_nonlinear_jacobian_only_counterexample_has_different_rollout():
    report = adversarial_nonlinear_remainder_counterexample()
    assert report["same_local_jacobian"] is True
    assert report["same_calibration_transition_at_zero"] is True
    assert report["different_finite_horizon_transfer"] is True


def test_independent_planning_gain_requires_protocol_and_paired_inference():
    records = [
        {
            "environment": "heldout-A", "episode_id": str(index),
            "baseline_loss": 1.0, "diagnostic_loss": 0.7,
            "protocol": _protocol(),
        }
        for index in range(12)
    ]
    report = audit_independent_planning_gain(
        records, min_pairs=10, min_environments=1, bootstrap_seed=7
    )
    assert report["positive_gain"] is True
    assert report["paired_inference"]["mean"] > 0.0

    blocked = audit_independent_planning_gain(
        [{**records[0], "protocol": _protocol(runtime_state_access=True)}],
        min_pairs=1,
    )
    assert blocked["protocol_gate"] is False
    assert blocked["positive_gain"] is False


def test_cross_environment_tool_fails_closed_when_matching_is_not_replicated():
    def record(architecture, seed, environment, risk):
        return {
            "checkpoint": f"{architecture}/seed_{seed}/epoch_1",
            "group": f"{environment}/in_distribution/{architecture}/seed_{seed}",
            "model_factors": {
                "latent_dimension": 2, "training_stage": 1,
                "action_distribution": "fixed-v1", "action_inverse_risk": 0.2,
            },
            "diagnostics": {
                "prediction_mse": 1.0,
                "predictor_spectrum": [{"real": 0.8, "imag": 0.0}, {"real": 0.4, "imag": 0.0}],
                "task_metric_alignment": 0.5,
                "task_metric_commutator": 0.5,
                "task_metric_schur_offdiagonal_energy": 0.5,
                "task_metric_transient_amplification": 0.5,
            },
            "heldout": {"planning_loss": risk, "planning_risk": risk},
            "protocol": {"environment": environment, "independent_external_evidence": True},
            "_architecture": architecture, "_seed": str(seed), "_environment": environment,
            "_stage": 1, "_action_distribution": "fixed-v1", "_latent_dimension": 2,
            "_spectrum": np.asarray([0.8, 0.4], dtype=np.complex128), "_missing_metrics": [],
        }

    report = cross_environment_diagnostic([
        record("a", 1, "A", 1.0), record("b", 2, "A", 0.9),
        record("a", 3, "B", 1.0), record("b", 4, "B", 1.3),
    ], require_external=False)
    assert report["summary"]["cross_environment_positive_gate"] is False
    assert report["summary"]["status"] == "gated_cross_environment_claim"
