import pytest

from experiments.decision_benchmark.audit_remaining_decision_spectral_stages import (
    _p1_prime_summary,
    _p1_endpoint_readout_binding,
    _p1_execution_identity,
    _p3_replication_summary,
    _p3v2_sweep_summary,
    _real_repair_preflight_summary,
)


def _report(seed: int, *, spectral: float, ordinary: float) -> dict:
    return {
        "protocol": {"seed": seed},
        "summary": {
            "spectral_rank1": {"mean_realized_regret": spectral},
            "random_rank1": {"mean_realized_regret": spectral + 2.0},
            "pca_axis_rank1": {"mean_realized_regret": spectral + 1.0},
            "ordinary_full_effect": {"mean_realized_regret": ordinary},
        },
    }


def test_p3_replication_gate_does_not_make_a_full_effect_capacity_reference_a_rank_one_gate() -> None:
    result = _p3_replication_summary([
        _report(1, spectral=1.0, ordinary=0.9), _report(2, spectral=0.5, ordinary=0.8),
    ])
    assert result["replication_count"] == 2
    assert result["all_replicates_pass_rank_matched_controls"] is True
    assert result["full_effect_is_capacity_reference_not_primary_gate"] is True


def test_p3_replication_gate_rejects_duplicate_seed() -> None:
    with pytest.raises(ValueError, match="distinct"):
        _p3_replication_summary([_report(1, spectral=0.5, ordinary=0.8), _report(1, spectral=0.5, ordinary=0.8)])


def test_stage_audit_keeps_uncertified_p1_prime_and_prefit_bank_failure_out_of_efficacy_claims() -> None:
    p1_prime = _p1_prime_summary({
        "schema": "pointmaze_v6_prospective_routing_post_execution_audit.v1",
        "status": "completed_post_execution_prospective_routing_audit",
        "decision": {"fresh_panel_route_warranted": False},
        "certification": {"would_authorize_on_calibration": False},
    })
    preflight = _real_repair_preflight_summary({
        "schema": "pointmaze_endpoint_repairability_router_v10_static_preflight.v1",
        "status": "rejected_static_preflight",
        "checks": {"visible_history": True, "top2_gap_lower_tail": False},
    })
    assert p1_prime["status"] == "no_go_uncertified_pre_outcome_route"
    assert preflight["status"] == "blocked_pre_fit_candidate_bank"
    assert preflight["failed_checks"] == ["top2_gap_lower_tail"]


def test_stage_audit_requires_verified_endpoint_readout_binding_for_each_outcome_root() -> None:
    audit = {
        "splits": {"outcome": {"per_root": {
            "one": {"endpoint_readout_binding": {
                "endpoint_candidate_is_deterministic_argmin": True,
                "realized_regret_recomputed_from_same_candidate_ordering": True,
                "maximum_absolute_cost_error": 0.0,
            }},
        }}},
    }
    result = _p1_endpoint_readout_binding(audit)
    assert result["status"] == "verified_endpoint_readout_binding"
    audit["splits"]["outcome"]["per_root"]["one"]["endpoint_readout_binding"]["endpoint_candidate_is_deterministic_argmin"] = False
    with pytest.raises(ValueError, match="does not bind endpoint choice"):
        _p1_endpoint_readout_binding(audit)


def test_stage_audit_requires_zero_error_independent_outcome_replay() -> None:
    identity = {
        "schema": "pointmaze_endpoint_execution_identity_audit.v1",
        "status": "passed_independent_outcome_identity_replay",
        "outcome_root_count": 2, "outcome_candidate_count": 32,
        "maximum_terminal_absolute_error": 0.0, "maximum_cost_absolute_error": 0.0,
    }
    assert _p1_execution_identity(identity)["status"] == "verified_independent_outcome_identity_replay"
    identity["maximum_cost_absolute_error"] = 1.0e-4
    with pytest.raises(ValueError, match="non-zero"):
        _p1_execution_identity(identity)


def test_p3v2_sweep_requires_common_random_numbers_and_retains_conditional_boundary() -> None:
    result = _p3v2_sweep_summary({
        "schema": "spectral_repair_synthetic_p3_v2_task_metric_sweep_cuda.v2",
        "status": "completed_task_metric_mechanism_sweep",
        "common_random_numbers": True,
        "records": [
            {"seed": 7, "task_output_0_weight": 64.0, "status": "failed_task_weighted_rank_matched_calibration"},
            {"seed": 7, "task_output_0_weight": 1024.0, "status": "passed_task_weighted_rank_matched_calibration"},
        ],
    })
    assert result["status"] == "conditional_task_critical_mode_calibration_pass"
    assert result["passed_task_output_0_weights"] == [1024.0]
