import json

import pytest

from experiments.gdd_wm.audit_pointmaze_endpoint_repairability_router_v10 import audit


def _row(*, released: bool, cost: float, endpoint: float) -> dict:
    return {
        "candidate_ids": ["cost", "endpoint"], "cost_candidate_id": "cost",
        "endpoint_candidate_id": "endpoint", "routed_candidate_id": "endpoint" if released else "cost",
        "candidate_disagreement": True, "routing_eligible": True, "released_endpoint": released,
        "cost_regret": cost, "endpoint_regret": endpoint,
        "routed_regret": endpoint if released else cost,
    }


def _report(*, policy: str = "routed") -> dict:
    return {
        "schema": "pointmaze_endpoint_repairability_router_v10.v1",
        "status": "completed_execution_grounded_endpoint_repairability_router",
        "control_contract": {
            "repair_arm": "endpoint_coupled_counterfactual", "fallback_arm": "direct_cost_only",
            "same_choice_roots_forced_to_cost": True,
            "outcome_execution_deferred_until_router_threshold_and_certificate_frozen": True,
            "outcome_cost_used_for_router_or_threshold_selection": False,
        },
        "calibration": {"routing": {"deployment_policy": policy, "routing_authorized": policy == "routed"}},
        "outcome": {"per_root": {
            "root-0": _row(released=policy == "routed", cost=1.0, endpoint=0.0),
            "root-1": _row(released=policy == "routed", cost=0.8, endpoint=0.0),
            "root-2": _row(released=False, cost=0.0, endpoint=0.2),
        }},
    }


def test_v10_outcome_audit_requires_a_frozen_routed_gain(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    result = audit(path, seed=7)
    assert result["status"] == "validated_held_out_endpoint_repair_gain"
    assert result["outcome_endpoint_release_count"] == 2
    assert result["outcome_upper_95_routed_minus_cost_regret"] <= 0.0


def test_v10_outcome_audit_recognizes_fallback_without_fabricating_gain(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report(policy="cost")), encoding="utf-8")
    result = audit(path, seed=7)
    assert result["status"] == "validated_fail_closed_cost_fallback"
    assert result["outcome_endpoint_release_count"] == 0


def test_v10_outcome_audit_rejects_same_choice_release(tmp_path) -> None:
    report = _report()
    row = report["outcome"]["per_root"]["root-2"]
    row.update({"endpoint_candidate_id": "cost", "candidate_disagreement": False, "routing_eligible": False, "released_endpoint": True, "routed_candidate_id": "endpoint", "routed_regret": 0.2})
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="same-choice root"):
        audit(path, seed=7)


def test_v10_outcome_audit_refuses_a_pre_outcome_opportunity_rejection(tmp_path) -> None:
    report = _report()
    report["status"] = "rejected_pre_outcome_calibration_opportunity"
    report["outcome"] = {"materialized": False}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="requires a V10 endpoint-repairability report"):
        audit(path, seed=7)


def test_v10_outcome_audit_refuses_a_train_only_crossfit_rejection(tmp_path) -> None:
    report = _report()
    report["status"] = "rejected_pre_calibration_crossfit_opportunity"
    report["outcome"] = {"materialized": False}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="requires a V10 endpoint-repairability report"):
        audit(path, seed=7)
