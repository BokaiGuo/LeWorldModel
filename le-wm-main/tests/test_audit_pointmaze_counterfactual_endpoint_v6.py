import json
import hashlib
from pathlib import Path

from experiments.gdd_wm.audit_pointmaze_counterfactual_endpoint_v6 import audit


def _report() -> dict:
    ordered = {
        "action_representation": "full_ordered_20x2",
        "input_fields": ["source_history_model", "goal_raw", "action_sequence"],
    }
    summary = {
        "action_representation": "canonical_summary_mean_std_first_last",
        "input_fields": ["source_history_model", "goal_raw", "action_sequence_summary"],
    }
    return {
        "schema": "pointmaze_counterfactual_endpoint_control.v1",
        "status": "completed_counterfactual_endpoint_control",
        "split_isolation": {
            "root_disjoint": True, "candidate_generalization_regime": "shared_candidate_in_support",
            "candidate_design": "goal_balanced_interaction_pairs",
            "goal_balanced_design_winner_entropy_normalized": 0.8,
            "minimum_goal_balanced_design_winner_entropy_normalized": 0.5,
            "outcome_labels_sealed_before_execution": True,
            "frozen_preflight_ledger_sha256": "a" * 64,
            "frozen_preflight_ledger_context_count": 33,
        },
        "action_representation_stress": {
            "minimum_summary_collision_pairs_per_root": 8,
            "per_root": {
                "train/r": {"summary_collision_pair_count": 8},
                "calibration/r": {"summary_collision_pair_count": 8},
                "outcome/r": {"summary_collision_pair_count": 8},
            },
        },
        "summary_prediction_aliasing_pre_outcome": {
            "calibration": {
                "checked_before_outcome_execution": True, "score_tolerance": 1.0e-12,
                "per_root": {"r": {"maximum_summary_cost_score_delta": 0.0, "maximum_summary_endpoint_score_delta": 0.0}},
            },
            "outcome": {
                "checked_before_outcome_execution": True, "score_tolerance": 1.0e-12,
                "per_root": {"r": {"maximum_summary_cost_score_delta": 0.0, "maximum_summary_endpoint_score_delta": 0.0}},
            },
        },
        "endpoint_ensemble_diagnostics_pre_outcome": {
            "calibration": {"available": True, "root_count": 1, "per_root": {"r": {}}},
            "outcome": {"available": True, "root_count": 1, "per_root": {"r": {}}},
        },
        "control_contract": {
            "primary_authorization_holds_ordered_action_representation_fixed": True,
            "outcome_execution_deferred_until_policy_and_contrast_bound_frozen": True,
            "endpoint_ensemble_disagreement_diagnostic_only": True,
            "factorial_diagnostic": {"selection_uses_ordered_endpoint_vs_ordered_cost_only": True},
        },
        "arms": {
            "cost_only": ordered, "planning_shaped_auxiliary": ordered, "counterfactual_endpoint": ordered,
            "summary_cost_only": summary, "summary_counterfactual_endpoint": summary,
        },
        "calibration": {"policy": {"deployment_policy": "endpoint"}},
        "outcome": {"authorized_deployment_policy": "endpoint"},
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bound_report(tmp_path: Path, payload: dict) -> Path:
    ledger = tmp_path / "preflight.json"
    ledger.write_text(json.dumps({
        "schema": "pointmaze_counterfactual_endpoint_v6_static_preflight.v1",
        "status": "passed_static_preflight", "checks": {"all": True},
    }))
    ledger_hash = _sha256(ledger)
    payload["split_isolation"]["frozen_preflight_ledger_sha256"] = ledger_hash
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"metadata": {"frozen_preflight_ledger": {
        "schema": "pointmaze_counterfactual_endpoint_v6_static_preflight.v1",
        "status": "passed_static_preflight", "path": str(ledger), "sha256": ledger_hash,
        "context_count": payload["split_isolation"]["frozen_preflight_ledger_context_count"],
    }}}))
    payload["provenance"] = {"context": str(context), "context_sha256": _sha256(context)}
    report = tmp_path / "report.json"
    report.write_text(json.dumps(payload))
    return report


def test_v6_contract_audit_accepts_complete_report(tmp_path) -> None:
    path = _write_bound_report(tmp_path, _report())
    assert audit(path)["status"] == "passed_contract_audit"


def test_v6_contract_audit_rejects_summary_side_channel_contract(tmp_path) -> None:
    payload = _report()
    payload["arms"]["summary_counterfactual_endpoint"]["action_representation"] = "full_ordered_20x2"
    path = _write_bound_report(tmp_path, payload)
    result = audit(path)
    assert result["status"] == "rejected_contract_audit"
    assert result["checks"]["summary_arm_contracts"] is False


def test_v6_contract_audit_rejects_missing_calibration_alias_or_split_collision_evidence(tmp_path) -> None:
    payload = _report()
    payload["summary_prediction_aliasing_pre_outcome"].pop("calibration")
    payload["action_representation_stress"]["per_root"].pop("calibration/r")
    path = _write_bound_report(tmp_path, payload)
    result = audit(path)
    assert result["status"] == "rejected_contract_audit"
    assert result["checks"]["strict_collision_stress"] is False
    assert result["checks"]["summary_alias_checked_pre_outcome"] is False
    assert result["checks"]["summary_alias_preserved"] is False


def test_v6_contract_audit_rejects_unbound_static_preflight_ledger(tmp_path) -> None:
    path = _write_bound_report(tmp_path, _report())
    payload = json.loads(path.read_text())
    payload["split_isolation"].pop("frozen_preflight_ledger_sha256")
    path.write_text(json.dumps(payload))
    result = audit(path)
    assert result["status"] == "rejected_contract_audit"
    assert result["checks"]["frozen_preflight_ledger_bound"] is False


def test_v6_contract_audit_rejects_tampered_preflight_ledger_bytes(tmp_path) -> None:
    path = _write_bound_report(tmp_path, _report())
    ledger = tmp_path / "preflight.json"
    ledger.write_text(json.dumps({"schema": "tampered"}))
    result = audit(path)
    assert result["status"] == "rejected_contract_audit"
    assert result["checks"]["frozen_preflight_ledger_hash_chain"] is False


def test_v6_contract_audit_rejects_missing_pre_outcome_ensemble_diagnostics(tmp_path) -> None:
    payload = _report()
    payload["endpoint_ensemble_diagnostics_pre_outcome"]["outcome"]["available"] = False
    path = _write_bound_report(tmp_path, payload)
    result = audit(path)
    assert result["status"] == "rejected_contract_audit"
    assert result["checks"]["endpoint_ensemble_diagnostics_pre_outcome"] is False
