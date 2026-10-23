import hashlib
import json

from experiments.gdd_wm.audit_pointmaze_prospective_repairability_matrix_v1 import (
    ARMS,
    DECISION_SCHEMA,
    OUTCOME_SCHEMA,
    POLICIES,
    audit,
)


def _write_fixture(tmp_path, *, leak=False):
    rows, labels = [], {}
    routes = {name: {} for name in POLICIES}
    for root_index in range(24):
        root = f"root-{root_index}"
        for policy in POLICIES:
            routes[policy][root] = {
                "repairability_diagnostic": "uniform_history_repair",
                "uniform_history_repair": "uniform_history_repair",
                "ordinary_preference": "ordinary_preference",
            }.get(policy, "action_effect_only")
        for candidate in range(3):
            # The history and preference arms place the true winner first; the
            # carrier deliberately selects the last candidate.  No outcome is
            # stored in the decision payload.
            scores = {
                "action_effect_only": float(2 - candidate),
                "uniform_history_repair": float(candidate),
                "ordinary_preference": float(candidate),
            }
            row = {"root_id": root, "candidate_id": candidate, "split": "outcome", "scores": scores}
            if leak: row["true_cost"] = float(candidate)
            rows.append(row)
            labels[f"{root}/{candidate}"] = {"real_task_cost": float(candidate)}
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"schema": DECISION_SCHEMA, "outcome_labels_opened": False, "records": rows, "root_routes": routes}), encoding="utf-8")
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text(json.dumps({"schema": OUTCOME_SCHEMA, "decision_sha256": hashlib.sha256(decision.read_bytes()).hexdigest(), "labels": labels}), encoding="utf-8")
    return decision, outcomes


def test_matrix_audit_reports_root_level_incremental_gain_and_holm_controls(tmp_path):
    decision, outcomes = _write_fixture(tmp_path)
    result = audit(decision, outcomes)
    assert result["integrity"]["outcome_bound_to_sealed_decision"] is True
    assert result["methods"]["repairability_diagnostic"]["mean_regret"] == 0.0
    comparison = result["paired_comparisons"]["repairability_diagnostic_over_action_effect_only"]
    assert comparison["gain"]["lower"] > 0.0
    assert 0.0 <= comparison["holm_adjusted_p"] <= 1.0


def test_matrix_audit_refuses_outcome_label_leakage_into_sealed_decision(tmp_path):
    decision, outcomes = _write_fixture(tmp_path, leak=True)
    try:
        audit(decision, outcomes)
    except ValueError as error:
        assert "outcome labels" in str(error)
    else:
        raise AssertionError("expected sealed decision leakage to be rejected")
