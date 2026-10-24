from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from experiments.decision_benchmark.audit_pointmaze_decision_spectral_mismatch import _pre_outcome_primary_metric, audit


def test_audit_requires_hash_bound_pre_outcome_endpoints_and_uses_root_units(tmp_path) -> None:
    actions = [
        [[-1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[0.0, -1.0]],
    ]
    prediction = {
        "candidate_ids": ["0", "1", "2", "3"], "candidate_action_sequences": actions,
        "goal_raw": [0.0, 0.0], "endpoint_candidate_id": "0", "cost_candidate_id": "1",
        "endpoint_predicted_cost": [0.8, 1.2, 1.1, 1.0],
        "endpoint_predicted_terminal_state": [[-0.8, 0.0, 0.0, 0.0], [0.0, 1.2, 0.0, 0.0], [1.1, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]],
    }
    bundle = {
        "schema": "pointmaze_counterfactual_endpoint_pre_outcome_predictions.v1",
        "outcome_labels_materialized": False,
        "calibration_predictions": {"cal": prediction}, "outcome_predictions": {"out": prediction},
    }
    bundle_path = tmp_path / "pre.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    realized = {
        "candidate_ids": ["0", "1", "2", "3"],
        "realized_candidate_terminal_state": [[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]],
        "endpoint_regret": 0.0, "cost_regret": 0.0,
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({
        "pre_outcome_prediction_bundle": {"schema": bundle["schema"], "path": str(bundle_path), "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest()},
        "calibration": {"per_root": {"cal": realized}}, "outcome": {"per_root": {"out": realized}},
    }), encoding="utf-8")
    output = tmp_path / "audit.json"
    result = audit(report_path=report_path, output=output)
    assert result["protocol"]["metric_selected_before_outcomes"] is True
    assert result["splits"]["outcome"]["per_root"]["out"]["support_rank"] == 2
    assert "frozen_pair_leverage_vs_endpoint_regret" in result["splits"]["outcome"]
    assert "frozen_pair_exact_task_contrast_error_vs_endpoint_regret" in result["splits"]["outcome"]
    assert "frozen_pair_exact_task_contrast_error_vs_endpoint_squared_regret" in result["splits"]["outcome"]
    assert result["splits"]["outcome"]["per_root"]["out"]["frozen_pair_exact_task_contrast"]["absolute_contrast_error"] >= 0.0
    assert result["splits"]["outcome"]["per_root"]["out"]["endpoint_squared_regret"] == 0.0
    assert result["splits"]["outcome"]["per_root"]["out"]["endpoint_readout_binding"]["endpoint_candidate_is_deterministic_argmin"] is True
    assert output.is_file()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("endpoint_predicted_cost", [1.8, 1.2, 1.1, 1.0], "do not equal Euclidean"),
        ("endpoint_candidate_id", "1", "not the deterministic argmin"),
    ],
)
def test_audit_rejects_broken_sealed_endpoint_score_binding(tmp_path, field, value, message) -> None:
    actions = [[[ -1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[0.0, -1.0]]]
    prediction = {
        "candidate_ids": ["0", "1", "2", "3"], "candidate_action_sequences": actions,
        "goal_raw": [0.0, 0.0], "endpoint_candidate_id": "0", "cost_candidate_id": "1",
        "endpoint_predicted_cost": [0.8, 1.2, 1.1, 1.0],
        "endpoint_predicted_terminal_state": [[-0.8, 0.0, 0.0, 0.0], [0.0, 1.2, 0.0, 0.0], [1.1, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]],
    }
    prediction[field] = value
    bundle = {"schema": "pointmaze_counterfactual_endpoint_pre_outcome_predictions.v1", "outcome_labels_materialized": False, "calibration_predictions": {"root": prediction}, "outcome_predictions": {"root": prediction}}
    bundle_path = tmp_path / "pre.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    realized = {"candidate_ids": ["0", "1", "2", "3"], "realized_candidate_terminal_state": [[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]], "endpoint_regret": 0.0, "cost_regret": 0.0}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"pre_outcome_prediction_bundle": {"schema": bundle["schema"], "path": str(bundle_path), "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest()}, "calibration": {"per_root": {"root": realized}}, "outcome": {"per_root": {"root": realized}}}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        audit(report_path=report_path, output=tmp_path / "audit.json")


def test_hash_bound_pre_outcome_spec_can_promote_only_its_declared_primary_metric(tmp_path) -> None:
    spec = {
        "schema": "pointmaze_p1_pre_outcome_diagnostic_spec.v1", "selected_before_outcomes": True,
        "primary_metric": "frozen_pair_squared_task_contrast_error",
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    metric, binding = _pre_outcome_primary_metric({
        "pre_outcome_diagnostic_spec": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "spec": spec},
    })
    assert metric == "frozen_pair_squared_task_contrast_error"
    assert binding["spec"] == spec


def test_squared_pair_contrast_primary_uses_squared_regret(tmp_path) -> None:
    """The declared squared task cost must also define the outcome target."""
    actions = [[[ -1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[0.0, -1.0]]]
    prediction = {
        "candidate_ids": ["0", "1", "2", "3"], "candidate_action_sequences": actions,
        "goal_raw": [0.0, 0.0], "endpoint_candidate_id": "0", "cost_candidate_id": "1",
        "endpoint_predicted_cost": [0.1, 1.0, 2.0, 3.0],
        "endpoint_predicted_terminal_state": [[-0.1, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [0.0, -3.0, 0.0, 0.0]],
    }
    spec = {"schema": "pointmaze_p1_pre_outcome_diagnostic_spec.v1", "selected_before_outcomes": True,
            "primary_metric": "frozen_pair_squared_task_contrast_error"}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    bundle = {
        "schema": "pointmaze_counterfactual_endpoint_pre_outcome_predictions.v1", "outcome_labels_materialized": False,
        "pre_outcome_diagnostic_spec": {"path": str(spec_path), "sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(), "spec": spec},
        "calibration_predictions": {"cal": prediction}, "outcome_predictions": {"out": prediction},
    }
    bundle_path = tmp_path / "pre.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    realized = {"candidate_ids": ["0", "1", "2", "3"],
                "realized_candidate_terminal_state": [[-2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [0.0, -4.0, 0.0, 0.0]],
                "endpoint_regret": 1.0, "cost_regret": 0.0}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({
        "pre_outcome_prediction_bundle": {"schema": bundle["schema"], "path": str(bundle_path), "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest()},
        "calibration": {"per_root": {"cal": realized}}, "outcome": {"per_root": {"out": realized}},
    }), encoding="utf-8")
    result = audit(report_path=report_path, output=tmp_path / "audit.json")
    outcome = result["splits"]["outcome"]
    assert outcome["per_root"]["out"]["endpoint_squared_regret"] == 3.0
    assert outcome["primary_metric_association"] is outcome["frozen_pair_exact_task_contrast_error_vs_endpoint_squared_regret"]
    assert result["protocol"]["pair_contrast_primary_outcome"].startswith("endpoint_squared_regret")
