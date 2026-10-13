import numpy as np
import pytest

from experiments.research_protocol_contracts import (
    heldout_planning_risk_prediction,
    protocol_completion_matrix,
    rgb_successor_verification_gate,
    terminal_pose_closure_gate,
    validate_matched_quality_record,
    validate_rgb_only_observation,
)


def test_rgb_only_boundary_rejects_privileged_state():
    with pytest.raises(ValueError, match="forbidden privileged"):
        validate_rgb_only_observation({"rgb": np.zeros((4, 4, 3)), "state": np.zeros(7)})


def test_rgb_successor_gate_is_fail_closed_without_both_images():
    result = rgb_successor_verification_gate(np.zeros((2, 2)), None)
    assert result["status"] == "blocked"
    assert not result["passed"]


def test_rgb_successor_gate_reports_image_consistency():
    result = rgb_successor_verification_gate(np.zeros((2, 2)), np.zeros((2, 2)))
    assert result["status"] == "pass"
    assert result["normalized_l1_error"] == 0.0


def test_matched_quality_contract_requires_all_factors_and_heldout_risk():
    result = validate_matched_quality_record({"diagnostics": {}, "model_factors": {}})
    assert result["status"] == "blocked"
    assert "heldout_planning_risk" in result["missing"]


def test_planning_risk_prediction_is_independent_pairwise_contract():
    result = heldout_planning_risk_prediction([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1])
    assert result["status"] == "pass"
    assert result["spearman_metric_vs_risk"] == pytest.approx(-1.0)


def test_pose_gate_requires_runtime_successor_verification_and_hold():
    row = {"final_xy": 1.0, "final_angle": 0.05, "hold_steps": 2, "runtime_successor_verification": False}
    result = terminal_pose_closure_gate([row])
    assert result["status"] == "fail"
    assert not result["passed"]


def test_protocol_matrix_marks_code_presence_separately_from_evidence():
    matrix = protocol_completion_matrix()
    assert matrix["adaptive_event_memory"]["status"] == "implemented"
    assert "positive scientific evidence" in matrix["claim_boundary"]
