from __future__ import annotations

from experiments.decision_benchmark import audit_pointmaze_c3_regret_budget_v23 as audit


def test_v23_audit_validates_sealed_positive_transfer_with_coverage_boundary() -> None:
    result = audit.audit()
    assert result["integrity"]["split_root_disjoint"] is True
    assert result["integrity"]["held_actions_supported_by_train"] is True
    assert result["integrity"]["gate_sha256_matches_outcome_binding"] is True
    assert result["certificate"]["epsilon_regret_released_root_count"] == 239
    assert result["certificate"]["unique_winner_released_root_count"] == 23
    assert result["outcome"]["paired_regret_gain_bootstrap"]["lower"] > 0.0
    assert result["outcome"]["by_regime"]["high_motion"]["root_count"] == 128
    assert result["outcome"]["by_regime"]["direct_zero_motion"]["root_count"] == 128
    assert result["outcome"]["calibration_by_regime"]["high_motion"]["root_count"] == 128
    assert result["outcome"]["calibration_by_regime"]["direct_zero_motion"]["root_count"] == 128
    assert result["diagnosis"] == "positive_epsilon_regret_transfer_with_empirical_coverage_shortfall"
