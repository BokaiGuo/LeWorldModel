from __future__ import annotations

from experiments.decision_benchmark import audit_pointmaze_c3_stratified_repair_v24 as audit


def test_v24_audit_separates_the_positive_high_motion_result_from_direct_coverage_boundary() -> None:
    result = audit.audit()
    assert result["integrity"]["split_root_disjoint"] is True
    assert result["integrity"]["held_actions_supported_by_train"] is True
    assert result["integrity"]["gate_sha256_matches_outcome_binding"] is True
    high = result["certificate"]["strata"]["high_motion"]
    direct = result["certificate"]["strata"]["direct_zero_motion"]
    assert high["release_count"] == 121
    assert high["paired_regret_gain_bootstrap"]["lower"] > 0.0
    assert high["coverage"] >= .9
    assert direct["paired_regret_gain"] == 0.0
    assert direct["coverage"] < .9
    assert result["diagnosis"] == "high_motion_repair_transfers_with_direct_stratum_residual_coverage_shortfall"
