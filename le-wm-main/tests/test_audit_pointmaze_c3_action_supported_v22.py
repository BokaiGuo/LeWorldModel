from __future__ import annotations

from experiments.decision_benchmark import audit_pointmaze_c3_action_supported_v22 as audit


def test_v22_failure_audit_validates_action_support_and_margin_failure() -> None:
    result = audit.audit()
    assert result["integrity"]["v20_v22_action_hash_sets_equal"] is True
    assert result["integrity"]["v20_v22_root_overlap_count"] == 0
    assert result["integrity"]["v22_gate_sha256_matches_outcome_binding"] is True
    assert result["certificate"]["released_root_count"] == 0
    assert result["certificate"]["all_roots_failed_margin_certificate"] is True
    assert result["certificate"]["predicted_winner_margin_above_bound_count"] == 0
    assert result["diagnosis"] == "action_supported_but_unique_winner_certificate_resolution_limited"
