from __future__ import annotations

import json

import pytest

from experiments.gdd_wm.aggregate_pointmaze_repairability_frontier_v2 import aggregate


def _report(path, *, candidate_hash: str, psd: float, mlp: float, routed: float, coverage: float = 0.0):
    payload = {
        "schema": "pointmaze_execution_grounded_repairability_frontier.v2",
        "control_contract": {"outcome_execution_deferred_until_frozen": True},
        "split_isolation": {"root_disjoint": True, "roles": {"train": ["t"], "gdd": ["g"], "calibration": ["c"], "outcome": ["o"]}},
        "provenance": {"candidate_dataset": f"/{candidate_hash}.json", "candidate_dataset_sha256": candidate_hash},
        "selector_configuration": {"selected_router_family": "all", "selected_router_gdd_auc": 0.5},
        "diagnosis": {"crossfit_label_prevalence": 0.1},
        "outcome": {
            "outcome_cost_used_for_selection": False, "repair_outcome_prevalence": 0.2,
            "router_coverage": coverage, "mean_regret": {"psd": psd, "mlp": mlp, "routed": routed},
            "preselection_prediction_verdict": {"status": "not_confirmed_in_this_pilot"},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_aggregate_does_not_pool_same_root_contexts_as_independent(tmp_path):
    left = _report(tmp_path / "left.json", candidate_hash="a", psd=0.1, mlp=0.2, routed=0.1)
    right = _report(tmp_path / "right.json", candidate_hash="b", psd=0.2, mlp=0.1, routed=0.2)
    result = aggregate([left, right])
    assert result["context_count"] == 2
    assert result["transport_summary"]["all_contexts_abstained"] is True
    assert "not independent" in result["claim_boundary"]


def test_aggregate_rejects_duplicate_candidate_context(tmp_path):
    left = _report(tmp_path / "left.json", candidate_hash="same", psd=0.1, mlp=0.2, routed=0.1)
    right = _report(tmp_path / "right.json", candidate_hash="same", psd=0.1, mlp=0.2, routed=0.1)
    with pytest.raises(ValueError, match="duplicate candidate context"):
        aggregate([left, right])
