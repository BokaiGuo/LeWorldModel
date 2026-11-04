import json
from pathlib import Path

import pytest

from experiments.spectral_reachability.pc_atlas_depth3_pruning_decision_audit_20261103_codex import explain


def test_explanation_checks_policy_and_records_eligibility():
    trace = {"root_index": 0, "root_name": "zero", "selected_indices_in_beam_order": [0, 2],
             "candidates": [{"cost": 1., "lookahead_cost": 1., "last_action": [0., 0.]},
                            {"cost": 2., "lookahead_cost": 1.005, "last_action": [.1, 0.]},
                            {"cost": 3., "lookahead_cost": 1.006, "last_action": [.9, 0.]}]}
    result = explain(trace, "lookahead_diverse", .01)
    assert result["selected_indices_in_beam_order"] == [0, 2]
    assert all(c["within_lookahead_tolerance"] for c in result["candidates"])
    with pytest.raises(ValueError, match="declared policy"):
        explain(trace, "lookahead", .01)


def test_saved_experiment_decisions_match_declared_policies():
    path = Path(__file__).resolve().parents[1] / "results/pc_atlas_depth3_pruning_alternative_20261103_codex.json"
    pairs = json.loads(path.read_text())["pairs"]
    for pair in pairs:
        for mode, arm in pair["arms"].items():
            for trace in arm["intermediate_trace"]:
                assert len(explain(trace, mode, arm["configuration"]["tolerance"])["candidates"]) >= 2
