from __future__ import annotations

import json
from pathlib import Path

from experiments.gdd_wm.audit_tworoom_support_gated_repair_v1 import audit


def _report(roots: list[int]) -> dict:
    arms = {
        "effect_adapter": {"roots": roots, "final_distance": [2.0] * len(roots), "success": [0] * len(roots)},
        "action_only": {"roots": roots, "final_distance": [3.0] * len(roots), "success": [0] * len(roots)},
        "no_dynamics": {"roots": roots, "final_distance": [4.0] * len(roots), "success": [0] * len(roots)},
        "support_gated": {"roots": roots, "final_distance": [1.0] * len(roots), "success": [1] * len(roots)},
    }
    ranking = {
        name: {"outcome": {"root_regret": {str(root): 0.0 if name == "support_gated" else 1.0 for root in roots}}}
        for name in arms
    }
    return {
        "schema": "tworoom_action_effect_repairability.v1",
        "protocol": {
            "root_counts": {"train": 96, "calibration": 32, "outcome": 32},
            "root_ids": {"outcome": roots}, "runtime_state_access": False,
            "training_outcome_roots_read": False, "support_gated_repair": True,
        },
        "closed_loop": arms, "ranking": ranking,
        "support_gate": {"outcome_labels_read": False, "status": "authorized_calibration_support_gate", "selected": {}},
    }


def test_support_gated_audit_aggregates_three_disjoint_panels(tmp_path: Path) -> None:
    paths = []
    for start in (100, 200, 300):
        path = tmp_path / str(start) / "report.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_report(list(range(start, start + 32)))), encoding="utf-8")
        paths.append(path)
    result = audit(paths)
    assert result["status"] == "completed_root_disjoint_support_gated_audit"
    assert result["comparisons"]["action_only"]["gated_closed_loop_distance_gain"]["estimate"] == 2.0
    assert result["comparisons"]["effect_adapter"]["gated_ranking_regret_gain"]["estimate"] == 1.0
