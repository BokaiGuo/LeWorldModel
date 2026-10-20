import json

import numpy as np

from experiments.gdd_wm.audit_root_local_effect_ranker import audit


def test_root_local_audit_uses_action_coverage_without_outcome_probe_selection(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    arrays = {
        f"realized_state::r0/{index}": np.asarray([float(index), 0.0, 0.0, 0.0])
        for index in range(4)
    }
    np.savez_compressed(source / "arrays.npz", **arrays)
    source_dataset = source / "candidate_dataset.json"
    source_dataset.write_text(json.dumps({"records": []}), encoding="utf-8")
    rows = []
    for index in range(4):
        rows.append({
            "instance_id": "r0",
            "candidate_id": str(index),
            "source_candidate_dataset": str(source_dataset),
            "realized_terminal_state_ref": f"arrays.npz::realized_state::r0/{index}",
            "source_state_raw": [0.0, 0.0, 0.0, 0.0],
            "goal_raw": [3.0, 0.0],
            "action_sequence": [[float(index), 0.0]] * 20,
            "real_task_cost": float(abs(3 - index)),
            "teacher_score": -float(abs(3 - index)),
            "split": "calibration",
        })
    dataset = tmp_path / "candidate_dataset.json"
    dataset.write_text(json.dumps({"records": rows}), encoding="utf-8")
    report = audit(dataset, tmp_path / "audit.json", probe_count=2)
    assert report["summary"]["root_count"] == 1
    assert len(report["roots"][0]["probe_candidate_ids"]) == 2
    assert report["protocol"]["outcome_labels_used_for_probe_selection"] is False
