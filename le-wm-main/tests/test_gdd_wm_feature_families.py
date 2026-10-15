import json

import numpy as np

from experiments.gdd_wm.audit_transfer_sufficient_feature_families import audit


def test_feature_family_audit_is_train_only(tmp_path):
    rows = []
    arrays = {}
    for split, root, offset in [("train", "r0", 0.0), ("outcome", "r1", 0.1)]:
        for candidate in range(2):
            key = f"predicted::{root}/{candidate:03d}"
            arrays[key] = np.array([float(candidate), 1.0], dtype=np.float32)
            rows.append({
                "instance_id": root,
                "candidate_id": candidate,
                "split": split,
                "predicted_latent_ref": f"arrays.npz::{key}",
                "source_latent_model": [0.0, 0.0],
                "source_state_raw": [0.0, 0.0, 0.0, 0.0],
                "goal_raw": [1.0, 0.0],
                "action_sequence": [[float(candidate), 0.0], [float(candidate), 0.0]],
                "real_task_cost": float(offset + (0.2 if candidate == 0 else 0.8)),
            })
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"metadata": {}, "records": rows}), encoding="utf-8")
    arrays_path = tmp_path / "arrays.npz"
    np.savez(arrays_path, **arrays)
    report = audit(dataset, arrays_path)
    assert report["status"] == "completed_diagnostic"
    assert report["protocol"]["outcome_used_for_fit_or_feature_selection"] is False
    assert set(report["feature_families"]) == {
        "predicted_terminal",
        "source_plus_terminal",
        "source_plus_delta",
        "source_delta_action",
        "source_delta_estimated_goalframe_action",
        "action_summary",
    }
    assert report["decision"]["family_selection"] == "not_performed"
