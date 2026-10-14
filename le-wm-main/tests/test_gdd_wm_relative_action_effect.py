import json

import numpy as np

from experiments.gdd_wm.audit_relative_action_effect_geometry import audit


def test_relative_effect_audit_has_fixed_variants_and_train_only_protocol(tmp_path):
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
    assert report["protocol"]["outcome_used_for_fit_or_variant_selection"] is False
    assert set(report["variants"]) == {
        "absolute_predicted_terminal",
        "root_centered_predicted_terminal",
        "root_centered_source_delta",
        "root_centered_source_delta_action",
        "root_centered_source_delta_action_interaction",
        "root_centered_source_delta_action_interaction_lr2",
        "root_centered_source_delta_action_interaction_lr4",
        "root_centered_source_delta_action_interaction_lr8",
        "root_centered_source_delta_action_interaction_lr16",
        "root_centered_source_delta_action_pairwise",
        "root_centered_source_delta_action_interaction_lr4_pairwise",
        "root_centered_source_delta_action_interaction_lr8_pairwise",
        "root_centered_source_delta_goalframe_action",
        "root_centered_source_delta_goalframe_action_pairwise",
    }
    assert report["protocol"]["low_rank_basis_fit_split"] == "train"
    assert report["variants"]["root_centered_source_delta_action_interaction_lr4"][
        "feature_dimension"
    ] == report["variants"]["root_centered_source_delta_action"]["feature_dimension"] + 4
    assert report["variants"]["root_centered_source_delta_action_pairwise"][
        "target"
    ] == "train_pairwise_cost_difference"
    assert report["decision"]["variant_selection"] == "not_performed"
