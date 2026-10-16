import json

import numpy as np

from experiments.gdd_wm.build_tworoom_indexed_canonical_readout import build


def test_indexed_readout_serializes_calibration_safe_gate(tmp_path) -> None:
    records = []
    arrays = {}
    for split, roots in (("train", range(3)), ("calibration", range(3, 5)), ("outcome", range(5, 7))):
        for root in roots:
            for candidate in ("candidate_000", "candidate_001"):
                key = f"root_{root}/{candidate}"
                source = np.asarray([float(root), 0.0, 1.0, -1.0], dtype=np.float32)
                action_value = 0.0 if candidate.endswith("000") else 1.0
                action = [[action_value, 0.0]] * 4
                arrays[key] = source + np.asarray([action_value, 0.0, 0.0, 0.0], dtype=np.float32)
                records.append({
                    "instance_id": f"tworoom_episode_{root}",
                    "split": split,
                    "candidate_id": candidate,
                    "action_sequence": action,
                    "predicted_latent_ref": f"arrays.npz::{key}",
                    "source_latent_model": source.tolist(),
                    "real_task_cost": float(root + action_value),
                })
    dataset = tmp_path / "candidate_dataset.json"
    dataset.write_text(json.dumps({"metadata": {"candidate_bank_hash": "test"}, "records": records}))
    array_path = tmp_path / "arrays.npz"
    np.savez(array_path, **arrays)

    report, artifact = build(dataset, array_path)

    assert artifact["gate_enabled"].tolist() == [0]
    assert report["selection"]["fallback"] is True
    assert report["selection"]["outcome_used_for_selection"] is False
