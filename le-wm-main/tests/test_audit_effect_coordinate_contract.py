import json

import numpy as np

from experiments.gdd_wm.audit_effect_coordinate_contract import audit


def test_coordinate_audit_separates_inertia_from_action_effect(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    np.savez_compressed(
        source_dir / "arrays.npz",
        **{
            "realized_state::r0/0": np.asarray([0.8, 0.0, 0.0, 0.0]),
            "realized_state::r0/1": np.asarray([0.0, 0.0, 0.0, 0.0]),
        },
    )
    source_dataset = source_dir / "candidate_dataset.json"
    source_dataset.write_text(json.dumps({"records": []}), encoding="utf-8")
    wrapper = tmp_path / "wrapper"
    wrapper.mkdir()
    wrapper_dataset = wrapper / "candidate_dataset.json"
    rows = [
        {
            "instance_id": "r0", "candidate_id": "a", "source_candidate_id": "0",
            "source_candidate_dataset": str(source_dataset),
            "realized_terminal_state_ref": "arrays.npz::realized_state::r0/0",
            "source_state_raw": [0.0, 0.0, 0.0, 0.0], "goal_raw": [1.0, 0.0],
            "action_sequence": [[1.0, 0.0]] * 20, "split": "calibration",
        },
        {
            "instance_id": "r0", "candidate_id": "b", "source_candidate_id": "1",
            "source_candidate_dataset": str(source_dataset),
            "realized_terminal_state_ref": "arrays.npz::realized_state::r0/1",
            "source_state_raw": [0.0, 0.0, 0.0, 0.0], "goal_raw": [1.0, 0.0],
            "action_sequence": [[0.0, 0.0]] * 20, "split": "calibration",
        },
    ]
    wrapper_dataset.write_text(json.dumps({"metadata": {"candidate_bank_coordinate_frame": "goal_relative"}, "records": rows}), encoding="utf-8")
    report = audit(wrapper_dataset, tmp_path / "audit.json")
    summary = report["split_summary"]["calibration"]["baseline_corrected_action_effect"]
    assert summary["mean"] == 1.0
    assert report["interpretation"]["coordinate_mismatch_evidence"] is False
