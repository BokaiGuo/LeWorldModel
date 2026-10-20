import json

import numpy as np

from experiments.gdd_wm.build_frozen_predictive_teacher_package import build


def _write_package(tmp_path, *, history_windows, predictor_history_windows):
    rows = []
    arrays = {}
    for split, root in (("train", "r0"), ("train", "r1"), ("outcome", "r2")):
        for candidate, value in (("a", 0.0), ("b", 1.0)):
            key = f"predicted::{root}/{candidate}"
            arrays[key] = np.asarray([value, 0.0], dtype=np.float32)
            rows.append({
                "instance_id": root,
                "candidate_id": candidate,
                "candidate_bank_context_id": "ctx0",
                "teacher_set_id": root,
                "split": split,
                "predicted_terminal_latent_ref": f"arrays.npz::{key}",
                "real_task_cost": 0.2 if candidate == "a" else 0.8,
                "goal_raw": [0.0, 0.0],
            })
    dataset = tmp_path / "candidate_dataset.json"
    dataset.write_text(json.dumps({
        "metadata": {
            "history_windows": history_windows,
            "predictor_history_windows": predictor_history_windows,
        },
        "records": rows,
    }))
    np.savez(tmp_path / "arrays.npz", **arrays)
    return dataset


def test_teacher_records_predictor_history_mismatch(tmp_path):
    dataset = _write_package(tmp_path, history_windows=2, predictor_history_windows=1)
    report = build(dataset, tmp_path / "out")
    assert report["summary"]["history_compatible"] is False
    payload = json.loads((tmp_path / "out" / "candidate_dataset.json").read_text())
    assert payload["metadata"]["teacher_history_contract"] == (
        "predicted_terminal_latent_uses_predictor_history_not_candidate_history"
    )
    assert (tmp_path / "out" / "arrays.npz").exists()


def test_teacher_accepts_matching_history_contract(tmp_path):
    dataset = _write_package(tmp_path, history_windows=1, predictor_history_windows=1)
    report = build(dataset, tmp_path / "out")
    assert report["summary"]["history_compatible"] is True
