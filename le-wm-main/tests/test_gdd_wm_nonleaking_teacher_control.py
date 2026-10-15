import json

import numpy as np

from experiments.gdd_wm.audit_nonleaking_teacher_control import audit


def test_control_fits_on_train_only_and_reports_heldout(tmp_path):
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
                "real_task_cost": float(offset + (0.2 if candidate == 0 else 0.8)),
            })
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"metadata": {"source": "test"}, "records": rows}), encoding="utf-8")
    arrays_path = tmp_path / "arrays.npz"
    np.savez(arrays_path, **arrays)
    report = audit(dataset, arrays_path)
    assert report["status"] == "completed_diagnostic"
    assert report["protocol"]["outcome_used_for_fit_or_selection"] is False
    assert report["execution_consistency"]["outcome"]["pairs"]["execution_consistency"] == 1.0
    assert report["decision"]["d1_ready"] is False


def test_control_rejects_train_outcome_root_overlap(tmp_path):
    rows = []
    arrays = {}
    for split in ("train", "outcome"):
        key = f"predicted::same/{len(rows):03d}"
        arrays[key] = np.array([1.0, 2.0], dtype=np.float32)
        rows.append({
            "instance_id": "same",
            "candidate_id": len(rows),
            "split": split,
            "predicted_latent_ref": f"arrays.npz::{key}",
            "real_task_cost": 0.5,
        })
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"metadata": {}, "records": rows}), encoding="utf-8")
    arrays_path = tmp_path / "arrays.npz"
    np.savez(arrays_path, **arrays)
    try:
        audit(dataset, arrays_path)
    except ValueError as exc:
        assert "roots overlap" in str(exc)
    else:
        raise AssertionError("expected root-disjointness failure")
