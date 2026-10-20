import hashlib
import json

import numpy as np
import pytest

from experiments.gdd_wm.train_projection_distillation import _fit_projection, run


def _write_dataset(path, d1_status="rejected_fail_closed"):
    rows = []
    for split in ("train", "outcome"):
        for candidate in ("a", "b"):
            rows.append({
                "instance_id": f"root_{split}",
                "teacher_set_id": f"set_{split}",
                "candidate_id": candidate,
                "teacher_score": 1.0 if candidate == "a" else 0.0,
                "real_task_cost": 0.0 if candidate == "a" else 1.0,
                "split": split,
            })
    path.write_text(json.dumps({"records": rows}))
    return {"records": rows}


def test_d2_does_not_train_when_d1_is_rejected(tmp_path):
    dataset = tmp_path / "dataset.json"
    _write_dataset(dataset)
    d1 = tmp_path / "d1.json"
    d1.write_text(json.dumps({"schema": "d1", "status": "rejected_fail_closed"}))
    output = tmp_path / "out"
    report = run(dataset_path=dataset, d1_path=d1, output_path=output)
    assert report["status"] == "blocked_d1_gate"
    assert not (output / "projection.npy").exists()
    saved = json.loads((output / "d2_projection_distillation.json").read_text())
    assert saved["gates"]["d1"]["passed"] is False


def test_d2_requires_independent_prediction_and_planning_gates(tmp_path):
    dataset = tmp_path / "dataset.json"
    payload = _write_dataset(dataset)
    d1 = tmp_path / "d1.json"
    d1.write_text(json.dumps({"schema": "d1", "status": "passed"}))
    repairability = tmp_path / "repairability.json"
    repairability.write_text(json.dumps({"status": "permitted_predicted_dynamics"}))
    table = tmp_path / "features.npz"
    keys = np.asarray([
        f"{row['instance_id']}::{row['teacher_set_id']}::{row['candidate_id']}"
        for row in payload["records"]
    ])
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    metadata = {
        "schema": "gdd_wm_student_feature_table.v0.1",
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "key_schema": "instance_id::teacher_set_id::candidate_id",
        "candidate_set_context_input_forbidden": True,
        "teacher_rank_input_forbidden": True,
        "held_out_outcome_input_forbidden": True,
        "dynamics_frozen": True,
    }
    np.savez(
        table,
        keys=keys,
        features=features,
        goal_features=np.zeros_like(features),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    report = run(
        dataset_path=dataset, d1_path=d1, output_path=tmp_path / "out",
        feature_table_path=table, repairability_audit_path=repairability,
    )
    assert report["status"] == "completed_diagnostic"
    assert report["permission"]["status"] == "rejected_missing_independent_gate"
    assert report["metrics"]["held_out"]["outcome"]["fitted_projection"]["comparable_pairs"] > 0


def test_d2_blocks_when_repairability_gate_falls_back(tmp_path):
    dataset = tmp_path / "dataset.json"
    _write_dataset(dataset)
    d1 = tmp_path / "d1.json"
    d1.write_text(json.dumps({"schema": "d1", "status": "passed"}))
    repairability = tmp_path / "repairability.json"
    repairability.write_text(json.dumps({"status": "fallback_action_only"}))
    output = tmp_path / "out"
    report = run(
        dataset_path=dataset, d1_path=d1, output_path=output,
        repairability_audit_path=repairability,
    )
    assert report["status"] == "blocked_repairability_gate"
    assert report["gates"]["repairability"]["passed"] is False


def test_d2_accepts_explicitly_permitted_nested_repairability_panel(tmp_path):
    dataset = tmp_path / "dataset.json"
    payload = _write_dataset(dataset)
    d1 = tmp_path / "d1.json"
    d1.write_text(json.dumps({"schema": "d1", "status": "passed"}))
    repairability = tmp_path / "repairability_panel.json"
    repairability.write_text(json.dumps({
        "schema": "gdd_wm_repairability_gate_panel.v0.1",
        "audits": [{"status": "permitted_predicted_dynamics"}, {"status": "fallback_action_only"}],
    }))
    table = tmp_path / "features.npz"
    keys = np.asarray([
        f"{row['instance_id']}::{row['teacher_set_id']}::{row['candidate_id']}"
        for row in payload["records"]
    ])
    metadata = {
        "schema": "gdd_wm_student_feature_table.v0.1",
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "key_schema": "instance_id::teacher_set_id::candidate_id",
        "candidate_set_context_input_forbidden": True,
        "teacher_rank_input_forbidden": True,
        "held_out_outcome_input_forbidden": True,
        "dynamics_frozen": True,
    }
    np.savez(table, keys=keys, features=np.ones((len(keys), 2)),
             goal_features=np.zeros((len(keys), 2)),
             metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    report = run(
        dataset_path=dataset, d1_path=d1, output_path=tmp_path / "out",
        feature_table_path=table, repairability_audit_path=repairability,
    )
    assert report["gates"]["repairability"]["passed"] is True
    assert report["status"] == "completed_diagnostic"


def test_d2_rejects_feature_table_with_wrong_dataset_provenance(tmp_path):
    dataset = tmp_path / "dataset.json"
    payload = _write_dataset(dataset)
    d1 = tmp_path / "d1.json"
    d1.write_text(json.dumps({"schema": "d1", "status": "passed"}))
    repairability = tmp_path / "repairability.json"
    repairability.write_text(json.dumps({"status": "permitted_predicted_dynamics"}))
    table = tmp_path / "features.npz"
    keys = np.asarray([
        f"{row['instance_id']}::{row['teacher_set_id']}::{row['candidate_id']}"
        for row in payload["records"]
    ])
    metadata = {
        "schema": "gdd_wm_student_feature_table.v0.1",
        "dataset_sha256": "wrong",
        "key_schema": "instance_id::teacher_set_id::candidate_id",
        "candidate_set_context_input_forbidden": True,
        "teacher_rank_input_forbidden": True,
        "held_out_outcome_input_forbidden": True,
        "dynamics_frozen": True,
    }
    np.savez(
        table,
        keys=keys,
        features=np.ones((len(keys), 2)),
        goal_features=np.zeros((len(keys), 2)),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="provenance.dataset_sha256"):
        run(
            dataset_path=dataset, d1_path=d1, output_path=tmp_path / "out",
            feature_table_path=table, repairability_audit_path=repairability,
        )


def test_projection_rank_cannot_exceed_train_sample_count():
    features = np.eye(2)
    goals = np.zeros_like(features)
    rows = [
        {"instance_id": "root", "teacher_set_id": "set", "teacher_score": 1.0},
        {"instance_id": "root", "teacher_set_id": "set", "teacher_score": 0.0},
    ]
    with pytest.raises(ValueError, match="train sample count"):
        _fit_projection(
            features, goals, rows, np.asarray([0]), rank=2, epochs=1,
            learning_rate=0.05, margin=1e-3, temperature=0.1, ridge=1e-4,
        )
