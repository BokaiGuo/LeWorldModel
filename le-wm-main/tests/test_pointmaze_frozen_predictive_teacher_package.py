import json

import numpy as np
import pytest

from experiments.gdd_wm.build_pointmaze_frozen_predictive_teacher_package import build
from experiments.gdd_wm.audit_d1d2_repairability_routing import audit


def _write_context(tmp_path, name, *, terminal_offset=0.0, include_outcome_cost=True):
    path = tmp_path / name
    path.mkdir()
    records = []
    arrays = {}
    for split, root in (("train", "train_a"), ("train", "train_b"), ("calibration", "cal_a"), ("outcome", "out_a")):
        root_offset = {"train": 0.0, "calibration": 50.0, "outcome": 100.0}[split]
        for candidate in range(3):
            key = f"predicted::{root}/{candidate}"
            arrays[key] = np.asarray(
                [candidate + (terminal_offset if candidate == 2 else 0.0), candidate * 0.5], dtype=np.float64
            )
            record = {
                "instance_id": root,
                "split": split,
                "candidate_id": f"{name}_{candidate}",
                "teacher_set_id": f"{root}_{name}",
                "candidate_bank_context_id": name,
                "action_sequence": [[
                    float(root_offset + candidate + (terminal_offset if candidate == 2 else 0.0)), 0.25,
                ], [float(root_offset + candidate + (terminal_offset if candidate == 2 else 0.0)), 0.25]],
                "source_latent_model": [0.1, -0.2],
                "predicted_terminal_latent_ref": f"arrays.npz::{key}",
                "real_task_cost": float(20.0 - candidate),
            }
            if not include_outcome_cost:
                record.pop("real_task_cost")
            records.append(record)
    payload = {
        "metadata": {
            "environment": "PointMaze",
            "checkpoint": "frozen-checkpoint.pt",
            "candidate_family": "toy-shared-anchor",
            "candidate_bank_coordinate_frame": "world_fixed",
            "predicted_horizon": 2,
        },
        "records": records,
    }
    dataset = path / "candidate_dataset.json"
    dataset.write_text(json.dumps(payload), encoding="utf-8")
    np.savez(path / "arrays.npz", **arrays)
    return dataset


def test_builds_train_only_outcome_free_predictive_package_with_hashes_and_context_contract(tmp_path):
    inputs = [
        _write_context(tmp_path, "ctx0", terminal_offset=0.0),
        _write_context(tmp_path, "ctx1", terminal_offset=0.1),
        _write_context(tmp_path, "ctx2", terminal_offset=0.2),
    ]

    report = build(inputs, tmp_path / "out")

    package = json.loads((tmp_path / "out" / "candidate_dataset.json").read_text(encoding="utf-8"))
    metadata = package["metadata"]
    assert report["status"] == "built_valid_predictive_teacher_package"
    assert metadata["teacher_training_uses_execution_outcomes"] is False
    assert metadata["teacher_training_target_fields"] == ["predicted_terminal_latent"]
    assert metadata["teacher_input_fields"] == ["source_latent_model", "action_sequence_summary"]
    assert len(metadata["teacher_fit_root_ids"]) == 2
    assert len(metadata["context_manifest"]) == 3
    assert metadata["candidate_dataset_input_sha256"]
    assert all("action_sequence_sha256" in row and "root_sha256" in row for row in package["records"])
    assert {row["teacher_score_source"] for row in package["records"]} == {"train_only_predictive_latent_residual"}


def test_teacher_scores_are_invariant_to_execution_cost_values(tmp_path):
    inputs = [
        _write_context(tmp_path, "ctx0", terminal_offset=0.0),
        _write_context(tmp_path, "ctx1", terminal_offset=0.1),
        _write_context(tmp_path, "ctx2", terminal_offset=0.2),
    ]
    first = build(inputs, tmp_path / "first")
    for path in inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["records"]:
            row["real_task_cost"] += 1000.0
        path.write_text(json.dumps(payload), encoding="utf-8")
    second = build(inputs, tmp_path / "second")

    first_rows = json.loads((tmp_path / "first" / "candidate_dataset.json").read_text())["records"]
    second_rows = json.loads((tmp_path / "second" / "candidate_dataset.json").read_text())["records"]
    assert first["summary"]["teacher_score_sha256"] == second["summary"]["teacher_score_sha256"]
    assert [row["teacher_score"] for row in first_rows] == [row["teacher_score"] for row in second_rows]


def test_rejects_root_split_or_candidate_context_mixing(tmp_path):
    inputs = [
        _write_context(tmp_path, "ctx0"),
        _write_context(tmp_path, "ctx1"),
        _write_context(tmp_path, "ctx2"),
    ]
    payload = json.loads(inputs[2].read_text(encoding="utf-8"))
    for row in payload["records"]:
        if row["instance_id"] == "out_a":
            row["split"] = "train"
    inputs[2].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="root split mismatch"):
        build(inputs, tmp_path / "out")


def test_rejects_candidate_contexts_from_different_frozen_world_models(tmp_path):
    inputs = [
        _write_context(tmp_path, "ctx0", terminal_offset=0.0),
        _write_context(tmp_path, "ctx1", terminal_offset=0.1),
        _write_context(tmp_path, "ctx2", terminal_offset=0.2),
    ]
    payload = json.loads(inputs[2].read_text(encoding="utf-8"))
    payload["metadata"]["checkpoint"] = "different-frozen-checkpoint.pt"
    inputs[2].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="same frozen world model"):
        build(inputs, tmp_path / "out")


def test_package_satisfies_the_teacher_context_and_held_sequence_d1_contract_before_probe_evidence(tmp_path):
    inputs = [
        _write_context(tmp_path, "ctx0", terminal_offset=0.0),
        _write_context(tmp_path, "ctx1", terminal_offset=0.1),
        _write_context(tmp_path, "ctx2", terminal_offset=0.2),
    ]
    build(inputs, tmp_path / "out")

    report = audit(tmp_path / "out" / "candidate_dataset.json")

    assert report["teacher_control"]["passed"] is True
    assert report["repeated_context_stability"]["passed"] is True
    assert report["held_candidate_separation"]["passed"] is True
    assert report["rejection_reasons"] == ["finite_intervention_repairability"]


def test_requires_three_genuine_contexts(tmp_path):
    inputs = [_write_context(tmp_path, "ctx0"), _write_context(tmp_path, "ctx1")]

    with pytest.raises(ValueError, match="at least 3 independent candidate contexts"):
        build(inputs, tmp_path / "out")
