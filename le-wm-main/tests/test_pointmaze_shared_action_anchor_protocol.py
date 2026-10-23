import hashlib
import json

import numpy as np
import pytest

from experiments.gdd_wm.audit_d1d2_repairability_routing import audit
from experiments.gdd_wm.build_pointmaze_frozen_predictive_teacher_package import build
from experiments.gdd_wm.build_pointmaze_shared_action_anchor_protocol import build_protocol


def _sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_source_bank(tmp_path, *, candidates=15):
    source = tmp_path / "source"
    source.mkdir()
    records, arrays = [], {}
    roots = (
        ("train", "train_a"),
        ("train", "train_b"),
        ("train", "train_c"),
        ("calibration", "cal_a"),
        ("outcome", "out_a"),
        ("outcome", "out_b"),
    )
    for root_index, (split, root) in enumerate(roots):
        for candidate in range(candidates):
            key = f"predicted::{root}/{candidate:03d}"
            # A nonlinear target prevents a ridge predictor from tying every
            # shared anchor exactly; the D1 stability audit can then verify a
            # nonzero, context-invariant preference rather than a vacuous tie.
            arrays[key] = np.asarray([root_index, candidate * candidate, candidate * 0.25], dtype=np.float64)
            records.append(
                {
                    "instance_id": root,
                    "split": split,
                    "environment": "PointMaze",
                    "candidate_id": candidate,
                    "action_sequence": [[float(candidate + 1), 0.25], [float(candidate + 1), 0.25]],
                    "source_latent_model": [float(root_index), -0.25],
                    "predicted_terminal_latent_ref": f"arrays.npz::{key}",
                    "predicted_latent_ref": f"arrays.npz::{key}",
                    "predicted_horizon": 2,
                    "real_task_cost": float(100 - candidate),
                }
            )
    dataset = source / "candidate_dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "metadata": {
                    "environment": "PointMaze",
                    "checkpoint": "frozen-checkpoint.pt",
                    "world_model_sha256": _sha256("frozen-checkpoint"),
                    "candidate_family": "source-world-bank-v1",
                    "candidate_bank_coordinate_frame": "world_fixed",
                    "predicted_horizon": 2,
                    "split_assignment": "hash_random_v1",
                },
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    np.savez(source / "arrays.npz", **arrays)
    return dataset


def test_builds_three_root_disjoint_contexts_with_exact_shared_anchors_and_provenance(tmp_path):
    source = _write_source_bank(tmp_path)

    report = build_protocol(source, tmp_path / "protocol")

    assert report["status"] == "built_valid_shared_action_anchor_protocol"
    assert report["root_split_counts"] == {"calibration": 1, "outcome": 2, "train": 3}
    assert report["protocol_sha256"]
    assert len(report["context_datasets"]) == 3
    for context in report["contexts"]:
        assert context["candidate_dataset_sha256"]
        assert context["arrays_sha256"]

    datasets = [tmp_path / "protocol" / entry["relative_candidate_dataset"] for entry in report["contexts"]]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in datasets]
    assert {payload["metadata"]["world_model_sha256"] for payload in payloads} == {_sha256("frozen-checkpoint")}
    assert {payload["metadata"]["candidate_family"] for payload in payloads} == {"gdd_shared_action_anchor_v1"}
    for root in ("train_a", "cal_a", "out_a"):
        rows_by_context = [
            [row for row in payload["records"] if row["instance_id"] == root]
            for payload in payloads
        ]
        anchors = [
            {row["action_sequence_sha256"] for row in rows if row["selection_role"] == "shared_anchor"}
            for rows in rows_by_context
        ]
        assert len(anchors[0]) == 2
        assert anchors[0] == anchors[1] == anchors[2]
        contextual = [
            {row["action_sequence_sha256"] for row in rows if row["selection_role"] == "context_specific"}
            for rows in rows_by_context
        ]
        assert all(len(item) == 1 for item in contextual)
        assert len(set.union(*contextual)) == 3


def test_protocol_unblocks_existing_predictive_builder_context_and_held_sequence_contracts(tmp_path):
    source = _write_source_bank(tmp_path)
    report = build_protocol(source, tmp_path / "protocol")
    inputs = [tmp_path / "protocol" / entry["relative_candidate_dataset"] for entry in report["contexts"]]

    build(inputs, tmp_path / "teacher")
    strict = audit(tmp_path / "teacher" / "candidate_dataset.json")

    assert strict["teacher_control"]["passed"] is True
    assert strict["repeated_context_stability"]["passed"] is True
    assert strict["held_candidate_separation"]["passed"] is True
    assert strict["status"] == "rejected_fail_closed"
    assert strict["rejection_reasons"] == ["finite_intervention_repairability"]


def test_rejects_source_bank_without_enough_disjoint_actions_for_all_splits(tmp_path):
    source = _write_source_bank(tmp_path, candidates=14)

    with pytest.raises(ValueError, match="at least 15 distinct action sequences"):
        build_protocol(source, tmp_path / "protocol")
