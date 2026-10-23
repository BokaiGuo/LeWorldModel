from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.decision_benchmark import evaluate_candidate_effect_adapter_external as external
from experiments.decision_benchmark.audit_effect_adapter_reproducibility import materialize
from experiments.decision_benchmark.audit_effect_adapter_result import audit as audit_result


def _rows(root: str, split: str) -> list[dict]:
    output = []
    for candidate in range(2):
        output.append({
            "instance_id": root,
            "split": split,
            "candidate_id": candidate,
            "action_sequence": np.full((2, 2), candidate + 0.1).tolist(),
        })
    return output


def test_rejects_development_root_reused_across_train_and_calibration() -> None:
    development = _rows("shared_root", "train") + _rows("shared_root", "calibration")
    outcome = _rows("external_root", "external_outcome")

    with pytest.raises(ValueError, match="multiple development splits"):
        external._validate_fixed_bank_protocol(development, outcome)


def test_rejects_development_outcome_rows_in_external_reproduction() -> None:
    development = _rows("train_root", "train") + _rows("legacy_outcome", "outcome")
    outcome = _rows("external_root", "external_outcome")

    with pytest.raises(ValueError, match="only train and calibration"):
        external._validate_fixed_bank_protocol(development, outcome)


def test_materialize_binds_one_shared_world_action_bank_and_relabels_only_held_split(tmp_path) -> None:
    records = (
        _rows("train_root", "train")
        + _rows("calibration_root", "calibration")
        + _rows("held_root", "outcome")
    )
    dataset = tmp_path / "source.json"
    arrays = tmp_path / "arrays.npz"
    dataset.write_text(
        json.dumps({
            "metadata": {
                "shared_world_action_bank": True,
                "candidate_bank_coordinate_frame": "world_fixed",
            },
            "records": records,
        }),
        encoding="utf-8",
    )
    np.savez(arrays, unused=np.zeros(1))

    report = materialize(dataset, arrays, tmp_path / "materialized")
    held = json.loads(Path(report["held_root_dataset"]).read_text())

    assert report["root_counts"] == {"train": 1, "calibration": 1, "outcome": 1}
    assert len(report["candidate_action_sha256"]) == 2
    assert {row["split"] for row in held["records"]} == {"external_outcome"}
    assert {row["source_split"] for row in held["records"]} == {"outcome"}


def test_result_audit_requires_all_three_matched_controls_on_every_held_root(tmp_path) -> None:
    input_audit = tmp_path / "input.json"
    replication = tmp_path / "replication.json"
    input_audit.write_text(json.dumps({
        "status": "materialized_valid_held_root_replication_inputs",
        "replication_semantics": "held-root replication; not newly collected external data",
        "root_counts": {"outcome": 2},
    }), encoding="utf-8")
    replication.write_text(json.dumps({
        "protocol": {
            "outcome_used_for_training_calibration_or_selection": False,
            "feature_mode": "frozen_latent_action",
        },
        "controls": {"paired_regret_gain": {
            name: {"groups": 2, "lower": 0.01, "estimate": 0.02, "upper": 0.03}
            for name in ("bank_only", "shuffled_prediction", "no_dynamics")
        }},
    }), encoding="utf-8")

    report = audit_result(input_audit, replication, tmp_path / "audit.json")

    assert report["status"] == "reproduced_bounded_positive_control"
