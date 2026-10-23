from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.decision_benchmark import audit_pointmaze_action_ood_observability as audit


def _rows() -> list[dict[str, object]]:
    rows = []
    for root, split, latent, state in (
        ("train-a", "train", [0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
        ("train-b", "train", [0.01, 0.0], [1.0, 0.0, 0.0, 0.0]),
        ("cal", "calibration", [0.005, 0.0], [0.5, 0.0, 0.0, 0.0]),
    ):
        for candidate, action in enumerate(([-0.2, 0.0], [0.2, 0.0])):
            sequence = np.repeat(np.asarray(action, dtype=np.float64)[None, :], 4, axis=0)
            rows.append({
                "instance_id": root, "split": split, "candidate_id": candidate,
                "source_latent_model": latent, "source_history_model": latent,
                "source_state_raw": state, "goal_raw": [1.0, 0.0],
                "action_sequence": sequence.tolist(),
                "realized_terminal_state_ref": f"arrays.npz::{root}/{candidate}",
            })
    return rows


def test_identity_history_is_not_counted_as_temporal_feature() -> None:
    status = audit.temporal_history_status(_rows(), {"history_windows": 1})
    assert status["status"] == "unavailable_identity_history"


def test_alias_conflict_uses_same_action_and_train_only_thresholds(tmp_path: Path) -> None:
    rows = _rows()
    arrays = {}
    for row in rows:
        root, candidate = str(row["instance_id"]), int(row["candidate_id"])
        source = np.asarray(row["source_state_raw"], dtype=np.float64)
        # Near latents but train-a and train-b respond differently to the same action.
        direction = -1.0 if root == "train-a" else 1.0
        arrays[f"{root}/{candidate}"] = source + np.asarray([direction * (candidate + 1), 0.0, 0.0, 0.0])
    outcome = {}
    unseen = []
    for row in rows:
        copied = {key: value for key, value in row.items() if key != "realized_terminal_state_ref"}
        copied["outcome_label_status"] = "sealed_until_postfit_join"
        unseen.append(copied)
        key = f"{copied['instance_id']}/{copied['candidate_id']}"
        outcome[key] = {
            "real_task_cost": 0.0,
            "realized_terminal_state": arrays[f"{copied['instance_id']}/{copied['candidate_id']}"].tolist(),
        }
    outcome_path = tmp_path / "outcomes.json"
    outcome_path.write_text(json.dumps(outcome))
    report = audit.alias_conflict_diagnostic(rows, arrays, unseen, outcome_path)
    assert report["definition"].startswith("same shared action")
    assert report["by_split"]["calibration"]["root_count"] == 1
    assert all(row["same_action_hashes"] for row in report["rows"])


def test_feature_families_have_expected_context_widths() -> None:
    row = _rows()[0]
    assert audit._context(row, "action_only").shape == (8,)
    assert audit._context(row, "frozen_source_latent").shape == (10,)
    assert audit._context(row, "physical_state_oracle").shape == (12,)
