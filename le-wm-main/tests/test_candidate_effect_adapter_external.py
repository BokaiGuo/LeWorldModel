from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.decision_benchmark import evaluate_candidate_effect_adapter_external as external


def _write_bank(
    root: Path,
    root_splits: list[tuple[str, str]],
    *,
    outcome_offset: float = 0.0,
) -> tuple[Path, Path]:
    records = []
    arrays = {}
    for root_index, (instance_id, split) in enumerate(root_splits):
        source = np.asarray([0.1 * root_index, 0.0, 0.0, 0.0])
        for candidate in range(8):
            sequence = np.zeros((20, 2), dtype=np.float64)
            sequence[:10, 0] = 0.05 * candidate
            sequence[10:, 1] = 0.025 * candidate
            terminal = source.copy()
            terminal[:2] += np.asarray([0.01 * candidate, -0.005 * candidate])
            key = f"realized_state::{instance_id}/{candidate:03d}"
            arrays[key] = terminal
            cost = float(np.linalg.norm(terminal[:2] - np.asarray([1.0, 0.0])))
            records.append({
                "instance_id": instance_id,
                "split": split,
                "candidate_id": candidate,
                "action_sequence": sequence.tolist(),
                "source_latent_model": [float(root_index), 0.25],
                "source_state_raw": source.tolist(),
                "goal_raw": [1.0, 0.0],
                "real_task_cost": cost + outcome_offset,
                "realized_terminal_state_ref": f"arrays.npz::{key}",
            })
    dataset = root / "candidate_dataset.json"
    payload = root / "arrays.npz"
    root.mkdir(parents=True, exist_ok=True)
    dataset.write_text(json.dumps({"records": records}), encoding="utf-8")
    np.savez(payload, **arrays)
    return dataset, payload


def test_external_evaluation_fits_only_development_train_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development_dataset, development_arrays = _write_bank(
        tmp_path / "development",
        [(f"train_{index}", "train") for index in range(3)]
        + [(f"calibration_{index}", "calibration") for index in range(2)],
    )
    outcome_dataset, outcome_arrays = _write_bank(
        tmp_path / "external",
        [(f"external_{index}", "external_outcome") for index in range(2)],
        outcome_offset=1000.0,
    )
    calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def fake_fit_predict(train_x, train_y, eval_x, seed, epochs, device):
        calls.append((train_x.copy(), train_y.copy(), eval_x.copy()))
        return np.zeros((len(eval_x), 2), dtype=np.float32), {
            "epochs": int(epochs), "device": str(device), "seed": int(seed)
        }

    monkeypatch.setattr(external, "_fit_predict", fake_fit_predict)
    report = external.run(
        development_dataset,
        development_arrays,
        outcome_dataset,
        outcome_arrays,
        tmp_path / "report.json",
        epochs=1,
        bootstrap_samples=10,
        device="cuda",
        alpha=0.5,
    )

    assert len(calls) == 10
    assert all(len(train_y) in {24, 18} for _, train_y, _ in calls)
    assert all(train_x.shape[1] == 18 for train_x, _, _ in calls)
    assert all(np.max(np.abs(train_y)) < 1.0 for _, train_y, _ in calls)
    assert report["protocol"]["outcome_used_for_training_calibration_or_selection"] is False
    assert report["summary"]["external_root_count"] == 2


def test_external_protocol_rejects_split_root_or_bank_mismatch(tmp_path: Path) -> None:
    development_dataset, _ = _write_bank(
        tmp_path / "development",
        [(f"train_{index}", "train") for index in range(3)]
        + [("calibration", "calibration")],
    )
    development = json.loads(development_dataset.read_text(encoding="utf-8"))["records"]
    external_records = [dict(row, split="external_outcome") for row in development[:8]]

    with pytest.raises(ValueError, match="overlap"):
        external._validate_fixed_bank_protocol(development, external_records)

    external_records = [dict(row, instance_id="external") for row in external_records]
    external_records[0]["split"] = "outcome"
    with pytest.raises(ValueError, match="external_outcome"):
        external._validate_fixed_bank_protocol(development, external_records)

    external_records[0]["split"] = "external_outcome"
    external_records[0]["action_sequence"][0][0] = 0.9
    with pytest.raises(ValueError, match="shared world-action bank"):
        external._validate_fixed_bank_protocol(development, external_records)
