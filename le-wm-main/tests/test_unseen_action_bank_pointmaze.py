from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.decision_benchmark import run_unseen_action_bank_pointmaze as unseen


def _write_source_bank(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    records, arrays = [], {}
    source_rows = [
        ("train_root", "train", [0.0, 0.0, 0.0, 0.0]),
        ("calibration_root", "calibration", [0.1, 0.0, 0.0, 0.0]),
        ("outcome_root", "outcome", [0.2, 0.0, 0.0, 0.0]),
    ]
    for root_index, (instance_id, split, state) in enumerate(source_rows):
        for candidate, value in enumerate((0.1, 0.4)):
            action = np.full((20, 2), [value, 0.0], dtype=np.float64)
            key = f"realized::{instance_id}/{candidate}"
            arrays[key] = np.asarray(state) + np.asarray([value, 0.0, 0.0, 0.0])
            records.append({
                "instance_id": instance_id,
                "split": split,
                "candidate_id": candidate,
                "action_sequence": action.tolist(),
                "source_state_raw": state,
                "source_latent_model": [float(root_index), 0.2],
                "goal_raw": [1.0, 0.0],
                "start_episode_id": str(100 + root_index),
                "source_frame": 5,
                "context_length": 2,
                "realized_terminal_state_ref": f"arrays.npz::{key}",
            })
    dataset = root / "candidate_dataset.json"
    array_path = root / "arrays.npz"
    dataset.write_text(json.dumps({
        "metadata": {"shared_world_action_bank": True, "candidate_bank_coordinate_frame": "world_fixed"},
        "records": records,
    }), encoding="utf-8")
    np.savez(array_path, **arrays)
    return dataset, array_path


def test_run_unseen_bank_seals_labels_until_postfit_and_uses_new_candidate_identity(
    tmp_path: Path, monkeypatch
) -> None:
    dataset, arrays = _write_source_bank(tmp_path / "source")
    monkeypatch.setattr(unseen, "require_cuda_allocation", lambda _device: {"status": "passed", "device": "cuda"})
    monkeypatch.setattr(
        unseen,
        "_simulate_sequence",
        lambda state, sequence: np.asarray(state, dtype=np.float64) + np.asarray([np.mean(sequence[:, 0]), np.mean(sequence[:, 1]), 0.0, 0.0]),
    )
    monkeypatch.setattr(
        unseen,
        "_ensemble_predict",
        lambda train_x, train_y, eval_x, **_kwargs: (np.zeros((len(eval_x), 2), dtype=np.float32), [{"seed": 0}]),
    )

    report = unseen.run(
        dataset, arrays, tmp_path / "result", epochs=1, bootstrap_samples=20,
        candidate_count=4, device="cuda",
    )

    contexts = json.loads((tmp_path / "result" / "unseen_action_bank_context.json").read_text())
    assert all("real_task_cost" not in row for row in contexts["records"])
    assert report["contract"]["candidate_id_overlap"] == []
    assert report["contract"]["action_hash_overlap"] == []
    assert report["unseen_bank"]["outcomes_opened_after_fit"] is True
    assert report["protocol"]["outcome_used_for_fit_or_model_selection"] is False
