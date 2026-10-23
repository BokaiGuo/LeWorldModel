from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.decision_benchmark import run_unseen_action_bank_ood_repair as repair
from experiments.decision_benchmark import run_unseen_action_bank_pointmaze as unseen


def _source_bank(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        ("train", "train", [0.0, 0.0, 0.0, 0.0]),
        ("calibration", "calibration", [0.1, 0.0, 0.0, 0.0]),
        ("outcome", "outcome", [0.2, 0.0, 0.0, 0.0]),
    ]
    records, arrays = [], {}
    for index, (root_id, split, state) in enumerate(rows):
        for candidate, magnitude in enumerate((0.1, 0.4)):
            action = np.full((20, 2), [magnitude, 0.0], dtype=np.float64)
            key = f"state::{root_id}/{candidate}"
            arrays[key] = np.asarray(state) + np.asarray([magnitude, 0.0, 0.0, 0.0])
            records.append({
                "instance_id": root_id,
                "split": split,
                "candidate_id": candidate,
                "action_sequence": action.tolist(),
                "source_state_raw": state,
                "source_latent_model": [float(index), 0.25],
                "goal_raw": [1.0, 0.0],
                "start_episode_id": str(100 + index),
                "source_frame": 5,
                "context_length": 2,
                "realized_terminal_state_ref": f"arrays.npz::{key}",
            })
    dataset = root / "bank.json"
    arrays_path = root / "arrays.npz"
    dataset.write_text(json.dumps({
        "metadata": {"shared_world_action_bank": True, "candidate_bank_coordinate_frame": "world_fixed"},
        "records": records,
    }), encoding="utf-8")
    np.savez(arrays_path, **arrays)
    return dataset, arrays_path


def _fake_simulator(state: np.ndarray, sequence: np.ndarray) -> np.ndarray:
    return np.asarray(state, dtype=np.float64) + np.asarray([
        np.mean(sequence[:, 0]), np.mean(sequence[:, 1]), 0.0, 0.0,
    ])


def test_diversified_train_bank_is_disjoint_from_fixed_and_held_action_templates() -> None:
    old = [
        {"action_sequence": np.zeros((20, 2)).tolist()},
        {"action_sequence": np.full((20, 2), 0.4).tolist()},
    ]
    held = [
        {"action_sequence": sequence.tolist()}
        for sequence in unseen.unseen_action_sequences(count=8)
    ]
    report = repair._assert_diverse_train_bank_disjoint(
        old, held, repair.diversified_train_action_sequences(count=12),
    )
    assert report["historical_hash_overlap"] == []
    assert report["held_hash_overlap"] == []
    assert len(report["diverse_action_hashes"]) == 12


def test_bilinear_effect_head_represents_state_conditioned_action_response() -> None:
    rng = np.random.default_rng(4)
    state = rng.normal(size=(100, 2))
    action = rng.normal(size=(100, 3))
    effect = np.column_stack((
        action[:, 0] * (1.0 + 0.3 * state[:, 0]),
        action[:, 1] * (1.0 - 0.2 * state[:, 1]),
    ))
    head = repair.fit_ridge_effect_head(state, action, effect, mode="bilinear", ridge=1.0e-4)
    prediction = head.predict(state, action)
    assert float(np.sqrt(np.mean((prediction - effect) ** 2))) < 0.03


def test_ood_repair_runner_uses_calibration_selection_and_sealed_held_outcomes(
    tmp_path: Path, monkeypatch
) -> None:
    dataset, arrays = _source_bank(tmp_path / "source")
    monkeypatch.setattr(unseen, "_simulate_sequence", _fake_simulator)
    context, outcomes, _states = unseen.materialize_unseen_bank(
        json.loads(dataset.read_text())["records"], tmp_path / "held", count=8,
    )
    monkeypatch.setattr(repair, "require_cuda_allocation", lambda _device: {"status": "passed", "device": "cuda"})
    monkeypatch.setattr(repair, "_simulate_sequence", _fake_simulator)
    monkeypatch.setattr(
        repair,
        "_ensemble_predict",
        lambda train_x, train_y, eval_x, **_kwargs: (np.zeros((len(eval_x), 2), dtype=np.float32), [{"seed": 0}]),
    )

    report = repair.run(
        dataset, arrays, context, outcomes, tmp_path / "result", epochs=1,
        bootstrap_samples=20, diverse_action_count=8, device="cuda",
    )

    assert report["protocol"]["outcome_used_for_architecture_hyperparameter_or_model_selection"] is False
    assert report["predeclared_selection"]["selected"]["config"] in repair.REPAIR_CONFIG_ORDER
    assert report["contracts"]["unseen_bank"]["action_hash_overlap"] == []
    assert report["contracts"]["diverse_training_bank"]["held_hash_overlap"] == []
    assert (tmp_path / "result" / "unseen_action_bank_ood_repair_result.json").is_file()
