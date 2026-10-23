from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.decision_benchmark import build_pointmaze_temporal_ood_bank as builder
from experiments.decision_benchmark import run_pointmaze_temporal_ood_test as runner


def _source_rows() -> list[dict[str, object]]:
    rows = []
    for index, (root, split) in enumerate((("train-a", "train"), ("train-b", "train"), ("cal", "calibration"), ("out", "outcome"))):
        state = [float(index), 0.0, 0.0, 0.0]
        for candidate, direction in enumerate(([-0.2, 0.0], [0.2, 0.0])):
            action = np.repeat(np.asarray(direction, dtype=np.float64)[None, :], 20, axis=0)
            rows.append({
                "instance_id": root, "split": split, "candidate_id": candidate,
                "start_episode_id": str(index), "source_frame": 20,
                "source_state_raw": state, "goal_raw": [1.0, 0.0],
                "source_latent_model": [float(index), 0.0],
                "source_history_model": [float(index - 3), 0.0, float(index - 2), 0.0, float(index - 1), 0.0, float(index), 0.0],
                "action_sequence": action.tolist(),
                "realized_terminal_state_ref": f"arrays.npz::terminal/{root}/{candidate}",
            })
    return rows


def _write_source(tmp_path: Path) -> tuple[Path, Path, Path]:
    data = tmp_path / "data"
    data.mkdir()
    states = np.zeros((4, 40, 4), dtype=np.float64)
    actions = np.zeros((4, 40, 2), dtype=np.float64)
    arrays = {}
    for index, row in enumerate(_source_rows()):
        episode = int(row["start_episode_id"])
        states[episode, 20] = np.asarray(row["source_state_raw"], dtype=np.float64)
        actions[episode, 5:20] = float(episode + 1) / 10.0
        terminal = np.asarray(row["source_state_raw"], dtype=np.float64) + np.asarray([(-0.2 if int(row["candidate_id"]) == 0 else 0.2), 0.0, 0.0, 0.0])
        arrays[f"terminal/{row['instance_id']}/{row['candidate_id']}"] = terminal
    torch.save(torch.as_tensor(states), data / "states.pth")
    torch.save(torch.as_tensor(actions), data / "actions.pth")
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"metadata": {"history_windows": 4}, "records": _source_rows()}))
    array_path = tmp_path / "arrays.npz"
    np.savez(array_path, **arrays)
    return source, array_path, data


def test_causal_history_rejects_post_decision_actions() -> None:
    actions = np.zeros((30, 2), dtype=np.float64)
    actions[5:20] = 1.0
    actions[20:] = 9.0
    frames, history = builder.causal_history(20, actions)
    assert frames == [5, 10, 15, 20]
    assert history.shape == (15, 2)
    assert float(history.max()) == 1.0


def test_builder_writes_causal_enriched_source_and_sealed_outcomes(tmp_path: Path, monkeypatch) -> None:
    source, _arrays, data = _write_source(tmp_path)
    monkeypatch.setattr(builder, "_simulate_sequence", lambda state, action: np.asarray(state) + np.asarray([float(action[:, 0].mean()), float(action[:, 1].mean()), 0.0, 0.0]))
    report = builder.build(source, data, tmp_path / "bank", candidate_count=8)
    context = json.loads(Path(report["context"]).read_text())
    enriched = json.loads(Path(report["temporal_source_dataset"]).read_text())
    assert report["source_action_hash_overlap"] == []
    assert report["source_template_overlap"] == []
    assert all("real_task_cost" not in row for row in context["records"])
    assert all(len(row["history_actions_raw"]) == 15 for row in context["records"])
    assert all(row["history_action_interval"][1] == row["source_frame"] for row in enriched["records"])


def test_history_feature_requires_causal_ledger() -> None:
    row = {
        "source_latent_model": [0.0, 0.0],
        "source_history_model": [0.0] * 8,
        "source_state_raw": [0.0] * 4,
        "source_frame": 20,
        "history_frame_indices": [5, 10, 15, 20],
        "history_action_interval": [5, 20],
        "history_actions_raw": np.zeros((15, 2)).tolist(),
        "action_sequence": np.zeros((20, 2)).tolist(),
    }
    assert runner._feature(row, "history_latent_action").shape == (8 + 30 + 40,)
    row["history_action_interval"] = [5, 21]
    try:
        runner._feature(row, "history_latent_action")
    except ValueError as exc:
        assert "does not end" in str(exc)
    else:
        raise AssertionError("post-decision history action was accepted")


def test_contract_rejects_candidate_hash_or_template_overlap() -> None:
    source = _source_rows()
    metadata = {"history_windows": 4}
    candidate = np.asarray(source[0]["action_sequence"], dtype=np.float64)
    row = {**source[0], "history_frame_indices": [5, 10, 15, 20], "history_action_interval": [5, 20], "history_actions_raw": np.zeros((15, 2)).tolist(), "candidate_action_hash": __import__("hashlib").sha256(np.ascontiguousarray(candidate).tobytes()).hexdigest(), "candidate_template_fingerprint": builder.temporal_template_fingerprint(candidate), "action_sequence": candidate.tolist(), "candidate_generator": builder.ACTION_OOD_FAMILY}
    ood_metadata = {"history_is_causal": True, "history_actions_are_strictly_pre_candidate": True, "candidate_family": builder.ACTION_OOD_FAMILY}
    try:
        runner._validate_contract(source, metadata, [row], ood_metadata)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlapping action candidate was accepted")


def test_alias_resolution_uses_each_representation_own_nearest_neighbor(tmp_path: Path) -> None:
    """A history-near root with matching effect must resolve a current alias."""
    rows, outcomes = [], {}
    # ``out`` is current-latent-near to a (effect 0) but history-near to b
    # (effect 10).  Reusing the current neighbour's divergence for history
    # would therefore falsely keep it marked as a history alias.
    for root, split, current, history, effect in (
        ("a", "train", 0.0, 0.0, 0.0),
        ("b", "train", 0.01, 100.0, 10.0),
        ("c", "train", 10.0, 100.01, 0.0),
        ("out", "outcome", 0.005, 100.005, 10.0),
    ):
        sequence = np.zeros((20, 2), dtype=np.float64)
        row = {
            "instance_id": root, "split": split, "candidate_id": 0,
            "source_latent_model": [current], "source_history_model": [history] * 4,
            "source_state_raw": [0.0, 0.0, 0.0, 0.0], "goal_raw": [1.0, 0.0],
            "source_frame": 20, "history_frame_indices": [5, 10, 15, 20],
            "history_action_interval": [5, 20], "history_actions_raw": np.zeros((15, 2)).tolist(),
            "candidate_action_hash": "shared", "action_sequence": sequence.tolist(),
        }
        rows.append(row)
        outcomes[f"{root}/0"] = {"realized_terminal_state": [effect, 0.0, 0.0, 0.0], "real_task_cost": 0.0}
    path = tmp_path / "outcomes.json"
    path.write_text(json.dumps(outcomes))
    report = runner.alias_resolution(rows, path)
    selected = next(row for row in report["rows"] if row["root"] == "out")
    assert selected["current_alias_conflict"] is True
    assert selected["history_alias_conflict"] is False
    assert selected["alias_resolved_by_history"] is True
