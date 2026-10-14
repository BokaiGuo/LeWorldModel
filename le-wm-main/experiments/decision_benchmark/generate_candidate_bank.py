"""Generate a goal-conditioned, equal-duration PointMaze counterfactual bank.

Each instance fixes one observed source state and one explicit environment goal.
All candidates execute exactly ``--steps`` simulator steps; only the complete
action sequence changes.  The adapter sees RGB-derived source latent plus the
candidate action sequence.  Simulator state is used only for offline task
cost/success labels.  Because this repository does not contain RGB rollouts
for these newly simulated candidates, realized visual latents remain
explicitly unavailable and the downstream validator stays fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.empirical_spectral_audit.enriched_record_schema import load_payload
from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import (
    _external_features,
    _legacy_pointmaze_xml,
    _load_adapter,
)
from experiments.empirical_spectral_audit.train_pointmaze_learned_adapter_panel import _predict


def _bank_hash(actions: np.ndarray) -> str:
    normalized = np.asarray(actions, dtype=np.float64).round(10).copy()
    normalized[normalized == 0.0] = 0.0
    canonical = normalized.tobytes()
    return hashlib.sha256(canonical).hexdigest()[:20]


def _candidate_sequences(
    source: np.ndarray,
    goal: np.ndarray,
    count: int,
    steps: int,
    seed: int,
    family: str = "paired_order_v2",
    canonical_action_frame: bool = False,
) -> np.ndarray:
    if count < 4 or steps < 1:
        raise ValueError("candidate count must be >=4 and steps must be positive")
    if steps % 4:
        raise ValueError("steps must be divisible by four to match the adapter action-block clock")
    source = np.asarray(source, dtype=np.float64)[:2]
    goal = np.asarray(goal, dtype=np.float64)[:2]
    delta = goal - source
    norm = max(float(np.linalg.norm(delta)), 1e-8)
    goal_direct = delta / norm
    goal_perpendicular = np.asarray([-goal_direct[1], goal_direct[0]])
    direct = np.asarray([1.0, 0.0]) if canonical_action_frame else goal_direct
    perpendicular = np.asarray([0.0, 1.0]) if canonical_action_frame else goal_perpendicular
    sequences: list[np.ndarray] = []

    def constant(direction: np.ndarray, magnitude: float) -> np.ndarray:
        return np.tile(np.clip(direction * magnitude, -1.0, 1.0), (steps, 1))

    if family == "goal_directed_mixture_v1":
        for magnitude in (0.25, 0.5, 0.75, 1.0):
            sequences.append(constant(direct, magnitude))
        for sign in (-1.0, 1.0):
            sequences.append(constant(perpendicular * sign, 0.75))
        for magnitude in (0.5, 0.75):
            sequence = np.zeros((steps, 2), dtype=np.float64)
            switch = max(1, steps // 2)
            sequence[:switch] = direct * magnitude
            sequence[switch:] = perpendicular * magnitude
            sequences.append(np.clip(sequence, -1.0, 1.0))
        sequences.append(np.zeros((steps, 2), dtype=np.float64))
    elif family == "paired_order_v2":
        # Hard pairs preserve the same cumulative action while changing its
        # temporal order.  A goal-direction dot product therefore cannot
        # resolve the pair; only dynamics and interaction timing can.
        half = max(1, steps // 2)
        quarter = max(1, steps // 4)

        def pair(first: np.ndarray, second: np.ndarray) -> None:
            left = np.zeros((steps, 2), dtype=np.float64)
            right = np.zeros((steps, 2), dtype=np.float64)
            left[:half] = first
            left[half:] = second
            right[:half] = second
            right[half:] = first
            sequences.extend((np.clip(left, -1.0, 1.0), np.clip(right, -1.0, 1.0)))

        pair(direct * 0.75, perpendicular * 0.5)
        pair(direct * 0.75, -perpendicular * 0.5)
        pair(direct * 0.9, -direct * 0.3)
        pulse_a = np.zeros((steps, 2), dtype=np.float64)
        pulse_b = np.zeros((steps, 2), dtype=np.float64)
        pulse_a[:quarter] = direct * 1.0
        pulse_b[-quarter:] = direct * 1.0
        sequences.extend((pulse_a, pulse_b))
        sequences.append(constant(direct, 0.35))
        sequences.append(constant(perpendicular, 0.75))
        sequences.append(np.zeros((steps, 2), dtype=np.float64))
    else:
        raise ValueError(f"unsupported candidate family: {family}")

    rng = np.random.default_rng(seed)
    while len(sequences) < count:
        sequence = rng.uniform(-0.75, 0.75, size=(steps, 2))
        if canonical_action_frame:
            # A rotation preserves the 2-norm.  Bound the local vectors so
            # every root remains inside the simulator's [-1, 1] controls.
            norms = np.linalg.norm(sequence, axis=1, keepdims=True)
            sequence = sequence * np.minimum(1.0, 0.75 / np.maximum(norms, 1.0e-12))
        sequences.append(sequence.astype(np.float64))
    local_sequences = np.stack(sequences[:count], axis=0)
    if not canonical_action_frame:
        return local_sequences
    # The candidate bank is sampled once in a goal-relative frame, then
    # rotated into each root's raw control coordinates.  This preserves the
    # intended candidate semantics while keeping the simulator action valid.
    return local_sequences[:, :, 0, None] * goal_direct + local_sequences[:, :, 1, None] * goal_perpendicular


def _to_goal_frame(actions: np.ndarray, source: np.ndarray, goal: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)[:2]
    goal = np.asarray(goal, dtype=np.float64)[:2]
    delta = goal - source
    norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-8:
        raise ValueError("goal-frame action coordinates require a nonzero source-goal displacement")
    direct = delta / norm
    perpendicular = np.asarray([-direct[1], direct[0]], dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    return np.stack([actions @ direct, actions @ perpendicular], axis=-1)


def _simulate_sequence(source_state: np.ndarray, sequence: np.ndarray) -> np.ndarray:
    import mujoco

    model = mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())
    data = mujoco.MjData(model)
    source_state = np.asarray(source_state, dtype=np.float64)
    data.qpos[:] = source_state[:2]
    data.qvel[:] = source_state[2:4]
    mujoco.mj_forward(model, data)
    for action in np.asarray(sequence, dtype=np.float64):
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
    return np.concatenate([np.asarray(data.qpos[:2]), np.asarray(data.qvel[:2])])


def _split_for_index(index: int, assignment: str = "legacy_v1") -> str:
    # Root-disjoint and deterministic.  The fifth fold is the held-out outcome.
    if assignment == "legacy_v1":
        return ("train", "train", "train", "calibration", "outcome")[index % 5]
    if assignment == "distance_balanced_v1":
        # Source rows are ordered by a pre-outcome support score.  Taking one
        # root from each ordered block for outcome prevents a support collapse
        # caused by assigning all near-goal roots to train/calibration.
        return ("outcome", "train", "train", "calibration", "train")[index % 5]
    raise ValueError(f"unsupported split assignment: {assignment}")


def generate(
    source_export: Path,
    checkpoint: Path,
    output: Path,
    *,
    goal: tuple[float, float] = (1.2, 1.2),
    candidates: int = 16,
    steps: int = 20,
    max_instances: int = 40,
    device: str = "cpu",
    candidate_family: str = "paired_order_v2",
    source_selection: str = "sorted_episode",
    split_assignment: str = "legacy_v1",
    source_window_start: int = 0,
    history_windows: int = 1,
    canonical_action_bank: bool = False,
) -> dict[str, Any]:
    payload = load_payload(source_export)
    model, checkpoint_payload = _load_adapter(checkpoint, device)
    pca_mean = np.asarray(checkpoint_payload["pca_mean"], dtype=np.float64)
    pca_basis = np.asarray(checkpoint_payload["pca_basis"], dtype=np.float64)
    goal_array = np.asarray(goal, dtype=np.float64)
    # Use one source observation per episode, not adjacent windows from the
    # same root.  The source export is frozen and never modified.
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in payload["enriched_records"]:
        episode = str(row["episode_id"])
        by_episode.setdefault(episode, []).append(row)
    by_episode = {
        episode: sorted(rows, key=lambda item: int(item.get("window_start", 0)))
        for episode, rows in by_episode.items()
    }
    source_rows_by_episode: dict[str, dict[str, Any]] = {}
    history_by_episode: dict[str, np.ndarray] = {}
    for episode, rows in by_episode.items():
        eligible = [row for row in rows if int(row.get("window_start", 0)) == int(source_window_start)]
        if len(eligible) != 1:
            continue
        source_rows_by_episode[episode] = eligible[0]
        history_rows = [
            row for row in rows
            if int(row.get("window_start", 0)) <= int(source_window_start)
        ][-int(history_windows):]
        if len(history_rows) != int(history_windows):
            continue
        history_by_episode[episode] = np.stack([
            (np.asarray(row["source_latent"], dtype=np.float64) - pca_mean) @ pca_basis.T
            for row in history_rows
        ]).reshape(-1)
    if history_windows < 1:
        raise ValueError("history_windows must be positive")
    if source_selection == "sorted_episode":
        selected_keys = sorted(source_rows_by_episode)
    elif source_selection == "nearest_goal":
        selected_keys = sorted(
            source_rows_by_episode,
            key=lambda key: (
                float(np.linalg.norm(np.asarray(source_rows_by_episode[key]["source_state_raw"], dtype=np.float64)[:2] - goal_array)),
                str(key),
            ),
        )
    elif source_selection == "reachable_goal":
        selected_keys = sorted(
            source_rows_by_episode,
            key=lambda key: (
                float(np.linalg.norm(np.asarray(source_rows_by_episode[key]["source_state_raw"], dtype=np.float64)[:2] - goal_array))
                + float(np.linalg.norm(np.asarray(source_rows_by_episode[key]["source_state_raw"], dtype=np.float64)[2:4])),
                str(key),
            ),
        )
    else:
        raise ValueError(f"unsupported source_selection: {source_selection}")
    source_rows = [source_rows_by_episode[key] for key in selected_keys[:max_instances]]
    if len(source_rows) < 5:
        raise ValueError("need at least five episode-disjoint source roots")
    all_records: list[dict[str, Any]] = []
    predicted_latents: dict[str, np.ndarray] = {}
    realized_states: dict[str, np.ndarray] = {}
    for instance_index, row in enumerate(source_rows):
        instance_id = f"pointmaze_episode_{row['episode_id']}"
        source_state = np.asarray(row["source_state_raw"], dtype=np.float64)
        source_history_projection = history_by_episode[str(row["episode_id"])]
        sequences = _candidate_sequences(
            source_state,
            goal_array,
            candidates,
            steps,
            7000 if canonical_action_bank else 7000 + instance_index,
            candidate_family,
            canonical_action_frame=canonical_action_bank,
        )
        bank_hash = _bank_hash(
            np.round(_to_goal_frame(sequences, source_state, goal_array), 8)
            if canonical_action_bank
            else sequences
        )
        source_latent = np.asarray(row["source_latent"], dtype=np.float64)
        source_projection = (source_latent - pca_mean) @ pca_basis.T
        action_blocks = sequences.reshape(candidates, 4, steps // 4, 2).reshape(candidates, -1)
        checkpoint_history_windows = int(checkpoint_payload.get("adapter_training", {}).get("history_windows", 1))
        if checkpoint_history_windows == 1:
            context_projection = np.repeat(source_projection[None], candidates, axis=0)
        else:
            if source_history_projection.size != checkpoint_history_windows * source_projection.size:
                raise ValueError(
                    "history context width does not match checkpoint: "
                    f"context={source_history_projection.size}, expected="
                    f"{checkpoint_history_windows * source_projection.size}"
                )
            context_projection = np.repeat(source_history_projection[None], candidates, axis=0)
        x = np.concatenate([context_projection, action_blocks], axis=1)
        prediction = _predict(
            model,
            x,
            int(pca_basis.shape[0]),
            device,
            checkpoint_payload["adapter_training"],
        )
        horizon = max(prediction)
        predicted = np.asarray(prediction[horizon], dtype=np.float32)
        for candidate_index, sequence in enumerate(sequences):
            final_state = _simulate_sequence(source_state, sequence)
            distance = float(np.linalg.norm(final_state[:2] - goal_array))
            record_key = f"{instance_id}/{candidate_index:03d}"
            predicted_latents[record_key] = predicted[candidate_index]
            realized_states[record_key] = final_state
            all_records.append({
                "instance_id": instance_id,
                "split": _split_for_index(instance_index, split_assignment),
                "environment": "PointMaze",
                "start_episode_id": str(row["episode_id"]),
                "source_frame": int(row.get("source_frame", int(source_window_start) + 5)),
                "context_length": int(history_windows) if int(history_windows) > 1 else 2,
                "goal_id": "legacy_pointmaze_goal_1p2_1p2",
                "candidate_id": candidate_index,
                "candidate_bank_hash": bank_hash,
                "action_sequence": sequence.tolist(),
                "total_action_steps": int(steps),
                "candidate_generator": candidate_family,
                "cem_stage": "fixed_counterfactual_bank",
                "source_observation_ref": f"frozen_export://{row['sample_id']}",
                "goal_observation_ref": "environment_goal://PointMaze/legacy_goal_1p2_1p2",
                "teacher_model_id": "offline_environment_cost_v1",
                "teacher_score": -distance,
                "teacher_rank": None,
                "teacher_set_id": f"{instance_id}_shared_bank",
                "real_success": bool(distance <= 0.25),
                "real_task_cost": distance,
                "realized_observation_ref": f"offline_mujoco_state://{record_key}",
                "predicted_latent_ref": f"arrays.npz::predicted::{record_key}",
                "realized_latent_ref": None,
                "action_support": {
                    "chart": "raw_control",
                    "min": float(np.min(sequence)),
                    "max": float(np.max(sequence)),
                    "duration_steps": int(steps),
                },
                "privileged_state_role": "offline_outcome_label_only",
                "source_state_raw": source_state.tolist(),
                "source_latent_model": source_projection.tolist(),
                "source_history_model": source_history_projection.tolist(),
                "goal_raw": goal_array.tolist(),
                "predicted_horizon": int(horizon),
                "predicted_terminal_latent_ref": f"arrays.npz::predicted::{record_key}",
                "realized_terminal_state_ref": f"arrays.npz::realized_state::{record_key}",
            })
        group = all_records[-candidates:]
        ranking = sorted(group, key=lambda item: (-float(item["teacher_score"]), int(item["candidate_id"])))
        for rank, item in enumerate(ranking):
            item["teacher_rank"] = int(rank)

    metadata = {
        "outcome_used_for_teacher_or_metric_selection": False,
        "snapshot_selection_from_outcome": False,
        "teacher_frozen_before_distillation": True,
        "shared_candidate_bank": True,
        "runtime_teacher_forbidden": True,
        "candidate_bank_count": int(candidates),
        "candidate_family": candidate_family,
        "candidate_bank_coordinate_frame": "goal_relative" if canonical_action_bank else "raw_control",
        "canonical_action_bank": bool(canonical_action_bank),
        "source_selection": source_selection,
        "split_assignment": split_assignment,
        "source_window_start": int(source_window_start),
        "source_frame": int(source_rows[0].get("source_frame", int(source_window_start) + 5)),
        "context_length": int(history_windows) if int(history_windows) > 1 else 2,
        "history_windows": int(history_windows),
        "predictor_history_windows": int(checkpoint_payload.get("adapter_training", {}).get("history_windows", 1)),
        "total_action_steps": int(steps),
        "goal_definition": "explicit legacy PointMaze environment goal coordinate",
        "goal_coordinate": goal_array.tolist(),
        "source_export": str(source_export.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "realized_latent_status": "unavailable_without_RGB_rollout_for_new_simulated_sequences",
    }
    output.mkdir(parents=True, exist_ok=True)
    arrays_path = output / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        **{f"predicted::{key}": value for key, value in predicted_latents.items()},
        **{f"realized_state::{key}": value for key, value in realized_states.items()},
    )
    dataset = {"metadata": metadata, "records": all_records}
    (output / "candidate_dataset.json").write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "pointmaze_goal_conditioned_candidate_bank.v1",
        "metadata": metadata,
        "record_count": len(all_records),
        "instance_count": len(source_rows),
        "split_instance_counts": {
            split: len({item["instance_id"] for item in all_records if item["split"] == split})
            for split in ("train", "calibration", "outcome")
        },
        "arrays": str(arrays_path.resolve()),
        "candidate_dataset": str((output / "candidate_dataset.json").resolve()),
        "claim_boundary": "Equal-duration goal-conditioned counterfactual outcome benchmark; realized visual latent and closed-loop RGB evidence remain unavailable.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-export", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal-x", type=float, default=1.2)
    parser.add_argument("--goal-y", type=float, default=1.2)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-instances", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-family", default="paired_order_v2", choices=("paired_order_v2", "goal_directed_mixture_v1"))
    parser.add_argument("--source-selection", default="sorted_episode", choices=("sorted_episode", "nearest_goal", "reachable_goal"))
    parser.add_argument("--split-assignment", default="legacy_v1", choices=("legacy_v1", "distance_balanced_v1"))
    parser.add_argument("--source-window-start", type=int, default=0)
    parser.add_argument("--history-windows", type=int, default=1)
    parser.add_argument("--canonical-action-bank", action="store_true")
    args = parser.parse_args()
    manifest = generate(
        args.source_export,
        args.checkpoint,
        args.output,
        goal=(args.goal_x, args.goal_y),
        candidates=args.candidates,
        steps=args.steps,
        max_instances=args.max_instances,
        device=args.device,
        candidate_family=args.candidate_family,
        source_selection=args.source_selection,
        split_assignment=args.split_assignment,
        source_window_start=args.source_window_start,
        history_windows=args.history_windows,
        canonical_action_bank=args.canonical_action_bank,
    )
    print(json.dumps({key: manifest[key] for key in ("record_count", "instance_count", "split_instance_counts", "claim_boundary")}, indent=2))


if __name__ == "__main__":
    main()
