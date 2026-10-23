"""Build a causal-temporal PointMaze action-OOD bank.

This builder is deliberately separate from the original unseen-action bank.
The previous bank carried only an identity ``source_history_model`` field, so
it could not decide whether actual observations before the decision point
repair frozen-latent aliasing.  Here a context consists of four frozen
observations at [t-15, t-10, t-5, t] and the *executed* controls on the three
intervals between them.  Candidate controls begin strictly after t.

The output context file contains no realized candidate outcome.  Outcomes are
written separately and the runner is responsible for opening them only after
all four predictors have been fitted.  This is an offline PointMaze mechanism
test; it does not claim visual deployment or a world-model result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark.audit_unseen_action_bank_contract import sequence_hash
from experiments.decision_benchmark.run_unseen_action_bank_pointmaze import (
    _load_records,
    _simulate_sequence,
)


SCHEMA = "pointmaze_temporal_ood_bank.v1"
ACTION_OOD_FAMILY = "world_fixed_four_pulse_temporal_ood_v1"
HISTORY_WINDOWS = 4
FRAME_STRIDE = 5


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def temporal_ood_sequences(*, count: int = 16, steps: int = 20) -> np.ndarray:
    """Return a fixed four-pulse bank disjoint from paired-order templates."""

    if count < 8 or steps < 8 or steps % 4:
        raise ValueError("count must be >=8 and steps must be divisible by four")
    pulse = steps // 4
    values: list[np.ndarray] = []
    for index in range(count):
        theta = 0.193 + 2.0 * np.pi * index / count
        radial = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)
        tangent = np.asarray([-radial[1], radial[0]], dtype=np.float64)
        amp = 0.36 + 0.05 * (index % 5)
        skew = (0.10 + 0.025 * (index % 4)) * (-1.0 if index % 2 else 1.0)
        pulses = np.asarray(
            [
                amp * radial + skew * tangent,
                0.65 * amp * radial - 0.85 * skew * tangent,
                0.90 * amp * radial + 0.40 * skew * tangent,
                0.30 * amp * radial - 0.55 * skew * tangent,
            ],
            dtype=np.float64,
        )
        sequence = np.repeat(pulses, pulse, axis=0)
        if float(np.abs(sequence).max()) > 1.0:
            raise RuntimeError("temporal OOD candidate exceeds PointMaze control support")
        values.append(sequence)
    bank = np.stack(values)
    if len({sequence_hash(item) for item in bank}) != len(bank):
        raise RuntimeError("temporal OOD bank contains duplicate sequences")
    return bank


def temporal_template_fingerprint(sequence: np.ndarray) -> str:
    """Fingerprint coarse timing, not numeric candidate identity."""

    value = np.asarray(sequence, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] % 4:
        raise ValueError("sequence must have [4*k, 2] shape")
    pulses = value.reshape(4, value.shape[0] // 4, 2).mean(axis=1)
    # Normalizing each pulse removes magnitude as an accidental identity leak
    # while preserving the signed temporal direction pattern.
    normalized = pulses / np.maximum(np.linalg.norm(pulses, axis=1, keepdims=True), 1.0e-12)
    return hashlib.sha256(np.round(normalized, 8).tobytes()).hexdigest()


def _root_rows(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    required = (
        "instance_id", "split", "source_state_raw", "source_latent_model", "source_history_model",
        "goal_raw", "start_episode_id", "source_frame", "action_sequence",
    )
    roots: dict[str, dict[str, Any]] = {}
    for row in records:
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"source record lacks fields: {missing}")
        root = str(row["instance_id"])
        reference = roots.setdefault(root, row)
        for key in ("split", "source_state_raw", "source_latent_model", "source_history_model", "goal_raw", "start_episode_id", "source_frame"):
            if key in {"split", "start_episode_id"}:
                if str(reference[key]) != str(row[key]):
                    raise ValueError(f"root has candidate-dependent {key}: {root}")
            elif not np.allclose(np.asarray(reference[key]), np.asarray(row[key]), rtol=0.0, atol=1e-10):
                raise ValueError(f"root has candidate-dependent {key}: {root}")
    if {str(row["split"]) for row in roots.values()} != {"train", "calibration", "outcome"}:
        raise ValueError("source records must contain root-disjoint train/calibration/outcome splits")
    return roots


def _source_template_fingerprints(records: list[dict[str, Any]]) -> set[str]:
    return {temporal_template_fingerprint(np.asarray(row["action_sequence"], dtype=np.float64)) for row in records}


def causal_history(
    source_frame: int,
    actions: np.ndarray,
    *,
    windows: int = HISTORY_WINDOWS,
    stride: int = FRAME_STRIDE,
) -> tuple[list[int], np.ndarray]:
    """Extract actions strictly before the candidate decision frame.

    The observation at ``source_frame`` is the final source observation.  For
    frames [t-15,t-10,t-5,t], actions [t-15:t] connect those observed frames;
    no action at t or later is retained.
    """

    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("PointMaze actions must have shape [time, 2]")
    if windows < 2 or stride < 1:
        raise ValueError("history needs at least two windows and positive stride")
    start = int(source_frame) - int((windows - 1) * stride)
    indices = list(range(start, int(source_frame) + 1, int(stride)))
    if start < 0 or indices[-1] != int(source_frame):
        raise ValueError("source frame has insufficient causal history")
    history = actions[start:int(source_frame)]
    if history.shape != ((windows - 1) * stride, 2):
        raise ValueError("causal history action window has invalid shape")
    return indices, history.copy()


def build(
    source_dataset: Path,
    dataset_root: Path,
    output: Path,
    *,
    candidate_count: int = 16,
    candidate_steps: int = 20,
) -> dict[str, Any]:
    """Materialize contexts and separately sealed outcomes without fitting."""

    metadata, records = _load_records(source_dataset)
    if int(metadata.get("history_windows", 0)) != HISTORY_WINDOWS:
        raise ValueError("source dataset must declare four frozen causal history windows")
    roots = _root_rows(records)
    lat_width = len(np.asarray(next(iter(roots.values()))["source_latent_model"]).reshape(-1))
    if any(len(np.asarray(row["source_history_model"]).reshape(-1)) != HISTORY_WINDOWS * lat_width for row in roots.values()):
        raise ValueError("source_history_model is not a four-frame frozen history")
    actions_path = dataset_root / "actions.pth"
    states_path = dataset_root / "states.pth"
    if not actions_path.exists() or not states_path.exists():
        raise FileNotFoundError("dataset root must contain actions.pth and states.pth")
    actions = np.asarray(torch.load(actions_path, map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    states = np.asarray(torch.load(states_path, map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    bank = temporal_ood_sequences(count=candidate_count, steps=candidate_steps)
    hashes = [sequence_hash(value) for value in bank]
    templates = [temporal_template_fingerprint(value) for value in bank]
    source_hashes = {sequence_hash(np.asarray(row["action_sequence"], dtype=np.float64)) for row in records}
    source_templates = _source_template_fingerprints(records)
    if set(hashes) & source_hashes:
        raise RuntimeError("temporal OOD action hashes overlap the source bank")
    if set(templates) & source_templates:
        raise RuntimeError("temporal OOD timing templates overlap the source bank")

    contexts: list[dict[str, Any]] = []
    outcomes: dict[str, dict[str, Any]] = {}
    root_history: dict[str, dict[str, Any]] = {}
    for root, source in sorted(roots.items()):
        episode = int(source["start_episode_id"])
        source_frame = int(source["source_frame"])
        if episode < 0 or episode >= len(actions) or episode >= len(states):
            raise ValueError(f"episode outside dataset arrays: {episode}")
        indices, past_actions = causal_history(source_frame, actions[episode])
        state = np.asarray(source["source_state_raw"], dtype=np.float64)
        if not np.allclose(states[episode, source_frame], state, rtol=0.0, atol=1e-8):
            raise ValueError(f"source state does not match recorded decision frame for {root}")
        root_history[root] = {
            "episode": episode,
            "frame_indices": indices,
            "action_interval": [indices[0], source_frame],
            "action_sha256": hashlib.sha256(np.ascontiguousarray(past_actions).tobytes()).hexdigest(),
        }
        goal = np.asarray(source["goal_raw"], dtype=np.float64)
        for candidate_id, candidate in enumerate(bank):
            key = f"{root}/{candidate_id}"
            terminal = _simulate_sequence(state, candidate)
            contexts.append({
                "instance_id": root,
                "split": str(source["split"]),
                "environment": "PointMaze",
                "start_episode_id": str(episode),
                "source_frame": source_frame,
                "history_frame_indices": indices,
                "history_action_interval": [indices[0], source_frame],
                "history_actions_raw": past_actions.tolist(),
                "source_latent_model": list(source["source_latent_model"]),
                "source_history_model": list(source["source_history_model"]),
                "source_state_raw": state.tolist(),
                "goal_raw": goal.tolist(),
                "candidate_id": candidate_id,
                "candidate_generator": ACTION_OOD_FAMILY,
                "candidate_bank_hash": _canonical_hash(hashes),
                "candidate_action_hash": hashes[candidate_id],
                "candidate_template_fingerprint": templates[candidate_id],
                "action_sequence": candidate.tolist(),
                "total_action_steps": int(candidate_steps),
                "outcome_label_status": "sealed_until_postfit_join",
            })
            outcomes[key] = {
                "real_task_cost": float(np.linalg.norm(terminal[:2] - goal[:2])),
                "realized_terminal_state": terminal.tolist(),
            }
    output.mkdir(parents=True, exist_ok=True)
    context_path = output / "temporal_ood_context.json"
    outcome_path = output / "temporal_ood_outcomes.json"
    source_temporal_path = output / "temporal_source_history_dataset.json"
    context_payload = {
        "metadata": {
            "schema": SCHEMA,
            "source_dataset": str(source_dataset.resolve()),
            "history_windows": HISTORY_WINDOWS,
            "history_frame_stride": FRAME_STRIDE,
            "history_is_causal": True,
            "history_actions_are_strictly_pre_candidate": True,
            "candidate_family": ACTION_OOD_FAMILY,
            "candidate_action_hashes": hashes,
            "candidate_template_fingerprints": templates,
            "outcomes_sealed_from_feature_construction": True,
            "root_history": root_history,
        },
        "records": contexts,
    }
    context_path.write_text(json.dumps(context_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outcome_path.write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Preserve the original (non-OOD) candidate sequences for fitting, but add
    # only source-side history that was available before each original
    # candidate.  In particular, this does not copy any temporal-OOD outcome
    # into training features.
    enriched_source = []
    for row in records:
        copied = dict(row)
        history = root_history[str(row["instance_id"])]
        episode = int(history["episode"])
        start, end = [int(value) for value in history["action_interval"]]
        copied.update({
            "history_frame_indices": list(history["frame_indices"]),
            "history_action_interval": [start, end],
            "history_actions_raw": actions[episode, start:end].tolist(),
            "history_action_sha256": str(history["action_sha256"]),
        })
        enriched_source.append(copied)
    source_temporal_path.write_text(json.dumps({
        "metadata": {
            **metadata,
            "schema": "pointmaze_temporal_source_history.v1",
            "history_windows": HISTORY_WINDOWS,
            "history_frame_stride": FRAME_STRIDE,
            "history_is_causal": True,
            "history_actions_are_strictly_pre_candidate": True,
            "source_history_origin": "recorded PointMaze actions before each original candidate decision frame",
        },
        "records": enriched_source,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "schema": SCHEMA,
        "context": str(context_path.resolve()),
        "sealed_outcomes": str(outcome_path.resolve()),
        "temporal_source_dataset": str(source_temporal_path.resolve()),
        "root_counts": {split: sum(str(row["split"]) == split for row in roots.values()) for split in ("train", "calibration", "outcome")},
        "candidate_count": candidate_count,
        "history_contract": "four pre-candidate frozen observations plus executed raw actions from t-15 through t-1",
        "candidate_contract": "all candidates begin at source frame t and share one held action-OOD bank",
        "source_action_hash_overlap": sorted(set(hashes) & source_hashes),
        "source_template_overlap": sorted(set(templates) & source_templates),
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--candidate-steps", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
