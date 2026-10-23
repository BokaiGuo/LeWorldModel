"""Build the independently seeded causal-temporal PointMaze OOD-v2 bank.

V2 never reuses a v1 outcome root.  From the remaining source roots it forms
one deterministic 32/24/24 train/calibration/outcome split, and it draws a
new four-pulse action bank from a declared seed.  The source history contains
only observations at t-15,t-10,t-5,t and actions on [t-15,t); candidates
start at t.  Candidate outcomes are stored in a separate sealed file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark.audit_unseen_action_bank_contract import sequence_hash
from experiments.decision_benchmark.build_pointmaze_temporal_ood_bank import (
    ACTION_OOD_FAMILY as V1_ACTION_OOD_FAMILY,
    FRAME_STRIDE,
    HISTORY_WINDOWS,
    _root_rows,
    _source_template_fingerprints,
    causal_history,
    temporal_template_fingerprint,
)
from experiments.decision_benchmark.run_unseen_action_bank_pointmaze import _load_records, _simulate_sequence


SCHEMA = "pointmaze_temporal_ood_v2_bank.v1"
V2_ACTION_OOD_FAMILY = "world_fixed_random_four_pulse_temporal_ood_v2"
DEFAULT_ROOT_SPLIT_SEED = 202610212
DEFAULT_ACTION_BANK_SEED = 202610213
SPLIT_COUNTS = {"train": 32, "calibration": 24, "outcome": 24}


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def v2_action_sequences(*, seed: int, count: int = 16, steps: int = 20) -> np.ndarray:
    """Draw a declared, deterministic action bank from a fresh seed."""

    if count < 8 or steps < 8 or steps % 4:
        raise ValueError("count must be >=8 and steps must be divisible by four")
    rng = np.random.default_rng(int(seed))
    sequences: list[np.ndarray] = []
    for _ in range(int(count)):
        angles = rng.uniform(-np.pi, np.pi, size=4)
        magnitudes = rng.uniform(0.24, 0.72, size=4)
        # Four independently sampled pulses deliberately differ from the
        # structured v1 radial/tangential schedule and paired-order training
        # templates.  The seed, not outcome labels, fixes every value.
        pulses = np.stack((np.cos(angles), np.sin(angles)), axis=1) * magnitudes[:, None]
        sequence = np.repeat(pulses, steps // 4, axis=0)
        if float(np.abs(sequence).max()) > 1.0:
            raise RuntimeError("v2 action exceeds control support")
        sequences.append(sequence)
    bank = np.stack(sequences)
    if len({sequence_hash(item) for item in bank}) != len(bank):
        raise RuntimeError("v2 action bank has duplicate sequence hashes")
    if len({temporal_template_fingerprint(item) for item in bank}) != len(bank):
        raise RuntimeError("v2 action bank has duplicate timing templates")
    return bank


def _v1_outcome_roots(v1_context: Path) -> tuple[set[str], set[str], set[str]]:
    metadata, records = _load_records(v1_context)
    if metadata.get("candidate_family") != V1_ACTION_OOD_FAMILY:
        raise ValueError("v1 context has an unexpected action family")
    outcomes = {str(row["instance_id"]) for row in records if str(row["split"]) == "outcome"}
    hashes = {str(row["candidate_action_hash"]) for row in records}
    templates = {str(row["candidate_template_fingerprint"]) for row in records}
    if not outcomes or not hashes or not templates:
        raise ValueError("v1 context lacks roots or action identity fields")
    return outcomes, hashes, templates


def _fresh_split(roots: list[str], *, seed: int) -> dict[str, str]:
    expected = sum(SPLIT_COUNTS.values())
    if len(roots) != expected:
        raise ValueError(f"v2 needs exactly {expected} eligible roots, received {len(roots)}")
    ordered = np.asarray(sorted(roots), dtype=object)
    shuffled = ordered[np.random.default_rng(int(seed)).permutation(len(ordered))].tolist()
    result: dict[str, str] = {}
    offset = 0
    for split in ("train", "calibration", "outcome"):
        width = SPLIT_COUNTS[split]
        for root in shuffled[offset:offset + width]:
            result[str(root)] = split
        offset += width
    return result


def build(
    source_dataset: Path,
    dataset_root: Path,
    v1_context: Path,
    output: Path,
    *,
    root_split_seed: int = DEFAULT_ROOT_SPLIT_SEED,
    action_bank_seed: int = DEFAULT_ACTION_BANK_SEED,
    candidate_count: int = 16,
    candidate_steps: int = 20,
) -> dict[str, Any]:
    metadata, records = _load_records(source_dataset)
    if int(metadata.get("history_windows", 0)) != HISTORY_WINDOWS:
        raise ValueError("v2 requires a four-frame frozen-history source bank")
    roots = _root_rows(records)
    v1_outcomes, v1_hashes, v1_templates = _v1_outcome_roots(v1_context)
    eligible = sorted(set(roots) - v1_outcomes)
    split = _fresh_split(eligible, seed=root_split_seed)
    if set(split) & v1_outcomes:
        raise RuntimeError("v2 root split reuses a v1 outcome root")
    lat_width = len(np.asarray(next(iter(roots.values()))["source_latent_model"]).reshape(-1))
    if any(len(np.asarray(row["source_history_model"]).reshape(-1)) != HISTORY_WINDOWS * lat_width for row in roots.values()):
        raise ValueError("source history does not contain four frozen frames")
    actions_path, states_path = dataset_root / "actions.pth", dataset_root / "states.pth"
    if not actions_path.exists() or not states_path.exists():
        raise FileNotFoundError("dataset root must contain actions.pth and states.pth")
    actions = np.asarray(torch.load(actions_path, map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    states = np.asarray(torch.load(states_path, map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    bank = v2_action_sequences(seed=action_bank_seed, count=candidate_count, steps=candidate_steps)
    hashes = [sequence_hash(item) for item in bank]
    templates = [temporal_template_fingerprint(item) for item in bank]
    source_hashes = {sequence_hash(np.asarray(row["action_sequence"], dtype=np.float64)) for row in records}
    source_templates = _source_template_fingerprints(records)
    if set(hashes) & source_hashes or set(hashes) & v1_hashes:
        raise RuntimeError("v2 action sequence hash overlaps source or v1")
    if set(templates) & source_templates or set(templates) & v1_templates:
        raise RuntimeError("v2 action template overlaps source or v1")

    history_by_root: dict[str, dict[str, Any]] = {}
    contexts: list[dict[str, Any]] = []
    outcomes: dict[str, Any] = {}
    for root in sorted(split):
        source = roots[root]
        episode, frame = int(source["start_episode_id"]), int(source["source_frame"])
        frames, history_actions = causal_history(frame, actions[episode])
        state = np.asarray(source["source_state_raw"], dtype=np.float64)
        if not np.allclose(states[episode, frame], state, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"source state does not match recorded frame for {root}")
        history_by_root[root] = {"episode": episode, "frames": frames, "interval": [frames[0], frame], "actions": history_actions}
        goal = np.asarray(source["goal_raw"], dtype=np.float64)
        for index, candidate in enumerate(bank):
            key = f"{root}/{5000 + index}"
            terminal = _simulate_sequence(state, candidate)
            contexts.append({
                "instance_id": root, "split": split[root], "environment": "PointMaze",
                "start_episode_id": str(episode), "source_frame": frame,
                "history_frame_indices": frames, "history_action_interval": [frames[0], frame],
                "history_actions_raw": history_actions.tolist(),
                "source_latent_model": list(source["source_latent_model"]),
                "source_history_model": list(source["source_history_model"]),
                "source_state_raw": state.tolist(), "goal_raw": goal.tolist(),
                "candidate_id": 5000 + index, "candidate_generator": V2_ACTION_OOD_FAMILY,
                "candidate_bank_hash": _json_hash(hashes), "candidate_action_hash": hashes[index],
                "candidate_template_fingerprint": templates[index], "action_sequence": candidate.tolist(),
                "total_action_steps": int(candidate_steps), "outcome_label_status": "sealed_until_postfit_join",
            })
            outcomes[key] = {"real_task_cost": float(np.linalg.norm(terminal[:2] - goal[:2])), "realized_terminal_state": terminal.tolist()}
    source_enriched = []
    for row in records:
        root = str(row["instance_id"])
        if root not in split:
            continue
        history = history_by_root[root]
        copied = dict(row)
        copied.update({
            "split": split[root], "history_frame_indices": history["frames"],
            "history_action_interval": history["interval"], "history_actions_raw": history["actions"].tolist(),
        })
        source_enriched.append(copied)
    output.mkdir(parents=True, exist_ok=True)
    context_path = output / "temporal_ood_v2_context.json"
    outcome_path = output / "temporal_ood_v2_outcomes.json"
    source_path = output / "temporal_ood_v2_source_history_dataset.json"
    context_path.write_text(json.dumps({"metadata": {
        "schema": SCHEMA, "candidate_family": V2_ACTION_OOD_FAMILY,
        "root_split_seed": int(root_split_seed), "action_bank_seed": int(action_bank_seed),
        "history_windows": HISTORY_WINDOWS, "history_frame_stride": FRAME_STRIDE,
        "history_is_causal": True, "history_actions_are_strictly_pre_candidate": True,
        "outcomes_sealed_from_feature_construction": True,
        "v1_outcome_roots_excluded": sorted(v1_outcomes),
        "v1_action_hash_overlap": [], "v1_template_overlap": [],
    }, "records": contexts}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outcome_path.write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_path.write_text(json.dumps({"metadata": {
        **metadata, "schema": "pointmaze_temporal_ood_v2_source_history.v1", "history_windows": HISTORY_WINDOWS,
        "history_is_causal": True, "history_actions_are_strictly_pre_candidate": True,
        "root_split_seed": int(root_split_seed), "v1_outcome_roots_excluded": sorted(v1_outcomes),
    }, "records": source_enriched}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "schema": SCHEMA, "context": str(context_path.resolve()), "sealed_outcomes": str(outcome_path.resolve()),
        "temporal_source_dataset": str(source_path.resolve()), "root_split_seed": int(root_split_seed), "action_bank_seed": int(action_bank_seed),
        "root_counts": {name: sum(value == name for value in split.values()) for name in SPLIT_COUNTS},
        "v1_outcome_root_overlap": sorted(set(split) & v1_outcomes),
        "source_action_hash_overlap": sorted(set(hashes) & source_hashes), "v1_action_hash_overlap": sorted(set(hashes) & v1_hashes),
        "source_template_overlap": sorted(set(templates) & source_templates), "v1_template_overlap": sorted(set(templates) & v1_templates),
        "history_contract": "four observed pre-candidate frozen frames and all actions on the preceding three frame intervals",
    }
    (output / "build_v2_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--v1-context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-split-seed", type=int, default=DEFAULT_ROOT_SPLIT_SEED)
    parser.add_argument("--action-bank-seed", type=int, default=DEFAULT_ACTION_BANK_SEED)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--candidate-steps", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
