"""Root-disjoint RGB macro-action ranking audit for OGB-Cube.

The one-step Cube ranking audit has a deliberate limitation: at the decision
point, most candidate actions do not yet move the cube, while the deployable
effect requires an approach/close/push sequence.  This experiment tests that
specific hypothesis with a fixed, task-agnostic macro bank.  The readout sees
current/previous RGB, the rendered goal, and a summary of the whole candidate
sequence.  Privileged state is used only for branch construction and terminal
labels; it is never an input to the ranker.

This is a guided contact-support mechanism audit, not pure RGB online
planning evidence.  A positive result would justify a later closed-loop
integration; a negative result would localize failure beyond one-step action
granularity.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import stable_worldmodel  # noqa: F401 - registers environments

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_cube_rgb_observability import (
    _goal_position,
    _largest_red_centroid,
    _make_env,
    _vec,
    extract_color_geometry_temporal_features,
    fit_ridge,
)
from experiments.audit_cube_temporal_action_ranking import _capture_branch_point


def _macro_bank(
    length: int = 20, direction_count: int = 4,
    approach_steps: int | list[int] | tuple[int, ...] = 4,
) -> np.ndarray:
    """Return a fixed zero plus a predeclared directional macro bank."""
    if length < 20:
        raise ValueError("macro length must be at least 20")
    if direction_count not in (4, 8, 16):
        raise ValueError("direction_count must be one of 4, 8, or 16")
    durations = [int(approach_steps)] if isinstance(approach_steps, (int, np.integer)) else [int(v) for v in approach_steps]
    if not durations or any(v < 1 or v >= length - 8 for v in durations):
        raise ValueError("approach_steps must leave room for descent, close, and push")
    if direction_count == 4:
        directions = np.asarray(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            dtype=np.float32,
        )
    else:
        angles = np.arange(direction_count, dtype=np.float32) * (2.0 * np.pi / direction_count)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    bank = [np.zeros((length, 5), dtype=np.float32)]
    for duration in durations:
        for direction in directions:
            sequence = np.zeros((length, 5), dtype=np.float32)
            if length == 20 and duration == 4:
                # Preserve the original short-horizon mechanism protocol exactly.
                approach_end, descend_end, close_end = 4, 10, 14
            else:
                # Longer macros are used only for deployable initial-state
                # support: cover a short lateral pulse, then descend and close.
                approach_end = min(length - 8, duration)
                descend_end = min(length - 8, approach_end + max(6, length // 4))
                close_end = min(length - 2, descend_end + max(4, length // 12))
            sequence[:approach_end, :2] = direction
            sequence[approach_end:descend_end, 2] = -1.0
            sequence[descend_end:close_end, 4] = -1.0
            sequence[close_end:, :2] = direction
            bank.append(sequence)
    return np.stack(bank, axis=0)


def _macro_summary(bank: np.ndarray) -> np.ndarray:
    bank = np.asarray(bank, dtype=np.float32)
    # A single whole-sequence mean aliases two physically different phases:
    # approaching the object and pushing it after contact.  Keep the legacy
    # summary fields, but add phase-local XY summaries so a factorized bank
    # can be represented without relying on candidate IDs.
    # Offline historical macros encode close as -1 while the deployable
    # online protocol uses +1; both denote the phase boundary here.
    close = np.abs(bank[:, :, 4]) > 0.5
    push_start = np.asarray([
        int(np.flatnonzero(row)[-1]) + 1 if np.any(row) else bank.shape[1]
        for row in close
    ], dtype=np.int64)
    approach_end = np.minimum(push_start, max(1, bank.shape[1] // 2))
    approach = np.asarray([
        row[: int(end), :2].mean(axis=0) if int(end) > 0 else np.zeros(2, dtype=np.float32)
        for row, end in zip(bank, approach_end)
    ], dtype=np.float32)
    push = np.asarray([
        row[int(start):, :2].mean(axis=0) if int(start) < bank.shape[1] else np.zeros(2, dtype=np.float32)
        for row, start in zip(bank, push_start)
    ], dtype=np.float32)
    return np.concatenate(
        [bank.mean(axis=1), bank[:, -1, :], np.mean(np.abs(bank), axis=1), approach, push], axis=1,
    ).astype(np.float32)


def _rgb_action_geometry_interactions(
    branches: list[dict[str, Any]], bank: np.ndarray, device: str,
) -> torch.Tensor:
    """Return fixed RGB-goal/action bilinear features.

    Cube actions are expressed in world coordinates while the red-object
    chart is expressed in image coordinates.  A linear readout over the two
    feature groups cannot represent their camera/action-frame transform.  The
    outer product below is a predeclared interaction, not an outcome-tuned
    feature: it lets calibration learn the fixed local map from image-space
    goal displacement to world-space macro direction.
    """
    current_xy = torch.as_tensor(
        np.asarray([_largest_red_centroid(b["current_image"])[:2] for b in branches], dtype=np.float32),
        device=device,
    )
    goal_xy = torch.as_tensor(
        np.asarray([_largest_red_centroid(b["goal_image"])[:2] for b in branches], dtype=np.float32),
        device=device,
    )
    screen_delta = goal_xy - current_xy
    directions = torch.as_tensor(
        np.asarray([sequence[:, :2].mean(axis=0) for sequence in bank], dtype=np.float32),
        device=device,
    )
    # [root, candidate, screen_axis, world_action_axis].
    outer = screen_delta[:, None, :, None] * directions[None, :, None, :]
    outer = outer.reshape(len(branches) * len(bank), -1)
    dot = (screen_delta[:, None, :] * directions[None, :, :]).sum(dim=-1, keepdim=True)
    cross = (
        screen_delta[:, None, 0] * directions[None, :, 1]
        - screen_delta[:, None, 1] * directions[None, :, 0]
    ).unsqueeze(-1)
    return torch.cat([
        outer,
        dot.reshape(len(branches) * len(bank), 1),
        cross.reshape(len(branches) * len(bank), 1),
    ], dim=1)


def _branch_macro_effects(branch: dict[str, Any], bank: np.ndarray) -> np.ndarray:
    env = _make_env()
    try:
        effects = []
        for sequence in bank:
            _, info = env.reset(
                seed=int(branch["seed"]),
                options={
                    "task_id": int(branch["task_id"]),
                    "render_goal": True,
                    "variation_values": copy.deepcopy(branch["variation_values"]),
                    "state": np.asarray(branch["state"], dtype=np.float64),
                },
            )
            for action in sequence:
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            block = _vec(info, "privileged/block_0_pos", 3)[:2]
            effects.append(np.asarray(branch["goal"], dtype=np.float32) - block)
        return np.asarray(effects, dtype=np.float32)
    finally:
        env.close()


def _features(branches: list[dict[str, Any]], bank: np.ndarray, device: str) -> torch.Tensor:
    n = len(branches)
    current = np.repeat(np.asarray([b["current_image"] for b in branches]), len(bank), axis=0)
    previous = np.repeat(np.asarray([b["previous_image"] for b in branches]), len(bank), axis=0)
    goals = np.repeat(np.asarray([b["goal_image"] for b in branches]), len(bank), axis=0)
    previous_actions = np.repeat(
        np.asarray([b["previous_action"] for b in branches]), len(bank), axis=0,
    )
    actions = np.tile(bank.mean(axis=1), (n, 1))
    # The geometry helper expects a 5-D current candidate action.  Feed the
    # macro mean there, then append the fixed whole-sequence summary.
    base = extract_color_geometry_temporal_features(
        current, previous, goals, actions, previous_actions, device,
    )["color_geometry_temporal_action"]
    summary = torch.as_tensor(np.tile(_macro_summary(bank), (n, 1)), device=device)
    interactions = _rgb_action_geometry_interactions(branches, bank, device)
    return torch.cat([base, summary, interactions], dim=1)


def _fit(features: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    calibration = np.ones(features.shape[0], dtype=bool)
    _, mean, scale, weight = fit_ridge(features, target, calibration, ridge=1.0e-2)
    return mean, scale, weight


def _predict(features: torch.Tensor, fit: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> np.ndarray:
    mean, scale, weight = fit
    values = ((features.to(torch.float64) - mean) / scale.clamp_min(1.0e-4)) @ weight
    return values.detach().cpu().numpy()


def _rank_panel(
    branches: list[dict[str, Any]], bank: np.ndarray,
    fit: tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: str,
) -> list[dict[str, Any]]:
    features = _features(branches, bank, device)
    records = []
    for index, branch in enumerate(branches):
        start = index * len(bank)
        stop = start + len(bank)
        predicted = _predict(features[start:stop], fit)
        observed = _branch_macro_effects(branch, bank)
        actual = np.linalg.norm(observed, axis=1)
        predicted_scores = np.linalg.norm(predicted, axis=1)
        oracle = int(np.argmin(actual))
        selected = int(np.argmin(predicted_scores))
        records.append({
            "root_index": int(index),
            "seed": int(branch["seed"]),
            "task_id": int(branch["task_id"]),
            "candidate_count": int(len(bank)),
            "oracle_index": oracle,
            "selected_index": selected,
            "actual_candidate_error_m": actual.tolist(),
            "predicted_candidate_error_m": predicted_scores.tolist(),
            "oracle_error_m": float(actual[oracle]),
            "selected_error_m": float(actual[selected]),
            "regret_m": float(actual[selected] - actual[oracle]),
            "oracle_agreement": bool(actual[selected] <= actual[oracle] + 1.0e-6),
        })
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261150)
    parser.add_argument("--calibration-roots", type=int, default=40)
    parser.add_argument("--held-out-roots", type=int, default=32)
    parser.add_argument("--decision-frame", type=int, default=20)
    parser.add_argument("--macro-length", type=int, default=20)
    parser.add_argument("--direction-count", type=int, default=4)
    parser.add_argument("--approach-steps", type=int, nargs="+", default=[4])
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for Cube macro ranking but unavailable")
    if args.calibration_roots < 8 or args.held_out_roots < 4:
        raise ValueError("need at least 8 calibration roots and 4 held-out roots")
    bank = _macro_bank(args.macro_length, args.direction_count, args.approach_steps)
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        "[purpose] Cube fixed macro-action ranking; "
        f"device={args.device} calibration={args.calibration_roots} held_out={args.held_out_roots}"
    )
    calibration = [
        _capture_branch_point(
            seed=args.seed + index,
            task_id=int(args.tasks[index % len(args.tasks)]),
            decision_frame=args.decision_frame,
        )
        for index in range(args.calibration_roots)
    ]
    held_out = [
        _capture_branch_point(
            seed=args.seed + 1000 + index,
            task_id=int(args.tasks[index % len(args.tasks)]),
            decision_frame=args.decision_frame,
        )
        for index in range(args.held_out_roots)
    ]
    calibration_features = _features(calibration, bank, args.device)
    calibration_targets = torch.as_tensor(
        np.concatenate([_branch_macro_effects(branch, bank) for branch in calibration], axis=0),
        device=args.device, dtype=torch.float32,
    )
    fit = _fit(calibration_features, calibration_targets)
    records = _rank_panel(held_out, bank, fit, args.device)
    regrets = np.asarray([row["regret_m"] for row in records], dtype=np.float64)
    report = {
        "schema": "cube_macro_action_ranking.v1",
        "protocol": {
            "environment": "swm/OGBCube-v0",
            "device": args.device,
            "calibration_roots": args.calibration_roots,
            "held_out_roots": args.held_out_roots,
            "root_disjoint": True,
            "checkpoint_frozen": "not used; RGB geometry adapter control",
            "decision_frame": args.decision_frame,
            "macro_length": args.macro_length,
            "direction_count": args.direction_count,
            "approach_steps": args.approach_steps,
            "candidate_bank": bank.tolist(),
            "candidate_bank_task_agnostic": True,
            "guided_branch_prep": True,
            "runtime_inputs": "current/previous RGB, rendered goal, fixed macro summary",
            "privileged_state_role": "branch construction and terminal labels only",
            "fit": "calibration roots only",
        },
        "calibration": {"roots": args.calibration_roots, "feature_dim": int(calibration_features.shape[1])},
        "held_out": {
            "roots": len(records),
            "records": records,
            "mean_regret_m": float(regrets.mean()) if regrets.size else None,
            "oracle_agreement_rate": float(np.mean([r["oracle_agreement"] for r in records])) if records else None,
        },
        "claim_boundary": (
            "Guided RGB macro-action mechanism audit; not pure RGB online planning "
            "evidence and not a causal latent-dynamics claim."
        ),
    }
    (args.output / "macro_ranking.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["held_out"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
