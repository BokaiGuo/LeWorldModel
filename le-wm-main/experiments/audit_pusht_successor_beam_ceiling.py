"""Privileged short-horizon beam ceiling for the PushT high-angle panel.

This audit asks whether the greedy successor-oracle ceiling is hiding a
short-horizon action-sequence solution.  It uses real successors and is
therefore an upper-bound diagnostic only; it is not visual-LeWM evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import stable_worldmodel  # noqa: F401

from experiments.run_pusht_mode_aware_shared_candidate import (
    angle_error,
    candidate_actions,
    make_boundary_contact_roots,
    pose_cost,
)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    torch.zeros(8, device=device).sum().item()
    return device


def successor(
    env: gym.Env, state: np.ndarray, goal: np.ndarray,
    action: np.ndarray, pulse_steps: int,
) -> np.ndarray:
    observation, _ = env.reset(
        seed=0, options={"state": state.copy(), "goal_state": goal.copy()}
    )
    current = np.asarray(observation["state"], dtype=np.float64).copy()
    for _ in range(int(pulse_steps)):
        observation, _, terminated, truncated, _ = env.step(
            np.asarray(action, dtype=np.float32)
        )
        current = np.asarray(observation["state"], dtype=np.float64).copy()
        if terminated or truncated:
            break
    return current


def beam_action(
    env: gym.Env, state: np.ndarray, goal: np.ndarray, args: argparse.Namespace,
) -> tuple[int, np.ndarray, float]:
    """Return the first action of the best real-successor action sequence."""

    bank = candidate_actions(
        env, state, tuple(args.tangent_magnitudes), float(args.inward_magnitude),
        goal=goal, boundary_action_frame=True,
        contact_normal_magnitudes=tuple(args.contact_normal_magnitudes),
    )
    frontier: list[tuple[np.ndarray, list[int], float]] = [(state.copy(), [], 0.0)]
    for _depth in range(int(args.horizon)):
        expanded: list[tuple[np.ndarray, list[int], float]] = []
        for node_state, sequence, _ in frontier:
            node_bank = candidate_actions(
                env, node_state, tuple(args.tangent_magnitudes),
                float(args.inward_magnitude), goal=goal,
                boundary_action_frame=True,
                contact_normal_magnitudes=tuple(args.contact_normal_magnitudes),
            )
            for index, action in enumerate(node_bank):
                next_state = successor(env, node_state, goal, action, args.pulse_steps)
                expanded.append((
                    next_state, sequence + [int(index)], pose_cost(next_state, goal),
                ))
        expanded.sort(key=lambda item: item[2])
        frontier = expanded[: int(args.beam_width)]
    if not frontier:
        raise RuntimeError("beam search produced no candidates")
    best_state, sequence, score = min(frontier, key=lambda item: item[2])
    first_index = int(sequence[0])
    # The candidate bank is state-dependent; recompute the first bank so the
    # returned index is interpreted against the actual current state.
    first_action = np.asarray(bank[first_index], dtype=np.float64)
    return first_index, first_action, float(score)


def run_episode(env: gym.Env, root: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    current = np.asarray(root["state"], dtype=np.float64).copy()
    goal = np.asarray(root["goal"], dtype=np.float64).copy()
    trace: list[dict[str, Any]] = []
    for step in range(int(args.steps)):
        if pose_cost(current, goal) < float(args.stop_cost):
            break
        before = current.copy()
        selected, action, predicted_score = beam_action(env, current, goal, args)
        current = successor(env, current, goal, action, args.pulse_steps)
        trace.append({
            "step": int(step), "selected": int(selected),
            "oracle_sequence_score": float(predicted_score),
            "angle_before": angle_error(before[4], goal[4]),
            "angle_after": angle_error(current[4], goal[4]),
            "xy_after": float(np.linalg.norm(current[2:4] - goal[2:4])),
            "action": action.tolist(),
        })
    return {
        "initial_angle": angle_error(root["state"][4], goal[4]),
        "final_angle": angle_error(current[4], goal[4]),
        "final_xy": float(np.linalg.norm(current[2:4] - goal[2:4])),
        "pose_success": bool(
            np.linalg.norm(current[2:4] - goal[2:4]) < 20.0
            and angle_error(current[4], goal[4]) < np.pi / 9.0
        ),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261059)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--pulse-steps", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--stop-cost", type=float, default=1.0)
    parser.add_argument("--inward-magnitude", type=float, default=0.08)
    parser.add_argument("--tangent-magnitudes", type=float, nargs="+",
                        default=[0.04, 0.08, 0.12, 0.16, 0.24, 0.32])
    parser.add_argument("--contact-normal-magnitudes", type=float, nargs="+",
                        default=[0.02, 0.04, 0.08, 0.12, 0.16])
    args = parser.parse_args()
    device = resolve_device(args.device)
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        roots = make_boundary_contact_roots(env, args.episodes, args.seed + 2000)
        rows = [run_episode(env, root, args) for root in roots]
    finally:
        env.close()
    report: dict[str, Any] = {
        "schema": "pusht_successor_beam_ceiling.v1",
        "protocol": {
            "environment": "swm/PushT-v1", "resolved_device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "root_seed": args.seed + 2000, "episodes": len(rows),
            "steps": args.steps, "pulse_steps": args.pulse_steps,
            "horizon": args.horizon, "beam_width": args.beam_width,
            "runtime_successor_access": True,
        },
        "metrics": {
            "pose_success": float(np.mean([row["pose_success"] for row in rows])),
            "mean_final_xy": float(np.mean([row["final_xy"] for row in rows])),
            "mean_final_angle": float(np.mean([row["final_angle"] for row in rows])),
            "mean_initial_angle": float(np.mean([row["initial_angle"] for row in rows])),
        },
        "rows": rows,
        "claim_boundary": (
            "Privileged real-successor beam upper bound only; not a deployable "
            "visual policy and not evidence of frozen-LeWM causal dynamics."
        ),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"output": str(args.output), "device": str(device), "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
