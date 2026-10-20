"""Privileged successor-oracle ceiling for the PushT high-angle panel.

At every pulse this audit evaluates the *real* successor of every declared
candidate and executes the candidate with the lowest observed task cost.  It
is an upper-bound diagnosis only: it is not an admissible visual controller
and must never be used as evidence for frozen-LeWM causal dynamics.
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


def successor(env: gym.Env, state: np.ndarray, goal: np.ndarray, action: np.ndarray, pulse_steps: int) -> np.ndarray:
    observation, _ = env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
    current = np.asarray(observation["state"], dtype=np.float64).copy()
    for _ in range(int(pulse_steps)):
        observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        current = np.asarray(observation["state"], dtype=np.float64).copy()
        if terminated or truncated:
            break
    return current


def run_episode(env: gym.Env, root: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    current = np.asarray(root["state"], dtype=np.float64).copy()
    goal = np.asarray(root["goal"], dtype=np.float64).copy()
    trace: list[dict[str, Any]] = []
    for step in range(int(args.steps)):
        if pose_cost(current, goal) < float(args.stop_cost):
            break
        actions = candidate_actions(
            env, current, tuple(args.tangent_magnitudes), float(args.inward_magnitude),
            goal=goal, boundary_action_frame=True,
            contact_normal_magnitudes=tuple(args.contact_normal_magnitudes),
        )
        outcomes = [successor(env, current, goal, action, args.pulse_steps) for action in actions]
        scores = np.asarray([pose_cost(outcome, goal) for outcome in outcomes], dtype=np.float64)
        selected = int(np.argmin(scores))
        # Re-execute only the selected action from the actual current state.
        before = current.copy()
        current = successor(env, current, goal, actions[selected], args.pulse_steps)
        trace.append({
            "step": int(step),
            "selected": selected,
            "candidate_count": int(len(actions)),
            "oracle_score": float(scores[selected]),
            "best_candidate_score": float(scores.min()),
            "angle_before": angle_error(before[4], goal[4]),
            "angle_after": angle_error(current[4], goal[4]),
            "xy_after": float(np.linalg.norm(current[2:4] - goal[2:4])),
            "action": np.asarray(actions[selected]).tolist(),
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
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--pulse-steps", type=int, default=2)
    parser.add_argument("--stop-cost", type=float, default=1.0)
    parser.add_argument("--inward-magnitude", type=float, default=0.08)
    parser.add_argument("--tangent-magnitudes", type=float, nargs="+", default=[0.06, 0.10, 0.16])
    parser.add_argument("--contact-normal-magnitudes", type=float, nargs="+", default=[0.02, 0.04, 0.08])
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
    report = {
        "schema": "pusht_successor_oracle_ceiling.v1",
        "protocol": {
            "environment": "swm/PushT-v1", "resolved_device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "root_seed": args.seed + 2000, "episodes": len(rows),
            "steps": args.steps, "pulse_steps": args.pulse_steps,
            "candidate_bank": "same boundary-frame bank with 0.02/0.04/0.08 normal magnitudes",
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
            "Privileged real-successor upper bound only; it is not a deployable "
            "visual policy and cannot establish frozen-LeWM causal dynamics."
        ),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"output": str(args.output), "device": str(device), "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
