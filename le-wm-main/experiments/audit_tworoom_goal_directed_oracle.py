"""Measure a privileged goal-directed candidate ceiling for TwoRoom.

This experiment is intentionally evaluator-side.  It uses the simulator's
source and goal positions to construct a fixed, root-local action bank, then
replays every candidate from the same root.  No RGB observation, latent, or
learned cost is used to select a candidate.  The result answers only whether
the low-level candidate family can reach the goal; it is not an online policy
and must not be interpreted as world-model evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import gymnasium as gym
import numpy as np

from experiments.run_live_tworoom_planning import TwoRoomGoalAliasWrapper


def _directions() -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def _candidate_bank(
    source: np.ndarray,
    goal: np.ndarray,
    *,
    horizon: int,
    action_block: int,
    magnitudes: tuple[float, ...],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if horizon < 2 or action_block < 1:
        raise ValueError("horizon must be at least two and action_block positive")
    source = np.asarray(source, dtype=np.float64).reshape(-1)
    goal = np.asarray(goal, dtype=np.float64).reshape(-1)
    if source.size != 2 or goal.size != 2:
        raise ValueError("TwoRoom oracle expects 2-D source and goal positions")
    displacement = goal - source
    if np.linalg.norm(displacement) < 1.0e-8:
        raise ValueError("source and goal must be distinct")
    goal_angle = math.atan2(float(displacement[1]), float(displacement[0]))
    base_angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    directed_offsets = np.asarray(
        [0.0, np.pi / 8.0, -np.pi / 8.0, np.pi / 4.0, -np.pi / 4.0,
         np.pi / 2.0, -np.pi / 2.0, np.pi],
        dtype=np.float64,
    )
    angles = np.concatenate([base_angles, goal_angle + directed_offsets])
    angles = np.unique(np.round(np.mod(angles, 2.0 * np.pi), 10))
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    candidates: list[np.ndarray] = []
    labels: list[dict[str, Any]] = []

    def add(sequence: np.ndarray, label: dict[str, Any]) -> None:
        candidates.append(np.clip(sequence, -1.0, 1.0).astype(np.float32))
        labels.append(label)

    add(np.zeros((horizon * action_block, 2)), {"family": "zero"})
    for magnitude in magnitudes:
        for angle, direction in zip(angles, directions):
            sequence = np.tile(direction * magnitude, (horizon * action_block, 1))
            add(sequence, {"family": "constant", "magnitude": float(magnitude), "angle": float(angle)})

    # A two-stage bank separates initial alignment from goal-directed motion.
    # The first half uses a root-local approach direction; the second half uses
    # a direction close to the source-to-goal displacement.
    goal_direction = displacement / np.linalg.norm(displacement)
    split = max(1, (horizon * action_block) // 2)
    for magnitude in magnitudes:
        for first_angle, first_direction in zip(angles, directions):
            for second_scale in (0.5, 1.0):
                sequence = np.empty((horizon * action_block, 2), dtype=np.float64)
                sequence[:split] = first_direction * magnitude
                sequence[split:] = goal_direction * magnitude * second_scale
                add(sequence, {
                    "family": "two_stage_goal_directed",
                    "magnitude": float(magnitude),
                    "first_angle": float(first_angle),
                    "second_scale": float(second_scale),
                })
    return np.stack(candidates), labels


def _replay(env: gym.Env, seed: int, sequence: np.ndarray) -> tuple[float, bool]:
    env.reset(seed=int(seed))
    final_info: dict[str, Any] = {}
    terminated = truncated = False
    for action in np.asarray(sequence, dtype=np.float32):
        _, _, terminated, truncated, final_info = env.step(action)
        if terminated or truncated:
            break
    distance = float(np.asarray(final_info.get("distance_to_target", np.inf)).reshape(-1)[0])
    return distance, bool(terminated)


def run(*, output: Path, seed: int, roots: int, horizon: int, action_block: int,
        magnitudes: tuple[float, ...]) -> dict[str, Any]:
    if roots < 1:
        raise ValueError("roots must be positive")
    env = TwoRoomGoalAliasWrapper(gym.make("swm/TwoRoom-v1", render_mode="rgb_array"))
    summaries: list[dict[str, Any]] = []
    try:
        for index in range(int(roots)):
            root_seed = int(seed + index)
            _, info = env.reset(seed=root_seed)
            candidates, labels = _candidate_bank(
                np.asarray(info["state"], dtype=np.float64),
                np.asarray(info["goal_state"], dtype=np.float64),
                horizon=horizon,
                action_block=action_block,
                magnitudes=magnitudes,
            )
            costs = np.empty(len(candidates), dtype=np.float64)
            successes = np.zeros(len(candidates), dtype=np.int8)
            for candidate_index, sequence in enumerate(candidates):
                costs[candidate_index], success = _replay(env, root_seed, sequence)
                successes[candidate_index] = int(success)
            summaries.append({
                "seed": root_seed,
                "candidate_count": int(len(candidates)),
                "success_count": int(successes.sum()),
                "oracle_available": bool(successes.any()),
                "oracle_best_cost": float(costs.min()),
                "oracle_best_index": int(np.argmin(costs)),
                "oracle_best_label": labels[int(np.argmin(costs))],
            })
            print(
                f"[tworoom-oracle] root={index + 1}/{roots} seed={root_seed} "
                f"candidates={len(candidates)} success={int(successes.sum())} "
                f"best={float(costs.min()):.4f}", flush=True,
            )
    finally:
        env.close()
    report = {
        "schema": "tworoom_goal_directed_candidate_oracle.v0.1",
        "status": "completed_privileged_reachability_audit",
        "protocol": {
            "environment": "TwoRoom",
            "roots": int(roots),
            "seed_start": int(seed),
            "horizon": int(horizon),
            "action_block": int(action_block),
            "magnitudes": list(magnitudes),
            "runtime_policy_input": "none; simulator state used only to construct evaluator-side candidates",
            "learned_model_used": False,
            "privileged_state_role": "candidate-generation and outcome labels only",
        },
        "metrics": {
            "root_count": len(summaries),
            "oracle_available_roots": int(sum(item["oracle_available"] for item in summaries)),
            "oracle_available_rate": float(np.mean([item["oracle_available"] for item in summaries])),
            "zero_success_roots": int(sum(item["success_count"] == 0 for item in summaries)),
            "mean_success_count": float(np.mean([item["success_count"] for item in summaries])),
        },
        "root_summaries": summaries,
        "claim_boundary": (
            "privileged candidate reachability ceiling only; not visual planning, "
            "learned dynamics, or a deployable controller"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20264200)
    parser.add_argument("--roots", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--magnitudes", default="0.25,0.5,0.75,1.0")
    args = parser.parse_args()
    magnitudes = tuple(float(item) for item in args.magnitudes.split(",") if item.strip())
    if not magnitudes or any(value <= 0.0 for value in magnitudes):
        raise ValueError("--magnitudes must contain positive values")
    report = run(
        output=args.output, seed=args.seed, roots=args.roots,
        horizon=args.horizon, action_block=args.action_block,
        magnitudes=magnitudes,
    )
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
