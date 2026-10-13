"""Audit the RGB torque-sign interface against independent PushT successors.

This is a read-only support audit.  It does not promote a controller and does
not reuse learned adapter predictions.  For each root/action pair it compares
the image-derived boundary torque sign with the true one-step angular effect
measured from an independently reset simulator successor.  The output is
intended to distinguish a coordinate-sign failure from a ranker/support
failure before changing the runtime controller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from run_causal_visual_pusht_world_model import balanced_source_cases, terminal_pose_cases
from run_rgb_local_effect_adapter import (
    PUSHT_STATE_SIZE,
    action_bank,
    block_pose,
    contact_boundary_chart_from_images,
    goal_observation_image,
    theta_from_pose,
    wrap,
)


def _step_from_root(env: gym.Env, state: np.ndarray, goal: np.ndarray, action: np.ndarray) -> np.ndarray:
    env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
    observation, _reward, terminated, truncated, _info = env.step(
        np.asarray(action, dtype=np.float32),
    )
    if terminated or truncated:
        raise RuntimeError("one-step torque audit terminated unexpectedly")
    return np.asarray(observation["state"], dtype=np.float32)


def _raw_block_pose(state: np.ndarray) -> np.ndarray:
    """Read a PushT environment state before conversion to the model chart.

    ``terminal_pose_cases`` returns the environment's raw state, whose
    orientation is the scalar angle at index 4.  The learned-model
    ``observable_pose_vector`` instead stores ``sin(theta), cos(theta)`` at
    indices 4 and 5.  Mixing these two charts made the previous audit report
    many terminal roots as having zero angle error and invalidated its sign
    statistics.  This helper keeps the audit in the raw simulator chart until
    the explicit RGB boundary comparison below.
    """
    values = np.asarray(state, dtype=np.float32).reshape(-1)
    if values.shape[0] < 5:
        raise ValueError("raw PushT state must contain x, y, block x, block y, angle")
    return np.asarray([values[2], values[3], values[4]], dtype=np.float32)


def _summarize(rows: list[dict[str, float]]) -> dict[str, float | int | None]:
    if not rows:
        return {"rows": 0}
    values = np.asarray(rows, dtype=np.float64)
    improvement = values[:, 0]
    torque = values[:, 1]
    desired = values[:, 2]
    compatible = values[:, 3].astype(bool)
    improved = improvement > 1.0e-4
    # ``torque`` is the image-chart r x F under the renderer's y-down chart.
    # We report both hypotheses explicitly; the correct physical convention
    # should have a materially larger improvement rate among compatible rows.
    # A zero angle error is a no-op for torque control.  Including those rows
    # makes both sign hypotheses appear equally good because ``torque * 0`` is
    # simultaneously nonnegative and nonpositive.  Keep them in the raw row
    # count, but exclude them from sign discrimination.
    informative = np.abs(desired) >= 0.20
    same_sign = informative & (torque * desired >= 0.0)
    opposite_sign = informative & (torque * desired <= 0.0)
    return {
        "rows": int(len(rows)),
        "compatible_rows": int(np.sum(compatible)),
        "improved_rows": int(np.sum(improved)),
        "informative_rows": int(np.sum(informative)),
        "same_sign_rows": int(np.sum(same_sign)),
        "opposite_sign_rows": int(np.sum(opposite_sign)),
        "same_sign_improvement_rate": float(np.mean(improved[same_sign])) if np.any(same_sign) else None,
        "opposite_sign_improvement_rate": float(np.mean(improved[opposite_sign])) if np.any(opposite_sign) else None,
        "same_sign_contact_improvement_rate": float(np.mean(improved[same_sign & compatible]))
        if np.any(same_sign & compatible) else None,
        "opposite_sign_contact_improvement_rate": float(np.mean(improved[opposite_sign & compatible]))
        if np.any(opposite_sign & compatible) else None,
        "mean_improvement": float(np.mean(improvement)),
        "mean_improvement_same_sign": float(np.mean(improvement[same_sign])) if np.any(same_sign) else None,
        "mean_improvement_opposite_sign": float(np.mean(improvement[opposite_sign])) if np.any(opposite_sign) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=9173)
    parser.add_argument("--balanced-roots", type=int, default=24)
    parser.add_argument("--terminal-roots", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=80)
    try:
        roots = balanced_source_cases(
            env, args.balanced_roots, max(1, args.balanced_roots // 3),
            args.seed + 10_000, 0.6, 48,
        )
        roots.extend(terminal_pose_cases(
            env, args.terminal_roots, args.seed + 410_000, position_jitter=16.0,
        ))
        actions = action_bank(16, [0.08, 0.16, 0.24, 0.35])
        by_split: dict[str, list[dict[str, float]]] = {"all": [], "contact": [], "terminal": []}
        root_summaries: list[dict[str, object]] = []
        for root_id, (state, goal, image, _previous, _goal_image, _previous_action, contact) in enumerate(roots):
            current = _raw_block_pose(state)
            goal_pose = _raw_block_pose(goal)
            zero_state = _step_from_root(env, state, goal, actions[0])
            zero = block_pose(zero_state)
            image_chw = np.moveaxis(np.asarray(image, dtype=np.uint8), -1, 0)[None]
            chart = contact_boundary_chart_from_images(
                np.repeat(image_chw, len(actions), axis=0), actions,
            )
            root_rows: list[dict[str, float]] = []
            for index, action in enumerate(actions):
                successor = block_pose(_step_from_root(env, state, goal, action))
                desired = float(wrap(goal_pose[2] - current[2]))
                zero_error = abs(float(wrap(zero[2] - goal_pose[2])))
                action_error = abs(float(wrap(successor[2] - goal_pose[2])))
                row = {
                    "angle_improvement": float(zero_error - action_error),
                    "signed_torque": float(chart[index, 9]),
                    "desired_angle_error": desired,
                    "contact_compatible": float(np.linalg.norm(successor[:2] - current[:2]) <= 40.0),
                    "xy_displacement": float(np.linalg.norm(successor[:2] - zero[:2])),
                    "angle_effect": float(wrap(successor[2] - zero[2])),
                    "action_x": float(action[0]),
                    "action_y": float(action[1]),
                }
                root_rows.append(row)
                by_split["all"].append(row)
                if contact:
                    by_split["contact"].append(row)
                if root_id >= args.balanced_roots:
                    by_split["terminal"].append(row)
            root_summaries.append({
                "root": root_id,
                "contact": bool(contact),
                "initial_angle": float(current[2]),
                "goal_angle": float(goal_pose[2]),
                "zero_angle_error": float(abs(float(wrap(zero[2] - goal_pose[2])))),
                "summary": _summarize([
                    [row["angle_improvement"], row["signed_torque"], row["desired_angle_error"], row["contact_compatible"]]
                    for row in root_rows
                ]),
            })
        report = {
            "protocol": {
                "seed": args.seed,
                "balanced_roots": args.balanced_roots,
                "terminal_roots": args.terminal_roots,
                "actions": actions.tolist(),
                "label": "one-step simulator successor; RGB boundary torque only for sign; no learned adapter",
            },
            "summary": {key: _summarize([
                [row["angle_improvement"], row["signed_torque"], row["desired_angle_error"], row["contact_compatible"]]
                for row in rows
            ]) for key, rows in by_split.items()},
            "root_summaries": root_summaries,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
