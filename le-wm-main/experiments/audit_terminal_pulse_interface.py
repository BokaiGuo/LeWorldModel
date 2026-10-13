"""Audit terminal PushT pulses against independent simulator successors.

This script is intentionally outside the learned-adapter runtime.  It tests
the action interface that the runtime can construct from RGB geometry, while
the simulator is used only to score one fixed pulse from an independently
reset root.  The audit separates support-lever and centroid-lever bases,
both torque conventions, and pulse magnitudes/inward components.  It is used
to decide whether the negative terminal result is caused by sign, lever-arm
choice, insufficient excitation, or contact-preservation failure.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np

from run_causal_visual_pusht_world_model import terminal_pose_cases
from run_rgb_local_effect_adapter import (
    PUSHT_STATE_SIZE,
    contact_boundary_chart_from_images,
    resize_image,
    rgb_geometry_xy_angle_from_image,
    wrap,
)


def raw_block_pose(state: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype=np.float32).reshape(-1)
    return np.asarray([values[2], values[3], values[4]], dtype=np.float32)


def step_root(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    action: np.ndarray,
    repeat: int,
) -> np.ndarray:
    env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
    observation = None
    for _ in range(int(repeat)):
        observation, _reward, terminated, truncated, _info = env.step(
            np.asarray(action, dtype=np.float32),
        )
        if terminated or truncated:
            # A sufficiently effective pulse can already enter the simulator
            # success condition during a repeated-action audit.  The final
            # observation remains a valid successor for scoring; stop the
            # repeated application without discarding that positive case.
            break
    if observation is None:
        raise RuntimeError("terminal pulse audit received no observation")
    return np.asarray(observation["state"], dtype=np.float32)


def make_pulse(
    lever: np.ndarray,
    desired_sign: float,
    tangent_magnitude: float,
    inward_magnitude: float,
    sign_flip: bool,
) -> np.ndarray:
    lever = np.asarray(lever, dtype=np.float32).reshape(2)
    lever = lever / max(float(np.linalg.norm(lever)), 1.0e-6)
    tangent = np.asarray([-lever[1], lever[0]], dtype=np.float32)
    inward = -lever
    sign = -float(desired_sign) if sign_flip else float(desired_sign)
    return (
        float(inward_magnitude) * inward
        + sign * float(tangent_magnitude) * tangent
    ).astype(np.float32)


def summarize(rows: list[dict[str, float]]) -> dict[str, float | int | None]:
    if not rows:
        return {"rows": 0}
    values = rows
    angle = np.asarray([row["angle_improvement"] for row in values], dtype=np.float64)
    xy = np.asarray([row["xy_regression"] for row in values], dtype=np.float64)
    safe = np.asarray([row["xy_safe"] for row in values], dtype=bool)
    retained = np.asarray([row["contact_retained"] for row in values], dtype=bool)
    return {
        "rows": int(len(values)),
        "mean_angle_improvement": float(np.mean(angle)),
        "median_angle_improvement": float(np.median(angle)),
        "positive_angle_rate": float(np.mean(angle > 1.0e-3)),
        "mean_xy_regression": float(np.mean(xy)),
        "xy_safe_rate": float(np.mean(safe)),
        "contact_retention_rate": float(np.mean(retained)),
        "safe_positive_angle_rate": float(np.mean(angle[safe] > 1.0e-3))
        if np.any(safe) else None,
        "safe_mean_angle_improvement": float(np.mean(angle[safe]))
        if np.any(safe) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=9173)
    parser.add_argument("--terminal-roots", type=int, default=24)
    parser.add_argument("--position-jitter", type=float, default=16.0)
    parser.add_argument("--support-gap-max", type=float, default=15.0)
    parser.add_argument("--support-angle-samples", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action_repeat < 1 or args.terminal_roots < 1:
        raise ValueError("action-repeat and terminal-roots must be positive")

    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=80)
    rows: list[dict[str, float]] = []
    try:
        roots = terminal_pose_cases(
            env, args.terminal_roots, args.seed + 410_000,
            position_jitter=args.position_jitter,
            support_gap_max=args.support_gap_max,
            support_angle_samples=args.support_angle_samples,
        )
        for root_id, (state, goal, image, *_rest) in enumerate(roots):
            current = raw_block_pose(state)
            target = raw_block_pose(goal)
            rgb_xy, rgb_angle = rgb_geometry_xy_angle_from_image(image, "block")
            if rgb_xy is None or rgb_angle is None:
                continue
            # ``resize_image`` already returns CHW; the chart function
            # deliberately expects NCHW, matching the runtime call.
            resized = resize_image(image, args.image_size)
            nchw = resized[None]
            zero_action = np.zeros(2, dtype=np.float32)
            zero_state = step_root(env, state, goal, zero_action, args.action_repeat)
            zero_pose = raw_block_pose(zero_state)
            zero_error = abs(float(wrap(zero_pose[2] - target[2])))
            desired_sign_true = 1.0 if float(wrap(target[2] - current[2])) >= 0.0 else -1.0
            desired_sign_rgb = 1.0 if float(wrap(target[2] - rgb_angle)) >= 0.0 else -1.0
            chart_zero = contact_boundary_chart_from_images(
                nchw, zero_action[None],
            )[0]
            support_lever = chart_zero[2:4].copy()
            centroid_lever = np.asarray(
                [float(rgb_xy[0] - current[0]), float(rgb_xy[1] - current[1])],
                dtype=np.float32,
            )
            if float(np.linalg.norm(centroid_lever)) < 1.0e-6:
                continue
            for lever_name, lever in (("support", support_lever), ("centroid", centroid_lever)):
                for sign_source, desired_sign in (("true", desired_sign_true), ("rgb", desired_sign_rgb)):
                    for sign_flip in (False, True):
                        for tangent_magnitude in (0.04, 0.08, 0.12, 0.16):
                            for inward_magnitude in (0.0, 0.04, 0.08):
                                action = make_pulse(
                                    lever, desired_sign, tangent_magnitude,
                                    inward_magnitude, sign_flip,
                                )
                                successor = step_root(
                                    env, state, goal, action, args.action_repeat,
                                )
                                pose = raw_block_pose(successor)
                                error = abs(float(wrap(pose[2] - target[2])))
                                agent_distance = float(np.linalg.norm(
                                    successor[:2] - successor[2:4]
                                ))
                                rows.append({
                                    "root": float(root_id),
                                    "lever": lever_name,
                                    "sign_source": sign_source,
                                    "sign_flip": float(sign_flip),
                                    "tangent_magnitude": float(tangent_magnitude),
                                    "inward_magnitude": float(inward_magnitude),
                                    "desired_sign_true": float(desired_sign_true),
                                    "desired_sign_rgb": float(desired_sign_rgb),
                                    "rgb_angle_error": float(wrap(target[2] - rgb_angle)),
                                    "support_gap_px": float(chart_zero[8] * (PUSHT_STATE_SIZE / 2.0)),
                                    "angle_improvement": float(zero_error - error),
                                    "xy_regression": float(
                                        np.linalg.norm(pose[:2] - target[:2])
                                        - np.linalg.norm(zero_pose[:2] - target[:2])
                                    ),
                                    "xy_safe": float(
                                        np.linalg.norm(pose[:2] - target[:2])
                                        <= np.linalg.norm(zero_pose[:2] - target[:2]) + 3.0
                                    ),
                                    "contact_retained": float(agent_distance <= 100.0),
                                    "action_x": float(action[0]),
                                    "action_y": float(action[1]),
                                })
    finally:
        env.close()

    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        key = ":".join([
            str(row["lever"]), str(row["sign_source"]),
            "flipped" if row["sign_flip"] else "direct",
            f"tan={row['tangent_magnitude']:.2f}",
            f"in={row['inward_magnitude']:.2f}",
        ])
        grouped[key].append(row)
    report = {
        "protocol": {
            "seed": args.seed,
            "terminal_roots": args.terminal_roots,
            "position_jitter": args.position_jitter,
            "support_gap_max": args.support_gap_max,
            "support_angle_samples": args.support_angle_samples,
            "action_repeat": args.action_repeat,
            "label": "one-step independent simulator successor; no learned adapter",
        },
        "summary": summarize(rows),
        "groups": {key: summarize(value) for key, value in sorted(grouped.items())},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": report["summary"], "groups": report["groups"]}, indent=2))


if __name__ == "__main__":
    main()
