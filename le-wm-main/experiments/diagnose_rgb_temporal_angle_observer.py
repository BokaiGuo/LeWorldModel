"""Root-disjoint audit for the RGB temporal orientation observer.

The audit compares the existing framewise RGB KNN angle readout with a
temporal silhouette-registration track.  It uses simulator state only for
offline labels and never for the observer inputs.  The output is deliberately
an observer audit, not a planning result: it must establish angle/increment
quality before the observer is enabled in a closed-loop replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from run_rgb_local_effect_adapter import (
    balanced_source_cases,
    dynamic_rgb_shape_angle_rows,
    rgb_shape_raster_features,
    predict_orientation,
    terminal_pose_cases,
    train_rgb_angle_knn,
    temporal_rgb_angle_observation,
    wrap,
)


def _roots(env: gym.Env, count: int, seed: int, protocol: dict[str, Any]) -> list[Any]:
    terminal = int(round(count * float(protocol.get("terminal_root_fraction", 0.25))))
    roots = balanced_source_cases(
        env, count - terminal, int(protocol.get("root_rollout_steps", 8)), seed,
        float(protocol.get("contact_root_fraction", 0.60)),
        int(protocol.get("contact_approach_steps", 48)),
    )
    roots.extend(terminal_pose_cases(
        env, terminal, seed + 400_000,
        position_jitter=float(protocol.get("terminal_position_jitter", 16.0)),
        support_gap_max=float(protocol.get("terminal_root_support_gap", 0.0)),
        support_angle_samples=int(protocol.get("terminal_root_support_angle_samples", 32)),
    ))
    return roots


def _sequence(
    env: gym.Env, root: Any, seed: int, steps: int, contact_biased: bool,
) -> tuple[list[np.ndarray], np.ndarray]:
    state, goal = root[0], root[1]
    observation, _ = env.reset(
        seed=seed, options={"state": np.asarray(state).copy(), "goal_state": np.asarray(goal).copy()},
    )
    images: list[np.ndarray] = []
    angles: list[float] = []
    rng = np.random.default_rng(seed + 11)
    for step in range(steps + 1):
        images.append(np.asarray(env.render(), dtype=np.uint8).copy())
        current = np.asarray(observation["state"], dtype=np.float32)
        # PushT's raw state stores the block angle at index 4.  The helper
        # theta_from_pose intentionally expects the six-dimensional
        # sin/cos state_vector, so using it on raw simulator state would turn
        # the angle into atan2(theta, velocity_x) and invalidate this audit.
        angles.append(float(current[4]))
        if step == steps:
            break
        relative = current[2:4] - current[:2]
        distance = float(np.linalg.norm(relative))
        direction = relative / max(distance, 1e-6)
        if contact_biased and (bool(root[6]) or distance <= 85.0):
            tangent = np.asarray([-direction[1], direction[0]], dtype=np.float32)
            sign = 1.0 if (step + seed) % 2 == 0 else -1.0
            action = 0.20 * direction + sign * 0.14 * tangent
            action = np.clip(action, -0.35, 0.35)
        elif contact_biased:
            action = np.clip(0.75 * direction, -0.75, 0.75)
        elif (step + seed) % 3 == 0:
            action = rng.uniform(-0.85, 0.85, size=2)
        else:
            action = np.clip(0.65 * direction + 0.35 * rng.normal(size=2), -1.0, 1.0)
        observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        if terminated or truncated:
            break
    return images, np.asarray(angles, dtype=np.float32)


def _circular_mae(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(wrap(np.asarray(predicted) - np.asarray(target)))))


def _evaluate(
    env: gym.Env,
    roots: list[Any],
    decoder: Any,
    training: dict[str, Any],
    device: str,
    steps: int,
    contact_biased: bool,
) -> dict[str, Any]:
    static_errors: list[float] = []
    temporal_errors: list[float] = []
    true_increments: list[float] = []
    predicted_increments: list[float] = []
    registration_rows: list[dict[str, float]] = []
    branch_flips = 0
    frame_count = 0
    updates = 0
    for index, root in enumerate(roots):
        images, angles = _sequence(env, root, 900_000 + index, steps, contact_biased)
        features = rgb_shape_raster_features(np.asarray(images), "block")
        orientation = predict_orientation(decoder, features, training, device)
        frame_angles = np.arctan2(orientation[:, 0], orientation[:, 1]).astype(np.float32)
        static_errors.extend(np.abs(wrap(frame_angles - angles)).tolist())
        state: dict[str, Any] = {}
        temporal_track: list[float] = []
        for frame in range(len(images)):
            if frame == 0:
                selected = float(frame_angles[frame])
                state["initialized"] = True
                state["angle"] = selected
            else:
                selected, audit = temporal_rgb_angle_observation(
                    images[frame], images[frame - 1],
                    float(frame_angles[frame]), float(frame_angles[frame - 1]), state,
                )
                registration_rows.append(audit)
                updates += int(float(audit["temporal_angle_update_used"]) > 0.0)
                true_increment = float(wrap(angles[frame] - angles[frame - 1]))
                true_increments.append(true_increment)
                predicted_increments.append(float(audit["temporal_angle_delta"]))
                branch_flips += int(float(audit["temporal_angle_branch_flip"]) > 0.0)
            temporal_track.append(float(selected))
            frame_count += 1
        temporal_errors.extend(np.abs(wrap(np.asarray(temporal_track) - angles)).tolist())
    true_inc = np.asarray(true_increments, dtype=np.float32)
    pred_inc = np.asarray(predicted_increments, dtype=np.float32)
    informative = np.abs(true_inc) >= 0.02
    if bool(informative.any()):
        sign_accuracy = float(np.mean(np.sign(true_inc[informative]) == np.sign(pred_inc[informative])))
    else:
        sign_accuracy = None
    if len(true_inc) >= 2 and float(np.std(true_inc)) > 1e-6 and float(np.std(pred_inc)) > 1e-6:
        increment_corr = float(np.corrcoef(true_inc, pred_inc)[0, 1])
    else:
        increment_corr = None
    return {
        "roots": len(roots),
        "contact_biased": bool(contact_biased),
        "frames": frame_count,
        "static_angle_mae": _circular_mae(
            np.asarray(static_errors), np.zeros(len(static_errors)),
        ),
        "temporal_angle_mae": _circular_mae(
            np.asarray(temporal_errors), np.zeros(len(temporal_errors)),
        ),
        "increment_mae": _circular_mae(pred_inc, true_inc) if len(true_inc) else None,
        "increment_correlation": increment_corr,
        "wrong_sign_detection_rate": sign_accuracy,
        "framewise_branch_flip_rate": float(branch_flips / max(frame_count - len(roots), 1)),
        "temporal_update_rate": float(updates / max(frame_count - len(roots), 1)),
        "registration_mean_confidence": float(np.mean([
            row["temporal_angle_confidence"] for row in registration_rows
        ])) if registration_rows else None,
        "registration_mean_best_score": float(np.mean([
            row["temporal_angle_best_score"] for row in registration_rows
        ])) if registration_rows else None,
        "registration_rows": registration_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-roots", type=int, default=None)
    parser.add_argument("--val-roots", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.adapter_report.read_text())
    protocol = report["protocol"]
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_count = int(args.train_roots or protocol.get("train_roots", 480))
    val_count = int(args.val_roots or protocol.get("val_roots", 120))
    seed = int(protocol.get("data_seed", protocol.get("seed", 9181)))
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=128)
    try:
        train_roots = _roots(env, train_count, seed, protocol)
        # Match the adapter's declared root split exactly: validation roots
        # use data_seed+10_000, while the temporal rollout seed is independent
        # and only controls the dynamic augmentation trajectory.
        val_roots = _roots(env, val_count, seed + 10_000, protocol)
        train_static = rgb_shape_raster_features(np.asarray([root[2] for root in train_roots]), "block")
        train_dynamic, train_dynamic_targets = dynamic_rgb_shape_angle_rows(
            env, train_roots, int(protocol.get("image_size", 64)), seed + 620_000,
            int(protocol.get("readout_augmentation_steps", 4)), "raster",
        )
        if bool(protocol.get("rgb_angle_contact_augmentation", True)):
            contact_shape, contact_targets = dynamic_rgb_shape_angle_rows(
                env, train_roots, int(protocol.get("image_size", 64)), seed + 625_000,
                int(protocol.get("readout_augmentation_steps", 4)), "raster", contact_biased=True,
            )
            train_dynamic = np.concatenate([train_dynamic, contact_shape], axis=0)
            train_dynamic_targets = np.concatenate([train_dynamic_targets, contact_targets], axis=0)
        train_features = np.concatenate([train_static, train_dynamic], axis=0)
        train_targets = np.concatenate([
            np.asarray([[np.sin(root[0][4]), np.cos(root[0][4])] for root in train_roots], dtype=np.float32),
            train_dynamic_targets,
        ], axis=0)
        decoder, training = train_rgb_angle_knn(train_features, train_targets)
        decoder = decoder.to(device)
        random_eval = _evaluate(env, val_roots, decoder, training, device, args.rollout_steps, False)
        contact_eval = _evaluate(env, val_roots, decoder, training, device, args.rollout_steps, True)
    finally:
        env.close()
    output = {
        "protocol": {
            "adapter_report": str(args.adapter_report.resolve()),
            "train_roots": train_count,
            "validation_roots": val_count,
            "rollout_steps": args.rollout_steps,
            "device": device,
            "observer": "framewise_rgb_knn_vs_temporal_centered_silhouette_registration",
            "labels": "offline simulator angle labels; never observer inputs",
        },
        "random_dynamic": random_eval,
        "contact_biased_dynamic": contact_eval,
        "claim_boundary": "Offline root-disjoint observer audit only; no closed-loop planning claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    md = args.output.with_suffix(".md")
    md.write_text(
        "# RGB temporal angle observer audit\n\n"
        f"- Random dynamic angle MAE: {random_eval['static_angle_mae']:.6g} -> {random_eval['temporal_angle_mae']:.6g}.\n"
        f"- Contact dynamic angle MAE: {contact_eval['static_angle_mae']:.6g} -> {contact_eval['temporal_angle_mae']:.6g}.\n"
        f"- Contact increment correlation: {contact_eval['increment_correlation']}.\n"
        f"- Contact wrong-sign detection: {contact_eval['wrong_sign_detection_rate']}.\n"
        f"- Contact branch-flip rate: {contact_eval['framewise_branch_flip_rate']:.6g}.\n"
        "This audit is root-disjoint and offline; it does not establish planning success.\n"
    )
    print(json.dumps({k: v for k, v in output.items() if k != "random_dynamic" or True}, indent=2)[:12000])


if __name__ == "__main__":
    main()
