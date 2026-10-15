"""Audit RGB short-pulse action effects with successor re-observation.

Each candidate is executed from the same offline-reset root and then scored
from the rendered successor RGB frame.  The controller-facing signal is
therefore ``RGB_t -> action pulse -> RGB_{t+1}``; simulator state is used only
to reset matched counterfactuals and to score the offline reference ranking.

This is intentionally not a learned-world-model planning claim.  It tests the
specific repair suggested by the negative results: if a frozen RGB pose chart
is usable, can short-pulse re-observation recover action-to-angle transfer
without relying on the failed open-loop learned successor?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import torch

try:
    from .audit_rgb_analytic_pose import (
        apply_calibrated_harmonic_ridge,
        extract_moments,
        resize_image,
    )
    from .run_causal_visual_pusht_world_model import balanced_source_cases
except ImportError:
    from audit_rgb_analytic_pose import (  # type: ignore
        apply_calibrated_harmonic_ridge,
        extract_moments,
        resize_image,
    )
    from run_causal_visual_pusht_world_model import balanced_source_cases  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-calibration", required=True)
    parser.add_argument("--goal-pose-calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--roots", type=int, default=256)
    parser.add_argument("--contact-approach-steps", type=int, default=48)
    parser.add_argument("--fan-size", type=int, default=8)
    parser.add_argument("--pulse-magnitude", type=float, default=0.70)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def angle_cost(
    block_xy: np.ndarray, block_angle: np.ndarray,
    goal_xy: np.ndarray, goal_angle: np.ndarray,
) -> np.ndarray:
    translation = np.linalg.norm(block_xy - goal_xy, axis=-1) / 20.0
    orientation = wrap_angle(block_angle - goal_angle)
    return translation + 20.0 * (1.0 - np.cos(orientation))


def top1_by_root(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.mean(np.argmin(predicted, axis=1) == np.argmin(reference, axis=1)))


@torch.inference_mode()
def fit_observed_effect_adapter(
    source_features: dict[str, np.ndarray], successor_features: dict[str, np.ndarray],
    zero_successor_features: dict[str, np.ndarray],
    source_angle: np.ndarray, successor_angle: np.ndarray, actions: np.ndarray,
    zero_successor_angle: np.ndarray, source_xy: np.ndarray,
    zero_successor_xy: np.ndarray, goal_xy: np.ndarray, goal_angle: np.ndarray,
    root_ids: np.ndarray, device: str, seed: int,
) -> dict[str, Any]:
    """Fit a fixed-capacity paired RGB/action local-effect adapter.

    The old audit regressed the absolute action successor from the source
    frame.  That confounds passive motion and contact release with the action
    effect.  This adapter instead predicts the *paired* effect relative to a
    zero-action successor from the same root, then adds it to that observed
    zero-action baseline.  Both successors are RGB observations at runtime;
    simulator states remain labels/reference only.
    """
    roots = np.unique(root_ids)
    permutation = np.random.default_rng(seed + 77).permutation(roots)
    calibration_roots = set(permutation[: len(roots) // 2].tolist())
    calibration = np.asarray([root in calibration_roots for root in root_ids], dtype=bool)
    source_moments = source_features["raw_moments"][:, 3:10]
    zero_moments = zero_successor_features["raw_moments"][:, 3:10]
    source_xy = np.asarray(source_xy, dtype=np.float32)
    zero_successor_xy = np.asarray(zero_successor_xy, dtype=np.float32)
    base = np.column_stack([
        source_xy,
        zero_successor_xy,
        np.sin(source_angle), np.cos(source_angle),
        np.sin(zero_successor_angle), np.cos(zero_successor_angle),
        actions,
        source_moments,
        zero_moments,
    ]).astype(np.float32)
    x = torch.as_tensor(base, dtype=torch.float32, device=device)
    train_mask = torch.as_tensor(calibration, dtype=torch.bool, device=device)
    mean = x[train_mask].mean(dim=0)
    scale = x[train_mask].std(dim=0).clamp_min(1.0e-4)
    z = (x - mean) / scale
    # Root-disjoint calibration is intentionally low-data.  The previous
    # quadratic expansion (105 dimensions for 32 calibration roots) fit the
    # calibration roots but collapsed on held-out roots.  Keep the paired
    # finite-difference interface explicit and low-capacity: the action, the
    # observed zero-successor baseline, and central-shape context are enough
    # to test transfer without giving the adapter a memorization channel.
    design = torch.cat([
        torch.ones((len(z), 1), dtype=z.dtype, device=device), z,
    ], dim=1)
    target = torch.as_tensor(
        np.column_stack([
            successor_features["raw_moments"][:, 1:3]
            - zero_successor_features["raw_moments"][:, 1:3],
            np.cos(wrap_angle(successor_angle - zero_successor_angle)),
            np.sin(wrap_angle(successor_angle - zero_successor_angle)),
        ]), dtype=torch.float32, device=device,
    )
    x_train, y_train = design[train_mask], target[train_mask]
    gram = x_train.T @ x_train
    ridge = 1.0e-1 * torch.eye(gram.shape[0], device=device, dtype=gram.dtype)
    weight = torch.linalg.solve(gram + ridge, x_train.T @ y_train)
    prediction = (design @ weight).cpu().numpy()
    predicted_xy = zero_successor_xy + prediction[:, :2]
    predicted_angle = zero_successor_angle + np.arctan2(prediction[:, 3], prediction[:, 2])
    predicted_cost = angle_cost(
        predicted_xy, predicted_angle,
        np.asarray(goal_xy), np.asarray(goal_angle),
    )
    predicted_matrix = np.asarray([
        predicted_cost[root_ids == root] for root in roots
    ])
    return {
        "calibration_roots": int(len(calibration_roots)),
        "test_roots": int(len(roots) - len(calibration_roots)),
        "design_dim": int(design.shape[1]),
        "test_top1": top1_by_root(
            predicted_matrix[~np.asarray([root in calibration_roots for root in roots])],
            np.asarray([angle_cost(
                successor_features["raw_moments"][root_ids == root, 1:3],
                successor_angle[root_ids == root],
                goal_xy[root_ids == root], goal_angle[root_ids == root],
            ) for root in roots])[~np.asarray([root in calibration_roots for root in roots])],
        ),
        "test_angle_effect_mae_rad": float(np.mean(np.abs(wrap_angle(
            (predicted_angle - zero_successor_angle)[~calibration]
            - (successor_angle - zero_successor_angle)[~calibration],
        )))),
        "interface": "paired_action_successor_minus_zero_successor",
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is unavailable; RGB re-observed action audit refuses CPU fallback")
    if args.roots < 16 or args.fan_size < 4:
        raise ValueError("roots/fan-size are too small for the declared audit")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"[purpose] RGB short-pulse re-observation audit; device={args.device}; roots={args.roots}; fan={args.fan_size}")
    print("[protocol] state is offline reset/reference only; successor decision signal is RGB-only")

    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    source_images: list[np.ndarray] = []
    goal_images: list[np.ndarray] = []
    successor_images: list[np.ndarray] = []
    zero_successor_images: list[np.ndarray] = []
    source_states: list[np.ndarray] = []
    successor_states: list[np.ndarray] = []
    zero_successor_states: list[np.ndarray] = []
    goal_states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    root_ids: list[int] = []
    try:
        cases = balanced_source_cases(
            env, args.roots, root_rollout_steps=8, seed=args.seed,
            contact_fraction=1.0, contact_approach_steps=args.contact_approach_steps,
        )
        directions = np.linspace(0.0, 2.0 * np.pi, args.fan_size, endpoint=False)
        action_bank = np.asarray([
            [args.pulse_magnitude * math.cos(theta), args.pulse_magnitude * math.sin(theta)]
            for theta in directions
        ], dtype=np.float32)
        for root_id, case in enumerate(cases):
            state, goal, image, _previous, goal_image, _previous_action, _contact = case
            for action in action_bank:
                # Matched zero-action successor: removes passive motion and
                # contact-release drift from the action-effect estimate.
                env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
                zero_observation = None
                for _ in range(args.action_repeat):
                    zero_observation, _, terminated, truncated, _ = env.step(
                        np.zeros(2, dtype=np.float32)
                    )
                    if terminated or truncated:
                        break
                if zero_observation is None:
                    raise RuntimeError("PushT returned no observation after zero-action pulse")
                zero_successor_images.append(resize_image(env.render().copy(), args.image_size))
                zero_successor_states.append(
                    np.asarray(zero_observation["state"], dtype=np.float32).copy()
                )
                env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
                observation = None
                for _ in range(args.action_repeat):
                    observation, _, terminated, truncated, _ = env.step(action.copy())
                    if terminated or truncated:
                        break
                if observation is None:
                    raise RuntimeError("PushT returned no observation after action pulse")
                source_images.append(resize_image(image, args.image_size))
                goal_images.append(resize_image(goal_image, args.image_size))
                successor_images.append(resize_image(env.render().copy(), args.image_size))
                source_states.append(np.asarray(state, dtype=np.float32).copy())
                successor_states.append(np.asarray(observation["state"], dtype=np.float32).copy())
                goal_states.append(np.asarray(goal, dtype=np.float32).copy())
                actions.append(action.copy())
                root_ids.append(root_id)
    finally:
        env.close()

    source_images_np = np.asarray(source_images, dtype=np.uint8)
    goal_images_np = np.asarray(goal_images, dtype=np.uint8)
    successor_images_np = np.asarray(successor_images, dtype=np.uint8)
    source_features = extract_moments(source_images_np, args.device, args.batch_size)
    goal_features = extract_moments(goal_images_np, args.device, args.batch_size)
    successor_features = extract_moments(successor_images_np, args.device, args.batch_size)
    zero_successor_features = extract_moments(
        np.asarray(zero_successor_images, dtype=np.uint8), args.device, args.batch_size
    )
    source_angle = apply_calibrated_harmonic_ridge(
        source_features, Path(args.pose_calibration), args.device,
    )
    goal_angle = apply_calibrated_harmonic_ridge(
        goal_features, Path(args.goal_pose_calibration), args.device,
    )
    successor_angle = apply_calibrated_harmonic_ridge(
        successor_features, Path(args.pose_calibration), args.device,
    )
    zero_successor_angle = apply_calibrated_harmonic_ridge(
        zero_successor_features, Path(args.pose_calibration), args.device,
    )
    source_states_np = np.asarray(source_states)
    successor_states_np = np.asarray(successor_states)
    zero_successor_states_np = np.asarray(zero_successor_states)
    goal_states_np = np.asarray(goal_states)
    root_ids_np = np.asarray(root_ids, dtype=np.int64)
    actions_np = np.asarray(actions, dtype=np.float32)

    # Raw moment indices: [mass, mean_x, mean_y, ...].  The normalized image
    # chart is sufficient for ranking; no simulator positions enter runtime.
    visual_source_xy = source_features["raw_moments"][:, 1:3]
    visual_successor_xy = successor_features["raw_moments"][:, 1:3]
    visual_zero_successor_xy = zero_successor_features["raw_moments"][:, 1:3]
    visual_goal_xy = goal_features["raw_moments"][:, 1:3]
    reference_source_angle = source_states_np[:, 4]
    reference_successor_angle = successor_states_np[:, 4]
    reference_goal_angle = goal_states_np[:, 4]
    visual_source_angle = source_angle
    visual_successor_angle = successor_angle
    visual_goal_angle = goal_angle
    source_baseline_effect = wrap_angle(visual_successor_angle - visual_source_angle)
    visual_effect = wrap_angle(visual_successor_angle - zero_successor_angle)
    reference_effect = wrap_angle(
        reference_successor_angle - zero_successor_states_np[:, 4]
    )
    reference_source_effect = wrap_angle(
        reference_successor_angle - reference_source_angle
    )
    visual_effect_error = np.abs(wrap_angle(visual_effect - reference_effect))
    source_effect_error = np.abs(
        wrap_angle(source_baseline_effect - reference_source_effect)
    )

    visual_cost = angle_cost(
        visual_successor_xy, visual_successor_angle,
        visual_goal_xy, visual_goal_angle,
    )
    reference_cost = angle_cost(
        successor_states_np[:, 2:4], reference_successor_angle,
        goal_states_np[:, 2:4], reference_goal_angle,
    )
    roots = np.unique(root_ids_np)
    visual_matrix = np.asarray([visual_cost[root_ids_np == root] for root in roots])
    reference_matrix = np.asarray([reference_cost[root_ids_np == root] for root in roots])
    effect_matrix = np.asarray([visual_effect[root_ids_np == root] for root in roots])
    reference_effect_matrix = np.asarray([reference_effect[root_ids_np == root] for root in roots])
    action_matrix = np.asarray([actions_np[root_ids_np == root] for root in roots])
    contact_matrix = np.asarray([
        np.linalg.norm(source_states_np[root_ids_np == root, :2] - source_states_np[root_ids_np == root, 2:4], axis=1)
        <= 65.0
        for root in roots
    ])
    ranking = {
        "all_roots": int(len(roots)),
        "contact_roots": int(contact_matrix.any(axis=1).sum()),
        "visual_reobserved_top1": top1_by_root(visual_matrix, reference_matrix),
        "oracle_angle_only_top1": top1_by_root(
            20.0 * (1.0 - np.cos(effect_matrix - reference_goal_angle[::args.fan_size, None] + reference_source_angle[::args.fan_size, None])),
            reference_matrix,
        ),
    }
    contact_root_mask = contact_matrix.any(axis=1)
    if bool(contact_root_mask.any()):
        ranking["contact_visual_reobserved_top1"] = top1_by_root(
            visual_matrix[contact_root_mask], reference_matrix[contact_root_mask],
        )
    else:
        ranking["contact_visual_reobserved_top1"] = float("nan")
    observed_adapter = fit_observed_effect_adapter(
        source_features, successor_features, zero_successor_features,
        source_angle, successor_angle, actions_np, zero_successor_angle,
        visual_source_xy, visual_zero_successor_xy, visual_goal_xy, visual_goal_angle,
        root_ids_np, args.device, args.seed,
    )
    report: dict[str, Any] = {
        "protocol": {
            "seed": args.seed,
            "roots": int(len(roots)),
            "fan_size": args.fan_size,
            "pulse_magnitude": args.pulse_magnitude,
            "action_repeat": args.action_repeat,
            "paired_zero_action": True,
            "device": args.device,
            "cuda_device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            "runtime_signal": "RGB source, action pulse, RGB successor",
            "state_role": "offline reset and reference scoring only",
            "runtime_state_access": False,
        },
        "effect": {
            "n_pulses": int(len(actions_np)),
            "visual_angle_effect_mae_rad": float(np.mean(visual_effect_error)),
            "visual_angle_effect_rmse_rad": float(np.sqrt(np.mean(visual_effect_error ** 2))),
            "visual_angle_effect_15deg": float(np.mean(visual_effect_error <= np.deg2rad(15.0))),
            "source_to_action_angle_effect_mae_rad": float(np.mean(source_effect_error)),
            "paired_effect_mae_delta_vs_source": float(
                np.mean(visual_effect_error) - np.mean(source_effect_error)
            ),
            "reference_angle_effect_std_rad": float(np.std(reference_effect)),
        },
        "ranking": ranking,
        "observed_effect_adapter": observed_adapter,
        "claim_boundary": "RGB short-pulse re-observation audit; not open-loop world-model validity or universal planning evidence.",
    }
    (output / "rgb_reobserved_action_effect.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output / "rgb_reobserved_action_effect_arrays.npz",
        roots=roots, actions=action_matrix, visual_effect=effect_matrix,
        reference_effect=reference_effect_matrix, visual_cost=visual_matrix,
        reference_cost=reference_matrix,
        visual_zero_xy=np.asarray([visual_zero_successor_xy[root_ids_np == root] for root in roots]),
    )
    lines = [
        "# RGB re-observed short-pulse action-effect audit", "",
        f"Device: `{args.device}` ({report['protocol']['cuda_device']})", "",
        "| metric | value |", "|---|---:|",
        f"| visual angle-effect MAE (rad) | {report['effect']['visual_angle_effect_mae_rad']:.6f} |",
        f"| visual angle-effect <=15deg | {report['effect']['visual_angle_effect_15deg']:.4f} |",
        f"| source-to-action angle-effect MAE (rad) | {report['effect']['source_to_action_angle_effect_mae_rad']:.6f} |",
        f"| paired-minus-source effect MAE (rad) | {report['effect']['paired_effect_mae_delta_vs_source']:.6f} |",
        f"| visual re-observed top-1 | {ranking['visual_reobserved_top1']:.4f} |",
        f"| contact visual re-observed top-1 | {ranking['contact_visual_reobserved_top1']:.4f} |",
        f"| held-out observed-effect adapter top-1 | {observed_adapter['test_top1']:.4f} |",
        "", report["claim_boundary"], "",
    ]
    (output / "rgb_reobserved_action_effect.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "effect": report["effect"], "ranking": ranking}, indent=2))


if __name__ == "__main__":
    main()
