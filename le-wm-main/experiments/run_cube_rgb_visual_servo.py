"""RGB-only visual-servo approach and contact-phase Cube audit.

This experiment targets the diagnosed Cube failure boundary: an open-loop
macro can overshoot the block before descending.  A calibration-only image
Jacobian maps world xy actions to RGB effector motion.  At deployment the
controller uses only current RGB, rendered goal RGB, previous RGB/action, and
its finite phase state.  Simulator state is used only for paired reset and
final evaluator metrics.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

# Import PyTorch before configuring MuJoCo's headless renderer.  This keeps the
# CUDA/NVML probe visible in the managed host runtime while making RGB
# rendering deterministic when no X11 DISPLAY is present.
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers environments

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.audit_cube_rgb_observability import (
    _largest_effector_centroid,
    _make_env,
    _target_red_centroid,
    _vec,
)


XY_ACTIONS = np.asarray(
    [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
    dtype=np.float32,
)

# OGB-Cube uses relative gripper-opening actions: +1 closes and -1 opens.
# Keeping these names explicit prevents the approach/close phase from silently
# reversing the intended contact transition.
GRIPPER_CLOSE_ACTION = 1.0
GRIPPER_OPEN_ACTION = -1.0


def _capture_initial(seed: int, task_id: int) -> dict[str, Any]:
    env = _make_env()
    try:
        _, info = env.reset(seed=int(seed), options={"task_id": int(task_id), "render_goal": True})
        raw = env.unwrapped
        return {
            "seed": int(seed),
            "task_id": int(task_id),
            "variation_values": copy.deepcopy(raw.variation_space.value),
            "state": np.concatenate([raw.data.qpos.copy(), raw.data.qvel.copy()]).tolist(),
            "goal": _goal_position(env).copy(),
        }
    finally:
        env.close()


def _goal_position(env: gym.Env) -> np.ndarray:
    raw = env.unwrapped
    target_block = int(getattr(raw, "_target_block", 0))
    return np.asarray(raw.cur_task_info["goal_xyzs"][target_block], dtype=np.float32)


def _reset(env: gym.Env, initial: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    obs, info = env.reset(
        seed=int(initial["seed"]),
        options={
            "task_id": int(initial["task_id"]),
            "render_goal": True,
            "variation_values": copy.deepcopy(initial["variation_values"]),
            "state": np.asarray(initial["state"], dtype=np.float64),
        },
    )
    goal_image = np.asarray(env.unwrapped._cur_goal_rendered, dtype=np.uint8).copy()
    return np.asarray(obs, dtype=np.uint8).copy(), info, goal_image


def _image_xy(
    image: np.ndarray, *, effector: bool, target_reference_y: float | None = None,
) -> tuple[np.ndarray, bool]:
    value = (
        _largest_effector_centroid(image)
        if effector
        else _target_red_centroid(image, reference_y=target_reference_y)
    )
    return np.asarray(value[:2], dtype=np.float32), bool(value[2] > 0.5)


def _fit_rgb_action_map(
    initials: list[dict[str, Any]], probe_steps: int, descent_steps: int,
    device: str, chart_ridge: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit the RGB action map at the same height used by the runtime policy."""
    deltas: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    effector_features: list[np.ndarray] = []
    target_features: list[np.ndarray] = []
    approach_features: list[np.ndarray] = []
    push_features: list[np.ndarray] = []
    effector_world: list[np.ndarray] = []
    target_world: list[np.ndarray] = []
    approach_targets: list[np.ndarray] = []
    push_targets: list[np.ndarray] = []
    world_deltas: list[np.ndarray] = []
    env = _make_env()
    try:
        for initial in initials:
            base_obs, base_info, base_goal_image = _reset(env, initial)
            for _ in range(descent_steps):
                descent = np.zeros(5, dtype=np.float32)
                descent[2] = -1.0
                base_obs, _, terminated, truncated, base_info = env.step(descent)
                if terminated or truncated:
                    break
            base_goal_xy, base_goal_valid = _image_xy(base_goal_image, effector=False)
            base_xy, base_valid = _image_xy(base_obs, effector=True)
            base_target_xy, base_target_valid = _image_xy(
                base_obs, effector=False,
                target_reference_y=float(base_goal_xy[1]) if base_goal_valid else None,
            )
            if not base_valid or not base_target_valid or not base_goal_valid:
                continue
            base_effector_world = _vec(base_info, "proprio/effector_pos", 3)[:2].copy()
            base_target_world = _vec(base_info, "privileged/block_0_pos", 3)[:2].copy()
            effector_features.append(base_xy.copy())
            target_features.append(base_target_xy.copy())
            approach_features.append(_relative_approach_features(
                base_xy, base_target_xy, base_goal_xy,
            ))
            push_features.append(_relative_push_features(base_target_xy, base_goal_xy))
            effector_world.append(base_effector_world)
            target_world.append(base_target_world)
            approach_targets.append(base_target_world - base_effector_world)
            push_targets.append(np.asarray(initial["goal"], dtype=np.float32)[:2] - base_target_world)
            for action_xy in XY_ACTIONS:
                obs, info, goal_image = _reset(env, initial)
                for _ in range(descent_steps):
                    descent = np.zeros(5, dtype=np.float32)
                    descent[2] = -1.0
                    obs, _, terminated, truncated, info = env.step(descent)
                    if terminated or truncated:
                        break
                action = np.zeros(5, dtype=np.float32)
                action[:2] = action_xy
                for _ in range(probe_steps):
                    obs, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                after_goal_xy, after_goal_valid = _image_xy(goal_image, effector=False)
                after_xy, after_valid = _image_xy(obs, effector=True)
                after_target_xy, after_target_valid = _image_xy(
                    obs, effector=False,
                    target_reference_y=float(after_goal_xy[1]) if after_goal_valid else None,
                )
                if after_valid and after_target_valid and after_goal_valid:
                    actions.append(action_xy.copy())
                    deltas.append(after_xy - base_xy)
                    effector_features.append(after_xy.copy())
                    target_features.append(after_target_xy.copy())
                    approach_features.append(_relative_approach_features(
                        after_xy, after_target_xy, after_goal_xy,
                    ))
                    push_features.append(_relative_push_features(after_target_xy, after_goal_xy))
                    after_effector_world = _vec(info, "proprio/effector_pos", 3)[:2].copy()
                    after_target_world = _vec(info, "privileged/block_0_pos", 3)[:2].copy()
                    effector_world.append(after_effector_world)
                    target_world.append(after_target_world)
                    approach_targets.append(after_target_world - after_effector_world)
                    push_targets.append(np.asarray(initial["goal"], dtype=np.float32)[:2] - after_target_world)
                    world_deltas.append(after_effector_world - base_effector_world)
    finally:
        env.close()
    if len(actions) < 4:
        raise RuntimeError("RGB effector was not visible on enough calibration probes")
    action_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.float64, device=device)
    delta_tensor = torch.as_tensor(np.asarray(deltas), dtype=torch.float64, device=device)
    # delta = action @ map.  CUDA is used for the fit when requested.
    mapping = torch.linalg.lstsq(action_tensor, delta_tensor).solution
    mapping_np = mapping.detach().cpu().numpy().astype(np.float32)
    singular_values = torch.linalg.svdvals(mapping)
    condition = float(torch.linalg.cond(mapping).item())
    feature_tensor = torch.as_tensor(
        np.asarray(effector_features), dtype=torch.float64, device=device,
    )
    target_feature_tensor = torch.as_tensor(
        np.asarray(target_features), dtype=torch.float64, device=device,
    )
    ones = torch.ones((feature_tensor.shape[0], 1), dtype=torch.float64, device=device)
    effector_chart = torch.linalg.lstsq(
        torch.cat([feature_tensor, ones], dim=1),
        torch.as_tensor(np.asarray(effector_world), dtype=torch.float64, device=device),
    ).solution
    target_chart = torch.linalg.lstsq(
        torch.cat([target_feature_tensor, ones], dim=1),
        torch.as_tensor(np.asarray(target_world), dtype=torch.float64, device=device),
    ).solution
    world_action = torch.linalg.lstsq(
        action_tensor,
        torch.as_tensor(np.asarray(world_deltas), dtype=torch.float64, device=device),
    ).solution
    world_singular_values = torch.linalg.svdvals(world_action)
    approach_feature_tensor = torch.as_tensor(
        np.asarray(approach_features), dtype=torch.float64, device=device,
    )
    push_feature_tensor = torch.as_tensor(
        np.asarray(push_features), dtype=torch.float64, device=device,
    )
    approach_target_tensor = torch.as_tensor(
        np.asarray(approach_targets), dtype=torch.float64, device=device,
    )
    push_target_tensor = torch.as_tensor(
        np.asarray(push_targets), dtype=torch.float64, device=device,
    )
    approach_ones = torch.ones(
        (approach_feature_tensor.shape[0], 1), dtype=torch.float64, device=device,
    )
    push_ones = torch.ones(
        (push_feature_tensor.shape[0], 1), dtype=torch.float64, device=device,
    )
    approach_chart, approach_chart_rmse = _fit_standardized_ridge_chart(
        approach_feature_tensor, approach_target_tensor, chart_ridge,
    )
    push_chart, push_chart_rmse = _fit_standardized_ridge_chart(
        push_feature_tensor, push_target_tensor, chart_ridge,
    )
    return {
        "pixel_action_mapping": mapping_np,
        "effector_chart": effector_chart.detach().cpu().numpy().astype(np.float32),
        "target_chart": target_chart.detach().cpu().numpy().astype(np.float32),
        "relative_approach_chart": approach_chart,
        "relative_push_chart": push_chart,
        "world_action_mapping_per_step": (
            world_action.detach().cpu().numpy().astype(np.float32) / max(probe_steps, 1)
        ),
    }, {
        "device": device,
        "probe_steps": int(probe_steps),
        "descent_steps": int(descent_steps),
        "samples": int(len(actions)),
        "mapping": mapping_np.tolist(),
        "singular_values": singular_values.detach().cpu().numpy().tolist(),
        "numerical_rank": int(torch.linalg.matrix_rank(mapping).item()),
        "condition_number": condition,
        "mean_fit_residual": float(torch.linalg.norm(action_tensor @ mapping - delta_tensor, dim=1).mean().item()),
        "world_chart_samples": int(feature_tensor.shape[0]),
        "world_chart_effector_calibration_rmse_m": float(torch.linalg.norm(
            torch.cat([feature_tensor, ones], dim=1) @ effector_chart
            - torch.as_tensor(np.asarray(effector_world), dtype=torch.float64, device=device),
            dim=1,
        ).mean().item()),
        "world_chart_target_calibration_rmse_m": float(torch.linalg.norm(
            torch.cat([target_feature_tensor, ones], dim=1) @ target_chart
            - torch.as_tensor(np.asarray(target_world), dtype=torch.float64, device=device),
            dim=1,
        ).mean().item()),
        "relative_chart_samples": int(approach_feature_tensor.shape[0]),
        "relative_chart_ridge": float(chart_ridge),
        "relative_approach_calibration_rmse_m": float(approach_chart_rmse),
        "relative_push_calibration_rmse_m": float(push_chart_rmse),
        "world_action_mapping_per_step": (
            world_action.detach().cpu().numpy().astype(np.float32) / max(probe_steps, 1)
        ).tolist(),
        "world_action_singular_values_per_step": (
            world_singular_values.detach().cpu().numpy().astype(np.float32) / max(probe_steps, 1)
        ).tolist(),
    }


def _relative_approach_features(
    effector_xy: np.ndarray, block_xy: np.ndarray, goal_xy: np.ndarray,
) -> np.ndarray:
    effector_xy = np.asarray(effector_xy, dtype=np.float32)[:2]
    block_xy = np.asarray(block_xy, dtype=np.float32)[:2]
    goal_xy = np.asarray(goal_xy, dtype=np.float32)[:2]
    # Only relative quantities are retained.  This avoids fitting a nearly
    # collinear high-dimensional affine chart that extrapolates catastrophically
    # when a held-out root lies outside a small calibration panel.
    return np.concatenate([block_xy - effector_xy, goal_xy - block_xy]).astype(np.float32)


def _relative_push_features(block_xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    block_xy = np.asarray(block_xy, dtype=np.float32)[:2]
    goal_xy = np.asarray(goal_xy, dtype=np.float32)[:2]
    return (goal_xy - block_xy).astype(np.float32)


def _fit_standardized_ridge_chart(
    features: torch.Tensor, targets: torch.Tensor, ridge: float,
) -> tuple[dict[str, np.ndarray], float]:
    if ridge <= 0.0 or not np.isfinite(ridge):
        raise ValueError("chart ridge must be finite and positive")
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1.0e-3)
    standardized = (features - mean) / scale
    ones = torch.ones((standardized.shape[0], 1), dtype=standardized.dtype, device=standardized.device)
    design = torch.cat([standardized, ones], dim=1)
    penalty = torch.eye(design.shape[1], dtype=design.dtype, device=design.device) * float(ridge)
    penalty[-1, -1] = 0.0
    weights = torch.linalg.solve(
        design.T @ design + penalty,
        design.T @ targets,
    )
    prediction = design @ weights
    rmse = float(torch.linalg.vector_norm(prediction - targets, dim=1).mean().item())
    return {
        "mean": mean.detach().cpu().numpy().astype(np.float32),
        "scale": scale.detach().cpu().numpy().astype(np.float32),
        "weights": weights.detach().cpu().numpy().astype(np.float32),
    }, rmse


def _predict_linear(feature: np.ndarray, chart: np.ndarray) -> np.ndarray:
    values = np.asarray(feature, dtype=np.float64).reshape(-1)
    if isinstance(chart, dict):
        mean = np.asarray(chart["mean"], dtype=np.float64)
        scale = np.asarray(chart["scale"], dtype=np.float64)
        weights = np.asarray(chart["weights"], dtype=np.float64)
        values = (values - mean) / np.maximum(scale, 1.0e-3)
    row = np.concatenate([values, [1.0]])
    if isinstance(chart, dict):
        return (row @ weights).astype(np.float32)
    return (row @ np.asarray(chart, dtype=np.float64)).astype(np.float32)


def _action_from_world_error(
    world_error: np.ndarray, mapping_per_step: np.ndarray, gain: float,
) -> np.ndarray:
    inverse = np.linalg.pinv(np.asarray(mapping_per_step, dtype=np.float64), rcond=1.0e-4)
    action = np.asarray(world_error, dtype=np.float64) @ inverse
    return np.clip(float(gain) * action, -1.0, 1.0).astype(np.float32)


def _action_from_pixel_error(
    pixel_error: np.ndarray, mapping: np.ndarray, gain: float,
) -> np.ndarray:
    inverse = np.linalg.pinv(np.asarray(mapping, dtype=np.float64), rcond=1.0e-4)
    action = np.asarray(pixel_error, dtype=np.float64) @ inverse
    return np.clip(float(gain) * action, -1.0, 1.0).astype(np.float32)


def _adaptive_pixel_push_action(
    pixel_error: np.ndarray,
    mapping: np.ndarray,
    gain: float,
    direction_sign: float,
) -> np.ndarray:
    """Apply a short-pulse sign correction to the RGB push proposal.

    ``mapping`` is an effector-image Jacobian, not a contact-conditioned block
    Jacobian.  Once the gripper is closed, the contact side can invert the
    observed block response.  This controller therefore treats the calibrated
    inverse as a proposal only and applies a single scalar sign state updated
    from successive RGB block-to-goal errors.  It uses no privileged contact
    label; the evaluator-only contact trace is never read here.
    """
    proposal = _action_from_pixel_error(pixel_error, mapping, gain)
    return np.clip(float(direction_sign) * proposal, -1.0, 1.0).astype(np.float32)


def _run_servo(
    initials: list[dict[str, Any]], calibration_model: dict[str, np.ndarray], *, max_steps: int,
    approach_threshold: float, approach_pixel_threshold: float,
    pixel_approach_gain: float, world_action_gain: float, approach_controller: str,
    push_controller: str, push_pixel_gain: float,
    approach_standoff_m: float,
    descend_steps: int,
    close_steps: int, push_steps: int, max_observation_dropout: int,
) -> dict[str, Any]:
    if approach_controller not in {"pixel", "world"}:
        raise ValueError(f"unknown approach controller: {approach_controller}")
    if push_controller not in {"pixel", "pixel_adaptive", "world"}:
        raise ValueError(f"unknown push controller: {push_controller}")
    if approach_standoff_m < 0.0 or not np.isfinite(approach_standoff_m):
        raise ValueError("approach standoff must be finite and non-negative")
    episodes: list[dict[str, Any]] = []
    for initial in initials:
        env = _make_env()
        try:
            obs, info, goal_image = _reset(env, initial)
            # At the reset height the camera projection of the robot working
            # end and the table block has a stable depth offset.  Aligning
            # them before lowering therefore cannot establish contact support.
            # Use a fixed safe descent first, then run the RGB XY servo at the
            # table height where the pixel chart is meaningful.
            phase = "descend"
            phase_steps = 0
            previous_action = np.zeros(5, dtype=np.float32)
            distances: list[float] = []
            actions: list[list[float]] = []
            phase_trace: list[str] = []
            contact_trace: list[float] = []
            block_trace: list[list[float]] = []
            effector_trace: list[list[float]] = []
            observation_dropout_trace: list[int] = []
            approach_pixel_error_trace: list[list[float]] = []
            approach_world_error_trace: list[list[float]] = []
            push_pixel_error_trace: list[list[float]] = []
            last_block_xy: np.ndarray | None = None
            last_effector_xy: np.ndarray | None = None
            last_goal_xy: np.ndarray | None = None
            last_push_error_norm: float | None = None
            push_direction_sign = 1.0
            push_adaptation_sign_trace: list[float] = []
            push_error_norm_trace: list[float] = []
            dropout_count = 0
            for _ in range(max_steps):
                raw_goal_xy, raw_goal_valid = _image_xy(goal_image, effector=False)
                raw_block_xy, raw_block_valid = _image_xy(
                    obs, effector=False,
                    target_reference_y=float(raw_goal_xy[1]) if raw_goal_valid else None,
                )
                raw_effector_xy, raw_effector_valid = _image_xy(obs, effector=True)
                if raw_block_valid and raw_effector_valid and raw_goal_valid:
                    last_block_xy = raw_block_xy.copy()
                    last_effector_xy = raw_effector_xy.copy()
                    last_goal_xy = raw_goal_xy.copy()
                    dropout_count = 0
                else:
                    dropout_count += 1
                observation_dropout_trace.append(dropout_count)
                held = (
                    last_block_xy is not None
                    and last_effector_xy is not None
                    and last_goal_xy is not None
                    and dropout_count <= max_observation_dropout
                )
                block_xy = last_block_xy if held else raw_block_xy
                effector_xy = last_effector_xy if held else raw_effector_xy
                goal_xy = last_goal_xy if held else raw_goal_xy
                block_valid = bool(raw_block_valid or held)
                effector_valid = bool(raw_effector_valid or held)
                goal_valid = bool(raw_goal_valid or held)
                action = np.zeros(5, dtype=np.float32)
                if phase == "approach":
                    if block_valid and effector_valid:
                        if not goal_valid:
                            phase = "fallback"
                            phase_steps = 0
                        else:
                            pixel_error = np.asarray(block_xy - effector_xy, dtype=np.float32)
                            approach_pixel_error_trace.append(pixel_error.tolist())
                            if approach_controller == "pixel":
                                if float(np.linalg.norm(pixel_error)) <= approach_pixel_threshold:
                                    phase = "close"
                                    phase_steps = 0
                                else:
                                    action[:2] = _action_from_pixel_error(
                                        pixel_error,
                                        calibration_model["pixel_action_mapping"],
                                        pixel_approach_gain,
                                    )
                            else:
                                world_error = _predict_linear(
                                    _relative_approach_features(effector_xy, block_xy, goal_xy),
                                    calibration_model["relative_approach_chart"],
                                )
                                goal_delta_world = _predict_linear(
                                    _relative_push_features(block_xy, goal_xy),
                                    calibration_model["relative_push_chart"],
                                )
                                goal_delta_norm = float(np.linalg.norm(goal_delta_world))
                                if goal_delta_norm > 1.0e-6:
                                    # Approach from the side opposite the desired
                                    # block motion, leaving a small calibration-
                                    # frozen standoff before closing the gripper.
                                    world_error = world_error - (
                                        float(approach_standoff_m)
                                        * goal_delta_world / goal_delta_norm
                                    )
                                approach_world_error_trace.append(world_error.tolist())
                                if float(np.linalg.norm(world_error)) <= approach_threshold:
                                    phase = "close"
                                    phase_steps = 0
                                else:
                                    action[:2] = _action_from_world_error(
                                        world_error,
                                        calibration_model["world_action_mapping_per_step"],
                                        world_action_gain,
                                    )
                    else:
                        # Fail closed when the RGB support is not observable.
                        phase = "fallback"
                if phase == "descend":
                    action[2] = -1.0
                    phase_steps += 1
                    if phase_steps >= descend_steps:
                        phase, phase_steps = "approach", 0
                elif phase == "close":
                    action[4] = GRIPPER_CLOSE_ACTION
                    phase_steps += 1
                    if phase_steps >= close_steps:
                        phase, phase_steps = "push", 0
                elif phase == "push":
                    if block_valid and goal_valid:
                        pixel_push_error = np.asarray(goal_xy - block_xy, dtype=np.float32)
                        push_pixel_error_trace.append(pixel_push_error.tolist())
                        push_error_norm = float(np.linalg.norm(pixel_push_error))
                        push_error_norm_trace.append(push_error_norm)
                        if push_controller == "pixel":
                            action[:2] = _action_from_pixel_error(
                                pixel_push_error,
                                calibration_model["pixel_action_mapping"],
                                push_pixel_gain,
                            )
                        elif push_controller == "pixel_adaptive":
                            if (
                                last_push_error_norm is not None
                                and push_error_norm > last_push_error_norm + 1.0e-3
                            ):
                                push_direction_sign *= -1.0
                            action[:2] = _adaptive_pixel_push_action(
                                pixel_push_error,
                                calibration_model["pixel_action_mapping"],
                                push_pixel_gain,
                                push_direction_sign,
                            )
                            push_adaptation_sign_trace.append(push_direction_sign)
                        else:
                            world_error = _predict_linear(
                                _relative_push_features(block_xy, goal_xy),
                                calibration_model["relative_push_chart"],
                            )
                            action[:2] = _action_from_world_error(
                                world_error,
                                calibration_model["world_action_mapping_per_step"],
                                world_action_gain,
                            )
                        last_push_error_norm = push_error_norm
                    phase_steps += 1
                    if phase_steps >= push_steps:
                        phase, phase_steps = "done", 0
                elif phase == "fallback":
                    action[:] = 0.0
                elif phase == "done":
                    action[:] = 0.0
                phase_trace.append(phase)
                actions.append(action.tolist())
                contact_trace.append(float(np.max(np.asarray(
                    info.get("proprio/gripper_contact", [0.0]), dtype=np.float32,
                ))))
                block_trace.append(_vec(info, "privileged/block_0_pos", 3).tolist())
                effector_trace.append(_vec(info, "proprio/effector_pos", 3).tolist())
                previous_action = action.copy()
                obs, _, terminated, truncated, info = env.step(action)
                obs = np.asarray(obs, dtype=np.uint8).copy()
                block = _vec(info, "privileged/block_0_pos", 3)[:2]
                distances.append(float(np.linalg.norm(block - np.asarray(initial["goal"])[:2])))
                if terminated or truncated:
                    break
            final_block = _vec(info, "privileged/block_0_pos", 3)[:2]
            final_distance = float(np.linalg.norm(final_block - np.asarray(initial["goal"])[:2]))
            episodes.append({
                "seed": int(initial["seed"]),
                "task_id": int(initial["task_id"]),
                "steps": len(actions),
                "final_goal_distance_m": final_distance,
                "best_goal_distance_m": float(np.min(distances)) if distances else None,
                "actions": actions,
                "phase_trace": phase_trace,
                "contact_trace_evaluator_only": contact_trace,
                "block_trace_evaluator_only": block_trace,
                "effector_trace_evaluator_only": effector_trace,
                "contact_steps_evaluator_only": int(np.sum(np.asarray(contact_trace) > 1.0e-5)),
                "observation_dropout_trace_runtime": observation_dropout_trace,
                "approach_pixel_error_trace_runtime": approach_pixel_error_trace,
                "approach_world_error_trace_runtime": approach_world_error_trace,
                "push_pixel_error_trace_runtime": push_pixel_error_trace,
                "push_error_norm_trace_runtime": push_error_norm_trace,
                "push_adaptation_sign_trace_runtime": push_adaptation_sign_trace,
                "approach_controller": approach_controller,
                "push_controller": push_controller,
                "max_observation_dropout_runtime": int(max(observation_dropout_trace, default=0)),
                "previous_action_runtime": previous_action.tolist(),
            })
        finally:
            env.close()
    return {
        "episodes": episodes,
        "mean_final_goal_distance_m": float(np.mean([row["final_goal_distance_m"] for row in episodes])),
        "mean_best_goal_distance_m": float(np.mean([row["best_goal_distance_m"] for row in episodes])),
    }


def _run_no_lateral(initials: list[dict[str, Any]], *, max_steps: int) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for initial in initials:
        env = _make_env()
        try:
            obs, info, _ = _reset(env, initial)
            distances: list[float] = []
            for step in range(max_steps):
                action = np.zeros(5, dtype=np.float32)
                if 0 <= step < 20:
                    action[2] = -1.0
                elif 20 <= step < 25:
                    action[4] = -1.0
                obs, _, terminated, truncated, info = env.step(action)
                block = _vec(info, "privileged/block_0_pos", 3)[:2]
                distances.append(float(np.linalg.norm(block - np.asarray(initial["goal"])[:2])))
                if terminated or truncated:
                    break
            final_block = _vec(info, "privileged/block_0_pos", 3)[:2]
            episodes.append({
                "seed": int(initial["seed"]),
                "task_id": int(initial["task_id"]),
                "steps": len(distances),
                "final_goal_distance_m": float(np.linalg.norm(final_block - np.asarray(initial["goal"])[:2])),
                "best_goal_distance_m": float(np.min(distances)) if distances else None,
            })
        finally:
            env.close()
    return {
        "episodes": episodes,
        "mean_final_goal_distance_m": float(np.mean([row["final_goal_distance_m"] for row in episodes])),
        "mean_best_goal_distance_m": float(np.mean([row["best_goal_distance_m"] for row in episodes])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261180)
    parser.add_argument("--calibration-roots", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--probe-steps", type=int, default=4)
    parser.add_argument(
        "--max-steps", type=int, default=90,
        help="fixed episode budget; the repaired contact protocol leaves room after a tight approach",
    )
    parser.add_argument(
        "--approach-threshold", type=float, default=0.015,
        help="tight calibrated world-space approach threshold before closing the gripper",
    )
    parser.add_argument("--approach-pixel-threshold", type=float, default=0.05,
                        help="RGB normalized target-effector pixel error for contact approach")
    parser.add_argument("--pixel-approach-gain", type=float, default=0.20,
                        help="frozen proportional gain for the RGB pixel action inverse")
    parser.add_argument("--world-action-gain", type=float, default=0.20,
                        help="frozen proportional gain applied after the calibration action inverse")
    parser.add_argument("--approach-controller", choices=("pixel", "world"), default="world",
                        help="fixed RGB approach law: pixel inverse or calibration-only relative world chart")
    parser.add_argument("--approach-standoff-m", type=float, default=0.0,
                        help="calibration-frozen distance to stop on the side opposite the goal motion")
    parser.add_argument(
        "--push-controller", choices=("pixel", "pixel_adaptive", "world"), default="pixel_adaptive",
        help="RGB push law: fixed pixel inverse, RGB short-pulse sign-adaptive pixel inverse, or calibration world chart",
    )
    parser.add_argument("--push-pixel-gain", type=float, default=0.20,
                        help="frozen proportional gain for pixel re-observation push")
    parser.add_argument("--max-observation-dropout", type=int, default=10,
                        help="frozen RGB landmark hold window before fail-closed fallback")
    parser.add_argument("--chart-ridge", type=float, default=1.0e-2,
                        help="calibration-only standardized ridge for relative world charts")
    parser.add_argument("--descend-steps", type=int, default=20)
    parser.add_argument("--close-steps", type=int, default=5)
    parser.add_argument("--push-steps", type=int, default=25)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for RGB visual servo but unavailable")
    if args.calibration_roots < 4 or args.episodes < 4:
        raise ValueError("need at least 4 calibration roots and 4 test episodes")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[purpose] RGB-only Cube visual servo; device={args.device}")
    calibration = [
        _capture_initial(args.seed + index, int(args.tasks[index % len(args.tasks)]))
        for index in range(args.calibration_roots)
    ]
    test = [
        _capture_initial(args.seed + 1000 + index, int(args.tasks[index % len(args.tasks)]))
        for index in range(args.episodes)
    ]
    calibration_model, calibration_report = _fit_rgb_action_map(
        calibration, args.probe_steps, args.descend_steps, args.device, args.chart_ridge,
    )
    servo = _run_servo(
        test, calibration_model, max_steps=args.max_steps,
        approach_threshold=args.approach_threshold,
        approach_pixel_threshold=args.approach_pixel_threshold,
        pixel_approach_gain=args.pixel_approach_gain,
        world_action_gain=args.world_action_gain,
        approach_controller=args.approach_controller,
        approach_standoff_m=args.approach_standoff_m,
        push_controller=args.push_controller,
        push_pixel_gain=args.push_pixel_gain,
        descend_steps=args.descend_steps, close_steps=args.close_steps,
        push_steps=args.push_steps, max_observation_dropout=args.max_observation_dropout,
    )
    baseline = _run_no_lateral(test, max_steps=args.max_steps)
    servo_by_seed = {row["seed"]: row for row in servo["episodes"]}
    base_by_seed = {row["seed"]: row for row in baseline["episodes"]}
    delta = np.asarray([
        servo_by_seed[seed]["final_goal_distance_m"] - base_by_seed[seed]["final_goal_distance_m"]
        for seed in base_by_seed
    ], dtype=np.float64)
    report = {
        "schema": "cube_rgb_visual_servo.v10",
        "protocol": {
            "environment": "swm/OGBCube-v0",
            "device": args.device,
            "calibration_roots": args.calibration_roots,
            "test_episodes": args.episodes,
            "root_disjoint": True,
            "runtime_input": "current RGB, rendered goal RGB, previous RGB/action, finite phase state",
            "runtime_privileged_state": False,
            "evaluator_state_role": "paired reset and final distance only",
            "camera_action_map_fit": "RGB effector displacement from calibration probes only",
            "effector_feature": "purple-component union centroid in frozen working-region band",
            "block_feature": "largest sufficiently visible red component (physical cube) in current RGB; goal red component in goal RGB",
            "probe_steps": args.probe_steps,
            "calibration_descent_steps": args.descend_steps,
            "approach_threshold": args.approach_threshold,
            "approach_pixel_threshold": args.approach_pixel_threshold,
            "pixel_approach_gain": args.pixel_approach_gain,
            "world_action_gain": args.world_action_gain,
            "approach_controller": args.approach_controller,
            "approach_standoff_m": args.approach_standoff_m,
            "push_controller": args.push_controller,
            "push_pixel_gain": args.push_pixel_gain,
            "max_observation_dropout": args.max_observation_dropout,
            "chart_ridge": args.chart_ridge,
            "phase_schedule": f"fixed_descend -> RGB_{args.approach_controller}_approach -> close -> {args.push_controller}_push -> done",
            "gripper_action_semantics": "relative opening: +1 closes, -1 opens",
            "pixel_adaptive_push_semantics": (
                "proposal uses effector RGB Jacobian; after contact, reverse the proposal only when the observed RGB block-goal error increases"
                if args.push_controller == "pixel_adaptive" else None
            ),
        },
        "calibration": calibration_report,
        "visual_servo": servo,
        "no_lateral_baseline": baseline,
        "paired": {
            "seed_order": list(base_by_seed),
            "final_distance_delta_servo_minus_baseline_m": delta.tolist(),
            "mean_final_distance_delta_m": float(delta.mean()),
            "servo_wins": int(np.sum(delta < 0.0)),
            "servo_losses": int(np.sum(delta > 0.0)),
            "servo_ties": int(np.sum(delta == 0.0)),
        },
        "claim_boundary": "RGB-only visual-servo approach audit; not a universal world-model or causal latent-dynamics claim.",
    }
    (args.output / "visual_servo.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["paired"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
