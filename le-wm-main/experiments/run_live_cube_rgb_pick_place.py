"""RGB-only closed-loop pick-and-place mechanism arm for OGB-Cube.

This controller isolates the action-interface failure of the old planar-push
bank.  It uses RGB centroids and executed RGB block displacement only:
approach, descend, close, lift, orthogonal effect probes, small closed-loop
transport pulses, lower, and release.  Privileged state is retained only as
an evaluator label.  This is an exploratory mechanism arm, not frozen-LeWM
planning evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
from scipy import ndimage
import stable_worldmodel  # noqa: F401
import torch


# OGBench action[4] is an increment to the Robotiq driver joint.  Despite the
# ``proprio/gripper_opening`` field name, the environment's Cube oracle uses
# +1 to close/grasp and -1 to open/release.  Keep this explicit because using
# the semantic name rather than the environment convention silently reverses
# the carry protocol.
GRIPPER_OPEN_ACTION = -1.0
GRIPPER_CLOSE_ACTION = 1.0

from experiments.audit_cube_rgb_observability import (
    _goal_position,
    _make_env,
    _target_red_centroid,
    _largest_effector_centroid,
)
from experiments.run_live_cube_rgb_macro_ranker import (
    _fit_rgb_pixel_action_mapping,
    _rgb_approach_action,
)
from experiments.run_cube_rgb_visual_servo import (
    _action_from_world_error,
    _capture_initial,
    _fit_rgb_action_map,
    _reset,
    _predict_linear,
    _relative_approach_features,
    _relative_push_features,
)
from experiments.gpu_preflight import require_cuda_allocation


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _target_red_area(image: np.ndarray) -> float:
    """Return normalized area of the largest RGB red block component."""
    image = np.asarray(image, dtype=np.uint8)
    red = image[..., 0].astype(np.float32)
    green = image[..., 1].astype(np.float32)
    blue = image[..., 2].astype(np.float32)
    mask = (red > 80.0) & (red > 1.25 * green) & (red > 1.15 * blue)
    components, count = ndimage.label(mask)
    if count == 0:
        return 0.0
    sizes = np.bincount(components.reshape(-1))[1:]
    return float(np.max(sizes) / max(mask.size, 1))


def _side_rgb_frame(env: gym.Env) -> np.ndarray:
    """Capture the fixed side RGB view used as an implicit depth cue.

    This is the lightweight analogue of Evo-Depth's multi-view spatial branch:
    the controller remains RGB-only, while the side projection supplies a
    view-complementary vertical relation that a single front image cannot
    identify reliably.
    """
    return np.asarray(env.unwrapped.render(camera="side_pixels"), dtype=np.uint8)


def _side_vertical_gap(env: gym.Env) -> float | None:
    """Return normalized side-view block-to-effector vertical gap, if visible."""
    side = _side_rgb_frame(env)
    block = _target_red_centroid(side)[:2]
    effector = _largest_effector_centroid(side)[:2]
    if np.linalg.norm(block) <= 0.0 or np.linalg.norm(effector) <= 0.0:
        return None
    # Camera perturbations and task layouts can flip the image-space vertical
    # ordering. The contact cue is a distance, not an oriented residual.
    return float(abs(block[1] - effector[1]))


def _multiview_effector_error(env: gym.Env, front_image: np.ndarray) -> np.ndarray | None:
    """Return block-minus-effector error in front and side RGB projections."""
    front_block = _target_red_centroid(front_image)[:2]
    front_effector = _largest_effector_centroid(front_image)[:2]
    side = _side_rgb_frame(env)
    side_block = _target_red_centroid(side)[:2]
    side_effector = _largest_effector_centroid(side)[:2]
    values = np.concatenate((front_block - front_effector, side_block - side_effector))
    if not np.isfinite(values).all() or np.linalg.norm(values) <= 0.0:
        return None
    return values.astype(np.float64)


def _numpyify(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _numpyify(item) for key, item in value.items()}
    if isinstance(value, list):
        items = [_numpyify(item) for item in value]
        try:
            array = np.asarray(items)
        except (TypeError, ValueError):
            return items
        return array if array.dtype != object else items
    return value


def _restore_initial_conditions(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise ValueError("initial conditions must be a non-empty list")
    return [
        {
            **initial,
            "variation_values": _numpyify(initial["variation_values"]),
            "state": np.asarray(initial["state"], dtype=np.float64),
            "goal": np.asarray(initial["goal"], dtype=np.float32),
        }
        for initial in records
    ]


def _step_label(info: dict[str, Any], env: gym.Env) -> dict[str, Any]:
    block = np.asarray(info.get("privileged/block_0_pos", [np.nan] * 3), dtype=np.float64).reshape(-1)
    goal = np.asarray(_goal_position(env), dtype=np.float64).reshape(-1)
    return {
        "contact": float(np.max(np.asarray(info.get("proprio/gripper_contact", [0.0]), dtype=np.float64))),
        "block_goal_distance_m": float(np.linalg.norm(block[:2] - goal[:2]))
        if block.size >= 2 and goal.size >= 2 else None,
        "block_z": float(block[2]) if block.size >= 3 else None,
    }


def _approach_action(
    image: np.ndarray,
    *,
    pixel_mapping: np.ndarray | None,
    approach_model: dict[str, Any] | None,
    goal_image: np.ndarray,
    controller: str,
    pixel_gain: float,
    world_gain: float,
    spatial_mapping: np.ndarray | None = None,
    env: gym.Env | None = None,
) -> np.ndarray:
    """Return an RGB-only XY approach action.

    ``hybrid`` is deliberately a fixed, calibration-independent blend.  It
    does not inspect contact or task outcome: the world chart supplies the
    learned-coordinate proposal while the pixel Jacobian supplies a local
    image-space correction.  This is a mechanism arm for diagnosing chart
    extrapolation, not evidence for a learned world-model controller.
    """
    if controller not in {"pixel", "world", "hybrid", "spatial"}:
        raise ValueError(f"unknown approach controller: {controller}")
    if controller == "spatial" and spatial_mapping is not None and env is not None:
        error = _multiview_effector_error(env, image)
        if error is not None:
            # mapping is action -> [front_xy, side_xy]. Solve the calibrated
            # joint-view least-squares inverse, then apply a fixed gain.
            action = np.linalg.lstsq(
                np.asarray(spatial_mapping, dtype=np.float64).T,
                -error,
                rcond=1.0e-6,
            )[0]
            norm = float(np.linalg.norm(action))
            if np.isfinite(norm) and norm > 1.0e-8:
                action = float(pixel_gain) * action / max(1.0, norm)
                output = np.zeros(5, dtype=np.float32)
                output[:2] = np.clip(action[:2], -1.0, 1.0)
                return output
        return _rgb_approach_action(image, pixel_mapping, pixel_gain)
    pixel_action = _rgb_approach_action(image, pixel_mapping, pixel_gain)
    if controller == "pixel" or approach_model is None:
        return pixel_action
    block = _target_red_centroid(image)[:2]
    effector = _largest_effector_centroid(image)[:2]
    goal_xy = _target_red_centroid(goal_image)[:2]
    feature = _relative_approach_features(
        effector, block, goal_xy,
        family=str(approach_model["approach_feature_family"]),
    )
    world_error = _predict_linear(feature, approach_model["relative_approach_chart"])
    world_action = np.zeros(5, dtype=np.float32)
    world_action[:2] = _action_from_world_error(
        world_error,
        approach_model["world_action_mapping_per_step"],
        world_gain,
    )
    if controller == "world":
        return world_action
    # Fixed equal blend keeps both charts active and bounds the action.
    hybrid = 0.5 * world_action + 0.5 * pixel_action
    hybrid[:2] = np.clip(hybrid[:2], -1.0, 1.0)
    return hybrid


def _model_for_task(
    approach_model: dict[str, Any] | None,
    task_models: dict[int, dict[str, Any]] | None,
    task_id: int,
) -> dict[str, Any] | None:
    """Select a calibration-only task chart, with a global fallback."""
    if task_models:
        return task_models.get(int(task_id), approach_model)
    return approach_model


def _fit_effect(
    actions: list[np.ndarray], deltas: list[np.ndarray],
    *, min_mapping_norm: float = 1.0e-3,
    max_normalized_residual: float = 0.75,
    max_condition_number: float = 100.0,
) -> tuple[np.ndarray | None, int]:
    """Fit a carried-object effect map and reject weak/unstable fits.

    Rank alone is insufficient: zero-effect probes and inconsistent contact
    modes can still produce a full-rank least-squares matrix.  The thresholds
    are frozen protocol guards, not outcome-tuned controller parameters.
    """
    if len(actions) < 4:
        return None, 0
    x = np.asarray(actions, dtype=np.float64).reshape(-1, 2)
    y = np.asarray(deltas, dtype=np.float64).reshape(-1, 2)
    rank = int(np.linalg.matrix_rank(x))
    if rank < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None, rank
    singular_values = np.linalg.svd(x, compute_uv=False)
    condition_number = float(singular_values[0] / max(singular_values[-1], 1.0e-12))
    if not np.isfinite(condition_number) or condition_number > float(max_condition_number):
        return None, rank
    mapping, _, _, _ = np.linalg.lstsq(x, y, rcond=1.0e-6)
    if not np.isfinite(mapping).all():
        return None, rank
    mapping_norm = float(np.linalg.norm(mapping, ord="fro"))
    if not np.isfinite(mapping_norm) or mapping_norm < float(min_mapping_norm):
        return None, rank
    residual = x @ mapping - y
    scale = max(float(np.sqrt(np.mean(y ** 2))), 1.0e-8)
    normalized_residual = float(np.sqrt(np.mean(residual ** 2)) / scale)
    if not np.isfinite(normalized_residual) or normalized_residual > float(max_normalized_residual):
        return None, rank
    return mapping, rank


def _validate_effect_probe_protocol(
    transport_controller: str, lift_before_probes: bool,
) -> None:
    """Reject a carry-effect fit collected before the object is lifted.

    The transport controller consumes an action-to-object-effect map.  Probing
    before lift can instead measure free gripper motion or incidental contact,
    which is a different operator and was a direct source of root-dependent
    false mappings in the earlier mechanism arm.
    """
    if transport_controller in {"effect", "hybrid", "greedy_effect"} and not lift_before_probes:
        raise ValueError(
            f"{transport_controller} transport requires --lift-before-probes; "
            "pre-lift RGB probes are not a valid carried-object effect map"
        )


def _fit_local_rgb_pixel_mapping(
    initial: dict[str, Any], descent_steps: int, probe_steps: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Fit a deployment-root RGB effector Jacobian from RGB probes only.

    Each probe starts from the frozen initial condition and the actual episode
    is reset again afterwards.  No simulator state, contact label, goal cost,
    or task outcome enters the fit.  This isolates camera/action-chart
    extrapolation from the global calibration panel.
    """
    actions: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    directions = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    env = _make_env()
    try:
        for direction in directions:
            obs, _info, _goal = _reset(env, initial)
            for _ in range(int(descent_steps)):
                action = np.zeros(5, dtype=np.float32)
                action[2] = -1.0
                obs, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    break
            before = _largest_effector_centroid(np.asarray(obs, dtype=np.uint8))[:2]
            if np.linalg.norm(before) <= 0.0:
                continue
            action = np.zeros(5, dtype=np.float32)
            action[:2] = direction
            for _ in range(int(probe_steps)):
                obs, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    break
            after = _largest_effector_centroid(np.asarray(obs, dtype=np.uint8))[:2]
            if np.linalg.norm(after) <= 0.0:
                continue
            actions.append(direction.astype(np.float64))
            deltas.append((after - before).astype(np.float64))
    finally:
        env.close()
    if len(actions) < 4:
        return None, {"status": "insufficient_local_probe_rows", "rows": len(actions), "rank": 0}
    x = np.asarray(actions, dtype=np.float64)
    y = np.asarray(deltas, dtype=np.float64)
    mapping, _, rank, singular_values = np.linalg.lstsq(x, y, rcond=1.0e-6)
    residual = x @ mapping - y
    if int(rank) < 2 or not np.isfinite(mapping).all():
        return None, {
            "status": "rank_deficient_local_probe", "rows": len(actions), "rank": int(rank),
        }
    return mapping.astype(np.float32), {
        "status": "ok", "rows": len(actions), "rank": int(rank),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "fit_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mapping": mapping.astype(np.float32).tolist(),
    }


def _fit_local_rgb_multiview_mapping(
    initial: dict[str, Any], descent_steps: int, probe_steps: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Fit an RGB-only action-to-joint-view Jacobian for spatial approach.

    The output maps two planar action coordinates to four image coordinates:
    front-x/front-y and side-x/side-y.  It is a compact, calibrated analogue
    of Evo-Depth's spatial enhancement branch; no simulator state or task
    outcome enters the fit.
    """
    actions: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    directions = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32,
    )
    env = _make_env()
    try:
        for direction in directions:
            obs, _info, _goal = _reset(env, initial)
            for _ in range(int(descent_steps)):
                action = np.zeros(5, dtype=np.float32)
                action[2] = -1.0
                obs, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    break
            before = _multiview_effector_error(env, np.asarray(obs, dtype=np.uint8))
            if before is None:
                continue
            action = np.zeros(5, dtype=np.float32)
            action[:2] = direction
            for _ in range(int(probe_steps)):
                obs, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    break
            after = _multiview_effector_error(env, np.asarray(obs, dtype=np.uint8))
            if after is None:
                continue
            actions.append(direction.astype(np.float64))
            deltas.append((after - before).astype(np.float64))
    finally:
        env.close()
    if len(actions) < 4:
        return None, {"status": "insufficient_multiview_probe_rows", "rows": len(actions), "rank": 0}
    x = np.asarray(actions, dtype=np.float64)
    y = np.asarray(deltas, dtype=np.float64)
    mapping, _, rank, singular_values = np.linalg.lstsq(x, y, rcond=1.0e-6)
    residual = x @ mapping - y
    if int(rank) < 2 or not np.isfinite(mapping).all():
        return None, {"status": "rank_deficient_multiview_probe", "rows": len(actions), "rank": int(rank)}
    return mapping.astype(np.float32), {
        "status": "ok", "rows": len(actions), "rank": int(rank),
        "condition_number": float(singular_values[0] / max(singular_values[-1], 1.0e-12)),
        "fit_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mapping": mapping.astype(np.float32).tolist(),
    }


def _greedy_effect_action(
    desired: np.ndarray, mapping: np.ndarray | None, magnitude: float,
) -> np.ndarray | None:
    """Choose a bounded action from the observed contact-support directions."""
    if mapping is None or not np.isfinite(mapping).all():
        return None
    directions = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
         [0.70710678, 0.70710678], [-0.70710678, 0.70710678],
         [0.70710678, -0.70710678], [-0.70710678, -0.70710678]],
        dtype=np.float64,
    )
    desired = np.asarray(desired, dtype=np.float64).reshape(2)
    predicted = directions @ np.asarray(mapping, dtype=np.float64)
    scores = predicted @ desired
    if not np.isfinite(scores).all() or float(np.max(scores)) <= 1.0e-8:
        return None
    return (float(magnitude) * directions[int(np.argmax(scores))]).astype(np.float32)


def _inverse_effect_action(
    desired: np.ndarray, mapping: np.ndarray | None, magnitude: float,
) -> np.ndarray | None:
    """Invert an action-to-RGB-effect chart into a bounded XY action."""
    if mapping is None or not np.isfinite(mapping).all():
        return None
    desired = np.asarray(desired, dtype=np.float64).reshape(2)
    if not np.isfinite(desired).all() or np.linalg.norm(desired) <= 1.0e-8:
        return None
    proposal = desired @ np.linalg.pinv(np.asarray(mapping, dtype=np.float64))
    norm = float(np.linalg.norm(proposal))
    if norm <= 1.0e-8 or not np.isfinite(norm):
        return None
    proposal = proposal / max(1.0, norm / float(magnitude))
    return proposal.astype(np.float32)


def _run_episode(
    *, seed: int, task_id: int, pixel_mapping: np.ndarray | None,
    spatial_mapping: np.ndarray | None,
    approach_model: dict[str, Any] | None,
    initial: dict[str, Any] | None,
    open_steps: int, descent_steps: int, depth_aware_contact: bool,
    side_gap_threshold: float, side_extra_descent_steps: int,
    approach_steps: int, approach_gain: float,
    approach_threshold: float, approach_min_steps: int,
    approach_controller: str, approach_world_gain: float,
    approach_task_models: dict[int, dict[str, Any]] | None,
    approach_before_descend: bool,
    post_descend_steps: int, close_steps: int,
    post_descend_approach_steps: int,
    post_close_reapproach_steps: int,
    post_close_reapproach_pixel_threshold: float,
    lift_before_probes: bool,
    grasp_lift_steps: int, lift_steps: int, probe_magnitude: float, transport_steps: int,
    transport_magnitude: float, lower_steps: int, release_steps: int,
    goal_threshold: float, transport_lift_hold: bool,
    transport_lift_hold_magnitude: float,
    transport_lift_correction_gain: float, transport_lift_correction_period: int,
    transport_controller: str,
    lower_transport_magnitude: float,
    carry_vertical_servo_gain: float, carry_vertical_servo_deadband: float,
    carry_visual_scale_servo_gain: float,
    relift_interval: int, relift_steps: int,
    regrasp_on_effect_loss: bool, effect_loss_threshold: float,
    effect_loss_alignment_threshold: float,
    effect_loss_gap_threshold: float,
    effect_loss_patience: int, regrasp_approach_steps: int,
    regrasp_descend_steps: int, regrasp_close_steps: int,
    regrasp_lift_steps: int,
) -> dict[str, Any]:
    env = _make_env()
    try:
        reset_options: dict[str, Any] = {"task_id": int(task_id), "render_goal": True}
        if initial is not None:
            reset_options.update({
                "variation_values": copy.deepcopy(initial["variation_values"]),
                "state": np.asarray(initial["state"], dtype=np.float64),
            })
        obs, info = env.reset(seed=int(seed), options=reset_options)
        goal_image = np.asarray(env.unwrapped._cur_goal_rendered, dtype=np.uint8).copy()
        contact_trace: list[float] = []
        phases: list[str] = []
        label_trace: list[dict[str, Any]] = []
        effect_actions: list[np.ndarray] = []
        effect_deltas: list[np.ndarray] = []
        error_trace: list[float] = []
        carry_vertical_error_trace: list[float] = []
        carry_visual_scale_trace: list[dict[str, float]] = []
        effect_loss_trace: list[dict[str, float | int]] = []
        transport_action_source_trace: list[str] = []
        effect_map_enabled = True
        effect_map_rejection_reason: str | None = None
        regrasp_count = 0
        previous_approach_error: float | None = None

        def execute(action: np.ndarray, phase: str) -> bool:
            nonlocal obs, info
            before = np.asarray(obs, dtype=np.uint8).copy()
            obs, _reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            phases.append(phase)
            contact_trace.append(float(np.max(np.asarray(
                info.get("proprio/gripper_contact", [0.0]), dtype=np.float32,
            ))))
            label = {"phase": phase, **_step_label(info, env)}
            rgb_block = _target_red_centroid(np.asarray(obs, dtype=np.uint8))[:2]
            rgb_goal = _target_red_centroid(goal_image)[:2]
            label["rgb_goal_distance_px"] = (
                float(np.linalg.norm(rgb_block - rgb_goal))
                if np.linalg.norm(rgb_block) > 0.0 and np.linalg.norm(rgb_goal) > 0.0
                else None
            )
            label["rgb_block_area_fraction"] = _target_red_area(np.asarray(obs, dtype=np.uint8))
            rgb_effector = _largest_effector_centroid(np.asarray(obs, dtype=np.uint8))[:2]
            label["rgb_block_effector_gap"] = (
                float(np.linalg.norm(rgb_block - rgb_effector))
                if np.linalg.norm(rgb_block) > 0.0 and np.linalg.norm(rgb_effector) > 0.0
                else None
            )
            if depth_aware_contact:
                label["side_vertical_gap"] = _side_vertical_gap(env)
            label_trace.append(label)
            return bool(terminated or truncated)

        def run_rgb_approach(steps: int, phase: str) -> None:
            nonlocal previous_approach_error
            previous_approach_error = None
            for approach_index in range(int(steps)):
                image = np.asarray(obs, dtype=np.uint8)
                block = _target_red_centroid(image)[:2]
                effector = _largest_effector_centroid(image)[:2]
                error = float(np.linalg.norm(block - effector))
                error_trace.append(error)
                if (
                    error <= float(approach_threshold)
                    and approach_index + 1 >= int(approach_min_steps)
                ):
                    break
                selected_controller = approach_controller
                if (
                    approach_controller == "adaptive"
                    and previous_approach_error is not None
                    and error >= previous_approach_error - 1.0e-3
                ):
                    selected_controller = "pixel"
                elif approach_controller == "adaptive":
                    selected_controller = "hybrid"
                action = _approach_action(
                    image, pixel_mapping=pixel_mapping, spatial_mapping=spatial_mapping,
                    env=env,
                    approach_model=_model_for_task(
                        approach_model, approach_task_models, task_id,
                    ),
                    goal_image=goal_image, controller=selected_controller,
                    pixel_gain=approach_gain, world_gain=approach_world_gain,
                )
                previous_approach_error = error
                if execute(action, phase):
                    break

        # Reset starts with normalized gripper opening near zero. Open before
        # descending so the later close command can establish a carry.
        for _ in range(int(open_steps)):
            action = np.zeros(5, dtype=np.float32)
            action[4] = GRIPPER_OPEN_ACTION
            if execute(action, "pre_open"):
                break

        if approach_before_descend:
            run_rgb_approach(approach_steps, "rgb_approach_above")

        for _ in range(int(descent_steps)):
            action = np.zeros(5, dtype=np.float32); action[2] = -1.0
            if execute(action, "pre_descend"): break
        if depth_aware_contact:
            # A fixed number of descent pulses is not spatially invariant
            # across camera/height variations.  Use the side projection as a
            # conservative contact gate, with a hard declared budget.
            for _ in range(int(side_extra_descent_steps)):
                gaps = [row.get("side_vertical_gap") for row in label_trace]
                gap = gaps[-1] if gaps else None
                if gap is not None and gap <= float(side_gap_threshold):
                    break
                action = np.zeros(5, dtype=np.float32); action[2] = -1.0
                if execute(action, "depth_aware_descend"): break
        if not phases:
            return {"seed": seed, "task_id": task_id, "success": False, "phase_trace": phases}

        if not approach_before_descend:
            run_rgb_approach(approach_steps, "rgb_approach")

        for _ in range(int(post_descend_steps)):
            action = np.zeros(5, dtype=np.float32); action[2] = -1.0
            if execute(action, "post_descend"): break
        run_rgb_approach(post_descend_approach_steps, "post_descend_approach")
        for _ in range(int(grasp_lift_steps)):
            action = np.zeros(5, dtype=np.float32); action[2] = 1.0; action[4] = GRIPPER_CLOSE_ACTION
            if execute(action, "grasp_lift"): break
        for _ in range(int(close_steps)):
            action = np.zeros(5, dtype=np.float32); action[4] = GRIPPER_CLOSE_ACTION
            if execute(action, "close"): break
        # A close command can leave the gripper visually aligned but not
        # mechanically engaged.  This fixed RGB-only re-approach is a
        # contact-mode intervention; it is deliberately before effect probes
        # so the probes estimate the post-engagement action interface.
        close_rows = [row for row in label_trace if row["phase"] == "close"]
        close_rgb_error = close_rows[-1].get("rgb_goal_distance_px") if close_rows else None
        enable_post_close = (
            int(post_close_reapproach_steps) > 0
            and (
                float(post_close_reapproach_pixel_threshold) <= 0.0
                or (
                    close_rgb_error is not None
                    and close_rgb_error <= float(post_close_reapproach_pixel_threshold)
                )
            )
        )
        previous_approach_error = None
        for _ in range(int(post_close_reapproach_steps) if enable_post_close else 0):
            image = np.asarray(obs, dtype=np.uint8)
            action = _approach_action(
                image, pixel_mapping=pixel_mapping, spatial_mapping=spatial_mapping,
                env=env,
                approach_model=_model_for_task(approach_model, approach_task_models, task_id),
                goal_image=goal_image,
                controller=("hybrid" if approach_controller == "adaptive" else approach_controller),
                pixel_gain=approach_gain, world_gain=approach_world_gain,
            )
            if execute(action, "post_close_reapproach"): break
        if lift_before_probes:
            for _ in range(int(lift_steps)):
                action = np.zeros(5, dtype=np.float32); action[2] = 1.0; action[4] = GRIPPER_CLOSE_ACTION
                if execute(action, "lift"): break

        # Probe target RGB displacement in all four directions while holding.
        for direction in np.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float32):
            for _ in range(1):
                before = np.asarray(obs, dtype=np.uint8).copy()
                before_block = _target_red_centroid(before)[:2]
                action = np.zeros(5, dtype=np.float32)
                action[:2] = float(probe_magnitude) * direction
                action[4] = GRIPPER_CLOSE_ACTION
                if execute(action, "effect_probe"): break
                after_block = _target_red_centroid(np.asarray(obs, dtype=np.uint8))[:2]
                if np.linalg.norm(before_block) > 0.0 and np.linalg.norm(after_block) > 0.0:
                    effect_actions.append(action[:2].astype(np.float64))
                    effect_deltas.append((after_block - before_block).astype(np.float64))

        if not lift_before_probes:
            for _ in range(int(lift_steps)):
                action = np.zeros(5, dtype=np.float32); action[2] = 1.0; action[4] = GRIPPER_CLOSE_ACTION
                if execute(action, "lift"): break

        def run_regrasp() -> None:
            """Attempt one RGB-only recovery after observed effect loss.

            The trigger uses only executed RGB block displacement.  It is
            intentionally a single bounded intervention: repeated retries
            would turn an unidentifiable carry mode into an uncontrolled
            action loop.
            """
            nonlocal regrasp_count
            if regrasp_count >= 1:
                return
            regrasp_count += 1
            for _ in range(3):
                action = np.zeros(5, dtype=np.float32); action[4] = GRIPPER_OPEN_ACTION
                if execute(action, "regrasp_release"): return
            run_rgb_approach(regrasp_approach_steps, "regrasp_approach")
            for _ in range(int(regrasp_descend_steps)):
                action = np.zeros(5, dtype=np.float32); action[2] = -1.0
                if execute(action, "regrasp_descend"): return
            for _ in range(int(regrasp_close_steps)):
                action = np.zeros(5, dtype=np.float32); action[4] = GRIPPER_CLOSE_ACTION
                if execute(action, "regrasp_close"): return
            for _ in range(int(regrasp_lift_steps)):
                action = np.zeros(5, dtype=np.float32)
                action[2] = 1.0; action[4] = GRIPPER_CLOSE_ACTION
                if execute(action, "regrasp_lift"): return
            for direction in np.asarray(
                [[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float32,
            ):
                before = _target_red_centroid(np.asarray(obs, dtype=np.uint8))[:2]
                action = np.zeros(5, dtype=np.float32)
                action[:2] = float(probe_magnitude) * direction
                action[4] = GRIPPER_CLOSE_ACTION
                if execute(action, "regrasp_effect_probe"): return
                after = _target_red_centroid(np.asarray(obs, dtype=np.uint8))[:2]
                if np.linalg.norm(before) > 0.0 and np.linalg.norm(after) > 0.0:
                    effect_actions.append(action[:2].astype(np.float64))
                    effect_deltas.append((after - before).astype(np.float64))

        carry_reference = None
        carry_rgb_gap_reference: float | None = None
        carry_rgb_area_reference: float | None = None
        carry_rgb_area_slope: float | None = None
        reference_block = _target_red_centroid(np.asarray(obs, dtype=np.uint8))[:2]
        reference_effector = _largest_effector_centroid(np.asarray(obs, dtype=np.uint8))[:2]
        if np.linalg.norm(reference_block) > 0.0 and np.linalg.norm(reference_effector) > 0.0:
            carry_reference = float(reference_block[1] - reference_effector[1])
            carry_rgb_gap_reference = float(np.linalg.norm(reference_block - reference_effector))

        if float(carry_visual_scale_servo_gain) > 0.0:
            area_before = _target_red_area(np.asarray(obs, dtype=np.uint8))
            scale_probe = np.zeros(5, dtype=np.float32)
            scale_probe[2] = 0.05
            scale_probe[4] = 1.0
            if not execute(scale_probe, "carry_scale_probe"):
                area_after = _target_red_area(np.asarray(obs, dtype=np.uint8))
                carry_rgb_area_slope = (area_after - area_before) / 0.05
                if abs(carry_rgb_area_slope) >= 1.0e-5:
                    carry_rgb_area_reference = area_after

        effect_loss_streak = 0
        for step_index in range(int(transport_steps)):
            current = _target_red_centroid(np.asarray(obs, dtype=np.uint8))[:2]
            target = _target_red_centroid(goal_image)[:2]
            desired = target - current
            effect_mapping, rank = _fit_effect(effect_actions, effect_deltas)
            action = np.zeros(5, dtype=np.float32)
            chart_action = _inverse_effect_action(desired, pixel_mapping, transport_magnitude)
            effect_action = (
                _inverse_effect_action(desired, effect_mapping, transport_magnitude)
                if effect_map_enabled else None
            )
            action_source = "direct_goal_direction"
            if transport_controller == "greedy_effect":
                greedy = _greedy_effect_action(desired, effect_mapping, transport_magnitude)
                if greedy is not None:
                    action[:2] = greedy
                    action_source = "greedy_effect"
                elif chart_action is not None:
                    action[:2] = chart_action
                    action_source = "rgb_effector_chart_fallback"
            elif transport_controller == "effect" and effect_action is not None:
                action[:2] = effect_action
                action_source = "carried_object_effect"
            elif transport_controller == "effect" and chart_action is not None:
                # Keep the intervention paired: an invalid carried-object map
                # falls back to the same RGB chart as the baseline, never to a
                # separately fitted world chart.
                action[:2] = chart_action
                action_source = "rgb_effector_chart_fallback"
            elif transport_controller == "hybrid":
                if effect_action is not None and chart_action is not None:
                    action[:2] = 0.5 * (effect_action + chart_action)
                    action[:2] = np.clip(
                        action[:2], -float(transport_magnitude), float(transport_magnitude),
                    )
                    action_source = "hybrid_effect_chart"
                elif effect_action is not None:
                    action[:2] = effect_action
                    action_source = "carried_object_effect_fallback"
                elif chart_action is not None:
                    action[:2] = chart_action
                    action_source = "rgb_effector_chart_fallback"
            elif transport_controller == "chart" and chart_action is not None:
                action[:2] = chart_action
                action_source = "rgb_effector_chart"
            elif transport_controller == "direct_pixel":
                desired_norm = float(np.linalg.norm(desired))
                if desired_norm > 1.0e-8:
                    action[:2] = (
                        float(transport_magnitude) * desired / desired_norm
                    ).astype(np.float32)
                    action_source = "direct_block_goal_pixel"
            if not np.any(action[:2]) and approach_model is not None:
                world_goal = _predict_linear(
                    _relative_push_features(current, target),
                    approach_model["relative_push_chart"],
                )
                world_norm = float(np.linalg.norm(world_goal))
                if world_norm > 1.0e-8:
                    action[:2] = (
                        float(transport_magnitude) * world_goal / world_norm
                    ).astype(np.float32)
                action_source = "world_chart_fallback"
            if not np.any(action[:2]):
                direction = desired / max(float(np.linalg.norm(desired)), 1.0e-8)
                action[:2] = (float(transport_magnitude) * direction).astype(np.float32)
                action_source = "direct_goal_direction"
            transport_action_source_trace.append(action_source)
            # A full-height command on every transport step breaks the grasp.
            # Use a sparse, low-amplitude correction instead: the controller
            # has no privileged height input, so cadence is fixed on the
            # calibration split and remains a declared action primitive.
            if (
                carry_rgb_area_reference is not None
                and carry_rgb_area_slope is not None
            ):
                current_area = _target_red_area(np.asarray(obs, dtype=np.uint8))
                area_error = carry_rgb_area_reference - current_area
                action[2] = float(np.clip(
                    float(carry_visual_scale_servo_gain) * area_error / carry_rgb_area_slope,
                    -0.25, 0.25,
                ))
                carry_visual_scale_trace.append({
                    "area": current_area,
                    "area_error": area_error,
                    "z_action": float(action[2]),
                })
            elif transport_lift_hold:
                action[2] = float(transport_lift_hold_magnitude)
            elif carry_reference is not None and float(carry_vertical_servo_gain) > 0.0:
                current_effector = _largest_effector_centroid(np.asarray(obs, dtype=np.uint8))[:2]
                current_block = _target_red_centroid(np.asarray(obs, dtype=np.uint8))[:2]
                if np.linalg.norm(current_block) > 0.0 and np.linalg.norm(current_effector) > 0.0:
                    vertical_error = float(
                        (current_block[1] - current_effector[1]) - carry_reference
                    )
                    carry_vertical_error_trace.append(vertical_error)
                    if abs(vertical_error) > float(carry_vertical_servo_deadband):
                        action[2] = float(np.clip(
                            float(carry_vertical_servo_gain) * vertical_error,
                            -0.5, 0.5,
                        ))
            elif (
                float(transport_lift_correction_gain) > 0.0
                and int(transport_lift_correction_period) > 0
                and step_index % int(transport_lift_correction_period) == 0
            ):
                action[2] = float(transport_lift_correction_gain)
            action[4] = GRIPPER_CLOSE_ACTION
            before = np.asarray(obs, dtype=np.uint8).copy()
            if execute(action, "transport"): break
            after = np.asarray(obs, dtype=np.uint8)
            before_block = _target_red_centroid(before)[:2]
            after_block = _target_red_centroid(after)[:2]
            observed_delta = after_block - before_block
            observed_effect = float(np.linalg.norm(observed_delta))
            desired_norm = float(np.linalg.norm(desired))
            observed_alignment = (
                float(np.dot(observed_delta, desired) / (observed_effect * desired_norm))
                if observed_effect > 1.0e-8 and desired_norm > 1.0e-8 else -1.0
            )
            after_effector = _largest_effector_centroid(after)[:2]
            after_gap = (
                float(np.linalg.norm(after_block - after_effector))
                if np.linalg.norm(after_block) > 0.0 and np.linalg.norm(after_effector) > 0.0
                else float("inf")
            )
            gap_loss = (
                carry_rgb_gap_reference is not None
                and after_gap - carry_rgb_gap_reference > float(effect_loss_gap_threshold)
            )
            if np.linalg.norm(before_block) > 0.0 and np.linalg.norm(after_block) > 0.0:
                effect_actions.append(action[:2].astype(np.float64))
                effect_deltas.append((after_block - before_block).astype(np.float64))
            if (
                np.linalg.norm(before_block) <= 0.0
                or np.linalg.norm(after_block) <= 0.0
                or observed_effect < float(effect_loss_threshold)
                or observed_alignment < float(effect_loss_alignment_threshold)
                or gap_loss
            ) and np.linalg.norm(desired) > float(goal_threshold):
                effect_loss_streak += 1
            else:
                effect_loss_streak = 0
            effect_loss_trace.append({
                "step": int(step_index),
                "observed_effect": observed_effect,
                "observed_alignment": observed_alignment,
                "rgb_block_effector_gap": after_gap,
                "rgb_gap_loss": bool(gap_loss),
                "streak": int(effect_loss_streak),
            })
            # A static fit can be numerically well-conditioned yet point in
            # the wrong direction for this carried root.  Validate the first
            # executed effect online using only RGB displacement.  Once
            # rejected, the effect map is never re-enabled in this episode;
            # the paired RGB chart remains the fail-safe controller.
            if (
                transport_controller in {"effect", "hybrid", "greedy_effect"}
                and effect_map_enabled
                and desired_norm > float(goal_threshold)
                and (
                    observed_effect < float(effect_loss_threshold)
                    or observed_alignment < float(effect_loss_alignment_threshold)
                )
            ):
                effect_map_enabled = False
                effect_map_rejection_reason = (
                    "online_rgb_effect_validation_failed"
                )
            if (
                regrasp_on_effect_loss
                and effect_loss_streak >= int(effect_loss_patience)
                and regrasp_count == 0
            ):
                run_regrasp()
                effect_loss_streak = 0
            error_trace.append(float(np.linalg.norm(_target_red_centroid(after)[:2] - target)))
            if error_trace[-1] <= float(goal_threshold):
                break
            if (
                int(relift_interval) > 0
                and int(relift_steps) > 0
                and (step_index + 1) % int(relift_interval) == 0
                and step_index + 1 < int(transport_steps)
            ):
                for _ in range(int(relift_steps)):
                    relift = np.zeros(5, dtype=np.float32)
                    relift[2] = 1.0
                    relift[4] = 1.0
                    if execute(relift, "transport_relift"): break

        for _ in range(int(lower_steps)):
            action = np.zeros(5, dtype=np.float32); action[2] = -1.0; action[4] = GRIPPER_CLOSE_ACTION
            lower_mapping, _lower_rank = _fit_effect(effect_actions, effect_deltas)
            if lower_mapping is not None and float(lower_transport_magnitude) > 0.0:
                current = _target_red_centroid(np.asarray(obs, dtype=np.uint8))[:2]
                target = _target_red_centroid(goal_image)[:2]
                desired = target - current
                if np.linalg.norm(desired) > 1.0e-8:
                    lower_action = _inverse_effect_action(
                        desired, lower_mapping, lower_transport_magnitude,
                    )
                    if lower_action is not None:
                        action[:2] = lower_action
            if execute(action, "lower"): break
        for _ in range(int(release_steps)):
            action = np.zeros(5, dtype=np.float32); action[4] = GRIPPER_OPEN_ACTION
            if execute(action, "release"): break
        final = _step_label(info, env)
        success = bool(final["block_goal_distance_m"] is not None and final["block_goal_distance_m"] <= float(goal_threshold))
        return {
            "seed": int(seed), "task_id": int(task_id), "success": success,
            "final": final, "contact_observed": bool(any(v > 0.5 for v in contact_trace)),
            "effect_rows": len(effect_actions), "effect_rank": int(_fit_effect(effect_actions, effect_deltas)[1]),
            "approach_steps_used": int(sum(p == "rgb_approach" for p in phases)),
            "phase_trace": phases, "error_trace": error_trace,
            "carry_vertical_error_trace": carry_vertical_error_trace,
            "effect_loss_trace": effect_loss_trace,
            "transport_action_source_trace": transport_action_source_trace,
            "effect_map_rejection_reason": effect_map_rejection_reason,
            "effect_map_enabled_final": bool(effect_map_enabled),
            "carry_visual_scale_trace": carry_visual_scale_trace,
            "carry_rgb_area_reference": carry_rgb_area_reference,
            "carry_rgb_area_slope": carry_rgb_area_slope,
            "regrasp_count": int(regrasp_count),
            "label_trace": label_trace,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--initial-conditions", type=Path, default=None,
        help="replay frozen initial_conditions from a previous JSON artifact",
    )
    parser.add_argument("--seed", type=int, default=20261901)
    parser.add_argument("--test-roots", type=int, default=3)
    parser.add_argument("--calibration-roots", type=int, default=8)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--descent-steps", type=int, default=15)
    parser.add_argument(
        "--depth-aware-contact", action="store_true",
        help="use side RGB projection as a conservative implicit-depth contact gate",
    )
    parser.add_argument("--side-gap-threshold", type=float, default=0.04)
    parser.add_argument("--side-extra-descent-steps", type=int, default=0)
    parser.add_argument(
        "--open-steps", type=int, default=5,
        help="fixed gripper-opening prelude before descent; action[4]=-1",
    )
    parser.add_argument("--approach-steps", type=int, default=64)
    parser.add_argument("--approach-gain", type=float, default=0.25)
    parser.add_argument("--approach-threshold", type=float, default=0.05)
    parser.add_argument(
        "--approach-min-steps", type=int, default=0,
        help="minimum RGB approach steps before the centroid threshold can stop",
    )
    parser.add_argument(
        "--approach-controller", choices=("pixel", "world", "hybrid", "adaptive", "spatial"),
        default="world",
    )
    parser.add_argument(
        "--approach-before-descend", action="store_true",
        help="align XY at the elevated start pose before descending and closing",
    )
    parser.add_argument(
        "--task-conditioned-approach", action="store_true",
        help="fit one approach chart per task using calibration roots only",
    )
    parser.add_argument(
        "--local-pixel-approach", action="store_true",
        help="fit a per-deployment-root RGB Jacobian from reset-safe local probes",
    )
    parser.add_argument(
        "--fit-device", choices=("auto", "cpu", "cuda"), default="auto",
        help="device for calibration chart fits; fixed device improves replay reproducibility",
    )
    parser.add_argument("--approach-world-gain", type=float, default=1.0)
    parser.add_argument("--post-descend-steps", type=int, default=8)
    parser.add_argument("--post-descend-approach-steps", type=int, default=0)
    parser.add_argument("--post-close-reapproach-steps", type=int, default=0)
    parser.add_argument("--post-close-reapproach-pixel-threshold", type=float, default=0.0)
    parser.add_argument("--lift-before-probes", action="store_true")
    parser.add_argument("--close-steps", type=int, default=5)
    parser.add_argument("--grasp-lift-steps", type=int, default=0)
    parser.add_argument("--lift-steps", type=int, default=15)
    parser.add_argument("--probe-magnitude", type=float, default=0.15)
    parser.add_argument("--transport-steps", type=int, default=50)
    parser.add_argument("--transport-magnitude", type=float, default=0.2)
    parser.add_argument("--lower-steps", type=int, default=15)
    parser.add_argument("--release-steps", type=int, default=8)
    parser.add_argument("--goal-threshold", type=float, default=0.05)
    parser.add_argument("--transport-lift-hold", action="store_true")
    parser.add_argument(
        "--transport-lift-hold-magnitude", type=float, default=1.0,
        help="continuous vertical command used with --transport-lift-hold",
    )
    parser.add_argument("--transport-lift-correction-gain", type=float, default=0.0)
    parser.add_argument("--transport-lift-correction-period", type=int, default=5)
    parser.add_argument(
        "--transport-controller",
        choices=("chart", "direct_pixel", "effect", "hybrid", "greedy_effect"),
        default="chart",
    )
    parser.add_argument("--lower-transport-magnitude", type=float, default=0.0)
    parser.add_argument("--carry-vertical-servo-gain", type=float, default=0.0)
    parser.add_argument("--carry-vertical-servo-deadband", type=float, default=0.02)
    parser.add_argument("--carry-visual-scale-servo-gain", type=float, default=0.0)
    parser.add_argument("--relift-interval", type=int, default=0)
    parser.add_argument("--relift-steps", type=int, default=5)
    parser.add_argument("--regrasp-on-effect-loss", action="store_true")
    parser.add_argument("--effect-loss-threshold", type=float, default=0.002)
    parser.add_argument("--effect-loss-alignment-threshold", type=float, default=0.2)
    parser.add_argument("--effect-loss-gap-threshold", type=float, default=0.08)
    parser.add_argument("--effect-loss-patience", type=int, default=3)
    parser.add_argument("--regrasp-approach-steps", type=int, default=32)
    parser.add_argument("--regrasp-descend-steps", type=int, default=5)
    parser.add_argument("--regrasp-close-steps", type=int, default=5)
    parser.add_argument("--regrasp-lift-steps", type=int, default=5)
    args = parser.parse_args()
    _validate_effect_probe_protocol(args.transport_controller, args.lift_before_probes)
    if args.fit_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--fit-device=cuda requested but CUDA is unavailable")
    if args.fit_device == "cuda":
        # torch.cuda.is_available() can be true while device nodes, context
        # creation, or the EGL renderer are unusable.  Formal direct runs
        # must use the same fail-closed CUDA allocation boundary as the panel.
        require_cuda_allocation("cuda")
    fit_device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.fit_device == "auto" else args.fit_device
    )
    source_payload = (
        None if args.initial_conditions is None
        else json.loads(args.initial_conditions.read_text())
    )
    mapping, probe = _fit_rgb_pixel_action_mapping(
        seed=args.seed, tasks=args.tasks, roots=args.calibration_roots,
        descent_steps=args.descent_steps, probe_steps=4,
    )
    approach_model = None
    approach_task_models = None
    approach_report = None
    if args.approach_controller in {"world", "hybrid", "adaptive"}:
        if (
            source_payload is not None
            and source_payload.get("calibration_initial_conditions")
        ):
            calibration_initials = _restore_initial_conditions(
                source_payload["calibration_initial_conditions"]
            )
            if len(calibration_initials) != args.calibration_roots:
                raise ValueError("calibration-roots must match replay calibration conditions")
        else:
            calibration_initials = [
                _capture_initial(args.seed + index, int(args.tasks[index % len(args.tasks)]))
                for index in range(args.calibration_roots)
            ]
        approach_model, approach_report = _fit_rgb_action_map(
            calibration_initials, 4, args.descent_steps,
            fit_device, 1.0e-2, "bilinear",
        )
        if args.task_conditioned_approach:
            grouped: dict[int, list[dict[str, Any]]] = {}
            for initial in calibration_initials:
                grouped.setdefault(int(initial["task_id"]), []).append(initial)
            approach_task_models = {}
            task_reports: dict[int, Any] = {}
            for task_id, task_initials in grouped.items():
                # Two roots are the minimum for a task-local chart.  If a
                # task has only one calibration root, the global chart is the
                # declared fail-safe rather than fitting a degenerate model.
                if len(task_initials) < 2:
                    continue
                task_model, task_report = _fit_rgb_action_map(
                    task_initials, 4, args.descent_steps,
                    fit_device,
                    1.0e-2, "bilinear",
                )
                approach_task_models[int(task_id)] = task_model
                task_reports[int(task_id)] = task_report
            approach_report = {
                "global": approach_report,
                "task_conditioned": task_reports,
                "task_models": sorted(approach_task_models),
            }
    if args.initial_conditions is None:
        test_initials = [
            _capture_initial(args.seed + 1000 + i, int(args.tasks[i % len(args.tasks)]))
            for i in range(args.test_roots)
        ]
    else:
        test_initials = source_payload.get("initial_conditions", source_payload)
        if args.test_roots != len(test_initials):
            raise ValueError("test-roots must equal the replay initial-condition count")
        test_initials = _restore_initial_conditions(test_initials)
    local_pixel_mappings = [mapping for _ in test_initials]
    local_pixel_reports: list[dict[str, Any]] = []
    if args.local_pixel_approach:
        for initial in test_initials:
            local_mapping, local_report = _fit_local_rgb_pixel_mapping(
                initial, args.descent_steps, 4,
            )
            local_pixel_mappings[len(local_pixel_reports)] = (
                local_mapping if local_mapping is not None else mapping
            )
            local_pixel_reports.append(local_report)
    local_spatial_mappings: list[np.ndarray | None] = [None for _ in test_initials]
    local_spatial_reports: list[dict[str, Any]] = []
    if args.approach_controller == "spatial":
        for index, initial in enumerate(test_initials):
            spatial_mapping, spatial_report = _fit_local_rgb_multiview_mapping(
                initial, args.descent_steps, 4,
            )
            local_spatial_mappings[index] = spatial_mapping
            local_spatial_reports.append(spatial_report)
    rows = [_run_episode(
        seed=int(test_initials[i]["seed"]), task_id=int(test_initials[i]["task_id"]),
        pixel_mapping=local_pixel_mappings[i], spatial_mapping=local_spatial_mappings[i],
        approach_model=approach_model,
        initial=test_initials[i],
        open_steps=args.open_steps, descent_steps=args.descent_steps,
        depth_aware_contact=args.depth_aware_contact,
        side_gap_threshold=args.side_gap_threshold,
        side_extra_descent_steps=args.side_extra_descent_steps,
        approach_steps=args.approach_steps, approach_gain=args.approach_gain,
        approach_threshold=args.approach_threshold,
        approach_min_steps=args.approach_min_steps,
        approach_controller=args.approach_controller,
        approach_world_gain=args.approach_world_gain,
        approach_task_models=approach_task_models,
        approach_before_descend=args.approach_before_descend,
        post_descend_steps=args.post_descend_steps,
        post_descend_approach_steps=args.post_descend_approach_steps,
        post_close_reapproach_steps=args.post_close_reapproach_steps,
        post_close_reapproach_pixel_threshold=args.post_close_reapproach_pixel_threshold,
        lift_before_probes=args.lift_before_probes,
        close_steps=args.close_steps, lift_steps=args.lift_steps,
        grasp_lift_steps=args.grasp_lift_steps,
        probe_magnitude=args.probe_magnitude, transport_steps=args.transport_steps,
        transport_magnitude=args.transport_magnitude, lower_steps=args.lower_steps,
        release_steps=args.release_steps, goal_threshold=args.goal_threshold,
        transport_lift_hold=args.transport_lift_hold,
        transport_lift_hold_magnitude=args.transport_lift_hold_magnitude,
        transport_lift_correction_gain=args.transport_lift_correction_gain,
        transport_lift_correction_period=args.transport_lift_correction_period,
        transport_controller=args.transport_controller,
        lower_transport_magnitude=args.lower_transport_magnitude,
        carry_vertical_servo_gain=args.carry_vertical_servo_gain,
        carry_vertical_servo_deadband=args.carry_vertical_servo_deadband,
        carry_visual_scale_servo_gain=args.carry_visual_scale_servo_gain,
        relift_interval=args.relift_interval,
        relift_steps=args.relift_steps,
        regrasp_on_effect_loss=args.regrasp_on_effect_loss,
        effect_loss_threshold=args.effect_loss_threshold,
        effect_loss_alignment_threshold=args.effect_loss_alignment_threshold,
        effect_loss_gap_threshold=args.effect_loss_gap_threshold,
        effect_loss_patience=args.effect_loss_patience,
        regrasp_approach_steps=args.regrasp_approach_steps,
        regrasp_descend_steps=args.regrasp_descend_steps,
        regrasp_close_steps=args.regrasp_close_steps,
        regrasp_lift_steps=args.regrasp_lift_steps,
    ) for i in range(args.test_roots)]
    payload = {
        "schema": "live_cube_rgb_pick_place.v2",
        "claim_boundary": "RGB-only mechanism arm; no frozen-LeWM, universal, or causal-certificate claim",
        "protocol": _jsonable({key: value for key, value in vars(args).items() if key != "output"}),
        "rgb_probe_diagnostics": probe,
        "approach_chart_diagnostics": approach_report,
        "fit_device": fit_device,
        "local_pixel_approach_diagnostics": local_pixel_reports,
        "local_spatial_approach_diagnostics": local_spatial_reports,
        "initial_conditions_frozen": True,
        "initial_conditions_source": (
            "captured_before_test" if args.initial_conditions is None
            else str(args.initial_conditions)
        ),
        "initial_conditions": [
            {
                "seed": int(initial["seed"]),
                "task_id": int(initial["task_id"]),
                "variation_values": _jsonable(initial["variation_values"]),
                "state": _jsonable(initial["state"]),
                "goal": _jsonable(initial["goal"]),
            }
            for initial in test_initials
        ],
            "calibration_initial_conditions": [
            {
                "seed": int(initial["seed"]),
                "task_id": int(initial["task_id"]),
                "variation_values": _jsonable(initial["variation_values"]),
                "state": _jsonable(initial["state"]),
                "goal": _jsonable(initial["goal"]),
            }
            for initial in (
                calibration_initials
                if args.approach_controller in {"world", "hybrid", "adaptive"}
                else []
            )
        ],
        "rows": rows,
        "aggregate": {
            "roots": len(rows), "success_rate": float(np.mean([r.get("success", False) for r in rows])),
            "contact_rate": float(np.mean([r.get("contact_observed", False) for r in rows])),
            "mean_final_distance_m": float(np.mean([r["final"]["block_goal_distance_m"] for r in rows if r.get("final", {}).get("block_goal_distance_m") is not None])),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
