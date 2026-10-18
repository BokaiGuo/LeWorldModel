"""Online RGB-only Cube control using a calibrated macro ranker.

The ranker is fitted on calibration-only branch points and executes a declared
four-direction macro bank.  The default is the historical 20-step protocol;
longer banks and push-scale interventions are explicit exploratory options.
At runtime it receives only current RGB, previous RGB, the rendered RGB goal,
and the previous action.  Simulator state is used only for evaluator labels.
This runner is deliberately separate from the frozen-LeWM CEM runner: it
tests whether positive offline goal-conditioned macro ranking transfers to a
closed-loop controller, and records when that transfer fails.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

# Configure headless MuJoCo before the environment package is imported.  The
# CUDA runtime is checked by torch after import, while EGL keeps RGB rendering
# available on the GPU host.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers environments
import torch

from experiments.gpu_preflight import require_cuda_allocation

from experiments.audit_cube_macro_action_ranking import (
    _branch_macro_effects,
    _fit,
    _macro_bank,
    _macro_summary,
    _predict,
)
from experiments.audit_cube_rgb_observability import _make_env
from experiments.audit_cube_rgb_observability import (
    _goal_position,
    _largest_effector_centroid,
    _target_red_centroid,
)


def _rgb_approach_action(
    image: np.ndarray,
    pixel_action_mapping: np.ndarray | None = None,
    gain: float = 1.0,
) -> np.ndarray:
    """Move the visible end-effector toward the visible physical cube."""
    if not np.isfinite(float(gain)) or not 0.0 < float(gain) <= 1.0:
        raise ValueError("approach gain must be finite and in (0, 1]")
    block = _target_red_centroid(image)[:2]
    effector = _largest_effector_centroid(image)[:2]
    direction = np.asarray(block - effector, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1.0e-6:
        direction[:] = 0.0
    else:
        direction /= max(norm, 1.0)
    action = np.zeros(5, dtype=np.float32)
    if pixel_action_mapping is None:
        action[:2] = np.clip(float(gain) * direction, -1.0, 1.0)
    else:
        mapping = np.asarray(pixel_action_mapping, dtype=np.float64)
        if mapping.shape != (2, 2) or not np.isfinite(mapping).all():
            raise ValueError("pixel_action_mapping must be a finite 2x2 matrix")
        action[:2] = np.clip(
            np.asarray(direction, dtype=np.float64) @ np.linalg.pinv(mapping),
            -1.0, 1.0,
        ).astype(np.float32) * float(gain)
    return action


def _approach_observation(image: np.ndarray) -> dict[str, Any]:
    """Return RGB-only approach geometry and explicit validity flags."""
    block = _target_red_centroid(image)[:2]
    effector = _largest_effector_centroid(image)[:2]
    block_valid = bool(np.linalg.norm(block) > 0.0)
    effector_valid = bool(np.linalg.norm(effector) > 0.0)
    error = block - effector
    return {
        "block_xy": block.astype(np.float32).tolist(),
        "effector_xy": effector.astype(np.float32).tolist(),
        "approach_error_xy": error.astype(np.float32).tolist(),
        "approach_error_norm": float(np.linalg.norm(error))
        if block_valid and effector_valid else None,
        "block_valid": block_valid,
        "effector_valid": effector_valid,
    }


def _rgb_block_goal_error(
    image: np.ndarray, goal_image: np.ndarray,
) -> tuple[float, bool]:
    """Return physical-block RGB goal error for a feedback-only push gate."""
    block = _target_red_centroid(image)[:2]
    goal = _target_red_centroid(goal_image)[:2]
    valid = bool(np.linalg.norm(block) > 0.0 and np.linalg.norm(goal) > 0.0)
    return (float(np.linalg.norm(goal - block)), valid) if valid else (0.0, False)


def _mode_conditioned_reclose_decision(
    cycle_trace: list[dict[str, Any]],
    min_progress: float,
) -> tuple[bool, str, float | None]:
    """Decide whether the next cycle should reacquire contact from RGB only.

    The previous cycle's RGB goal-error decrease is used as a conservative
    contact-mode proxy.  Missing RGB evidence fails closed to reclose.  This
    gate is deliberately a protocol intervention, not a claim that RGB
    identifies simulator contact state.
    """
    if not cycle_trace:
        return False, "first_cycle", None
    previous = cycle_trace[-1]
    before = previous.get("before")
    after = previous.get("after")
    if before is None or after is None:
        return True, "invalid_rgb", None
    progress = float(before) - float(after)
    if not np.isfinite(progress) or progress < float(min_progress):
        return True, "insufficient_rgb_progress", progress
    return False, "progress_ok", progress


def _online_effect_action(
    action_history: list[np.ndarray], delta_history: list[np.ndarray],
    desired_delta: np.ndarray, candidate_actions: np.ndarray,
    min_mapping_norm: float = 1.0e-3,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Select a push from an RGB-observed within-contact effect map.

    The map is fitted only from executed RGB block displacements.  It is
    deliberately disabled until the observed action support has rank two and
    at least four rows; otherwise the caller must retain the calibrated
    ranker action rather than extrapolate from a one-direction contact trace.
    """
    actions = np.asarray(action_history, dtype=np.float64)
    deltas = np.asarray(delta_history, dtype=np.float64)
    candidates = np.asarray(candidate_actions, dtype=np.float64)
    desired = np.asarray(desired_delta, dtype=np.float64).reshape(2)
    if len(action_history) == 0 and len(delta_history) == 0:
        return None, {"status": "insufficient_support", "rows": 0, "rank": 0}
    if actions.ndim != 2 or actions.shape[1] != 2 or deltas.shape != actions.shape:
        raise ValueError("action and delta histories must both have shape [n, 2]")
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("candidate_actions must have shape [n, 2]")
    if not np.isfinite(actions).all() or not np.isfinite(deltas).all():
        raise ValueError("effect histories must be finite")
    rank = int(np.linalg.matrix_rank(actions)) if len(actions) else 0
    diagnostics = {"status": "insufficient_support", "rows": int(len(actions)), "rank": rank}
    if len(actions) < 4 or rank < 2:
        return None, diagnostics
    mapping, _, _, _ = np.linalg.lstsq(actions, deltas, rcond=1.0e-6)
    mapping_norm = float(np.linalg.norm(mapping, ord="fro"))
    if not np.isfinite(mapping_norm) or mapping_norm < float(min_mapping_norm):
        diagnostics.update({
            "status": "weak_effect",
            "mapping_norm": mapping_norm,
            "min_mapping_norm": float(min_mapping_norm),
            "mapping": mapping.astype(np.float64).tolist(),
        })
        return None, diagnostics
    predicted = candidates @ mapping
    scores = predicted @ desired
    selected = int(np.argmax(scores))
    diagnostics.update({
        "status": "ok",
        "selected_index": selected,
        "predicted_progress": scores.astype(np.float64).tolist(),
        "mapping": mapping.astype(np.float64).tolist(),
        "mapping_norm": mapping_norm,
        "min_mapping_norm": float(min_mapping_norm),
    })
    return candidates[selected].astype(np.float32), diagnostics


def _effect_mapping_norm(
    action_history: list[np.ndarray], delta_history: list[np.ndarray],
) -> float:
    """Return fitted effect strength; rank alone is not an excitation test."""
    actions = np.asarray(action_history, dtype=np.float64)
    deltas = np.asarray(delta_history, dtype=np.float64)
    if actions.ndim != 2 or deltas.shape != actions.shape or len(actions) < 4:
        return 0.0
    if not np.isfinite(actions).all() or not np.isfinite(deltas).all():
        return 0.0
    mapping, _, _, _ = np.linalg.lstsq(actions, deltas, rcond=1.0e-6)
    value = float(np.linalg.norm(mapping, ord="fro"))
    return value if np.isfinite(value) else 0.0


def _effect_probe_required(
    action_history: list[np.ndarray], delta_history: list[np.ndarray],
    min_mapping_norm: float = 1.0e-3,
) -> tuple[bool, int]:
    """Require orthogonal excitation until the 2-D effect map is identifiable."""
    actions = np.asarray(action_history, dtype=np.float64)
    if len(action_history) < 4 or actions.ndim != 2 or actions.shape[1] != 2:
        return False, int(np.linalg.matrix_rank(actions)) if actions.ndim == 2 else 0
    rank = int(np.linalg.matrix_rank(actions))
    weak = _effect_mapping_norm(action_history, delta_history) < float(min_mapping_norm)
    return bool(rank < 2 or weak), rank


def _support_probe_candidate_index(
    bank: np.ndarray, action_history: list[np.ndarray], force: bool = False,
) -> int | None:
    """Return an orthogonal nonzero candidate when observed push support is rank one."""
    actions = np.asarray(action_history, dtype=np.float64)
    if actions.size == 0 or actions.ndim != 2 or actions.shape[1] != 2:
        return None
    if int(np.linalg.matrix_rank(actions)) >= 2 and not force:
        return None
    anchor = actions[-1]
    norm = float(np.linalg.norm(anchor))
    if norm <= 1.0e-8:
        return None
    anchor /= norm
    bank_array = np.asarray(bank)
    close_indices = np.flatnonzero(bank_array[1, :, 4] > 0.5)
    push_start = int(close_indices[-1]) + 1 if len(close_indices) else 0
    directions = np.asarray(
        [sequence[push_start:, :2].mean(axis=0) for sequence in bank_array[1:]],
        dtype=np.float64,
    )
    valid = np.linalg.norm(directions, axis=1) > 1.0e-8
    if not np.any(valid):
        return None
    orthogonality = np.full(len(directions), np.inf, dtype=np.float64)
    orthogonality[valid] = np.abs(directions[valid] @ anchor)
    return int(np.argmin(orthogonality)) + 1


def _fit_rgb_pixel_action_mapping(
    *, seed: int, tasks: list[int], roots: int, descent_steps: int,
    probe_steps: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Fit an RGB-only local action-to-effector image Jacobian.

    The fit uses only executed XY probes and successive RGB effector
    centroids.  Simulator state is not read.  It is a calibration of the
    camera/action chart, not a task-outcome readout, and is therefore safe to
    use before the online ranker is queried.
    """
    if roots < 1 or descent_steps < 0 or probe_steps < 1:
        raise ValueError("roots and probe_steps must be positive; descent_steps non-negative")
    directions = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    actions: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    env = _make_env()
    try:
        for root in range(int(roots)):
            task_id = int(tasks[root % len(tasks)])
            for direction in directions:
                obs, _info = env.reset(
                    seed=int(seed + root),
                    options={"task_id": task_id, "render_goal": True},
                )
                for _ in range(int(descent_steps)):
                    descent = np.zeros(5, dtype=np.float32)
                    descent[2] = -1.0
                    obs, _reward, terminated, truncated, _info = env.step(descent)
                    if terminated or truncated:
                        break
                before = _largest_effector_centroid(np.asarray(obs, dtype=np.uint8))[:2]
                if not np.linalg.norm(before) > 0.0:
                    continue
                probe = np.zeros(5, dtype=np.float32)
                probe[:2] = direction
                for _ in range(int(probe_steps)):
                    obs, _reward, terminated, truncated, _info = env.step(probe)
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
        return None, {
            "status": "insufficient_rgb_probe_rows",
            "probe_rows": len(actions),
            "rank": 0,
            "condition_number": None,
        }
    action_matrix = np.asarray(actions, dtype=np.float64)
    delta_matrix = np.asarray(deltas, dtype=np.float64)
    mapping, _, rank, singular_values = np.linalg.lstsq(
        action_matrix, delta_matrix, rcond=1.0e-6,
    )
    residual = action_matrix @ mapping - delta_matrix
    condition = (
        float(singular_values[0] / singular_values[-1])
        if len(singular_values) > 1 and singular_values[-1] > 1.0e-8 else None
    )
    return mapping.astype(np.float32), {
        "status": "ok" if int(rank) == 2 else "rank_deficient",
        "probe_rows": len(actions),
        "rank": int(rank),
        "condition_number": condition,
        "fit_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mapping": mapping.astype(np.float32).tolist(),
    }


def _physical_features(
    branches: list[dict[str, Any]], bank: np.ndarray, device: str,
    contact_aware: bool = False,
) -> torch.Tensor:
    """Build the online ranker features with one physical RGB chart.

    The legacy macro audit uses the largest red component.  That is useful for
    its frozen offline protocol, but an online Cube frame can contain both the
    physical block and its red goal overlay.  This runner therefore uses the
    same ``_target_red_centroid`` detector for current, previous, and goal
    frames, so calibration and deployment cannot silently use different
    object identities.
    """
    n = len(branches)
    current = torch.as_tensor(
        np.repeat(
            np.asarray([_target_red_centroid(b["current_image"]) for b in branches], dtype=np.float32),
            len(bank), axis=0,
        ), device=device,
    )
    previous = torch.as_tensor(
        np.repeat(
            np.asarray([_target_red_centroid(b["previous_image"]) for b in branches], dtype=np.float32),
            len(bank), axis=0,
        ), device=device,
    )
    goal = torch.as_tensor(
        np.repeat(
            np.asarray([_target_red_centroid(b["goal_image"]) for b in branches], dtype=np.float32),
            len(bank), axis=0,
        ), device=device,
    )
    effector = torch.as_tensor(
        np.repeat(
            np.asarray([
                _largest_effector_centroid(b["current_image"])
                for b in branches
            ], dtype=np.float32),
            len(bank), axis=0,
        ), device=device,
    )
    previous_actions = torch.as_tensor(
        np.repeat(
            np.asarray([b["previous_action"] for b in branches], dtype=np.float32),
            len(bank), axis=0,
        ), device=device,
    )
    actions = torch.as_tensor(
        np.tile(bank.mean(axis=1), (n, 1)), device=device, dtype=torch.float32,
    )
    base = torch.cat([
        current, goal, current - goal, current - previous, actions, previous_actions,
    ], dim=1)
    summary = torch.as_tensor(
        np.tile(_macro_summary(bank), (n, 1)), device=device, dtype=torch.float32,
    )
    directions = torch.as_tensor(
        np.asarray([sequence[:, :2].mean(axis=0) for sequence in bank], dtype=np.float32),
        device=device,
    )
    screen_delta = (goal[:, :2] - current[:, :2]).reshape(n, len(bank), 2)
    outer = (
        screen_delta[:, :, :, None] * directions[None, :, None, :]
    ).reshape(n * len(bank), -1)
    dot = (screen_delta * directions[None, :, :]).sum(
        dim=-1, keepdim=True,
    ).reshape(n * len(bank), 1)
    cross = (
        screen_delta[:, :, 0] * directions[None, :, 1]
        - screen_delta[:, :, 1] * directions[None, :, 0]
    ).reshape(n * len(bank), 1)
    interactions = torch.cat([outer, dot, cross], dim=1)
    if contact_aware:
        contact_chart = torch.cat([effector, effector - current], dim=1)
        return torch.cat([base, contact_chart, summary, interactions], dim=1)
    return torch.cat([base, summary, interactions], dim=1)


def _online_macro_bank(length: int = 20, push_scale: float = 1.0) -> np.ndarray:
    """Return the deployable bank with the OGB-Cube close sign.

    The historical offline macro audit encoded the close segment as ``-1``.
    OGB-Cube's current action interface uses ``+1`` to close the gripper
    (the same convention as the validated visual-servo runner).  Keep the
    historical bank untouched for reproducibility and correct the sign only
    in this online transfer intervention.
    """
    if not np.isfinite(float(push_scale)) or not 0.0 < float(push_scale) <= 1.0:
        raise ValueError("push_scale must be finite and in (0, 1]")
    bank = _macro_bank(length=int(length), direction_count=4, approach_steps=4).copy()
    # The historical bank marks its close segment with -1.  Locate that
    # declared segment instead of hard-coding indices so the longer diagnostic
    # bank preserves the same phase structure.
    close_segment = bank[1, :, 4] < -0.5
    bank[1:, close_segment, 4] = 1.0
    close_end = int(np.flatnonzero(close_segment)[-1]) + 1
    bank[1:, close_end:, :2] *= float(push_scale)
    return bank


def _factorized_online_macro_bank(length: int = 20, push_scale: float = 1.0) -> np.ndarray:
    """Deployable bank with independent approach and push directions.

    The historical bank uses one direction for both phases.  That creates a
    support failure when the effector must approach from one side and push
    the object toward the goal from another.  This bank is finite and
    predeclared; it is a protocol intervention, not privileged state access.
    """
    if length < 20:
        raise ValueError("macro length must be at least 20")
    if not np.isfinite(float(push_scale)) or not 0.0 < float(push_scale) <= 1.0:
        raise ValueError("push_scale must be finite and in (0, 1]")
    directions = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    approach_end = 4
    descend_end = min(length - 8, approach_end + max(6, length // 4))
    close_end = min(length - 2, descend_end + max(4, length // 12))
    candidates = [np.zeros((length, 5), dtype=np.float32)]
    for approach in directions:
        for push in directions:
            for gain in (1.0, 0.5, 0.25):
                sequence = np.zeros((length, 5), dtype=np.float32)
                sequence[:approach_end, :2] = approach
                sequence[approach_end:descend_end, 2] = -1.0
                sequence[descend_end:close_end, 4] = 1.0
                sequence[close_end:, :2] = float(gain * push_scale) * push
                candidates.append(sequence)
    return np.stack(candidates, axis=0)


def _multiphase_factorized_online_macro_bank(
    length: int = 20, push_scale: float = 1.0,
) -> np.ndarray:
    """Deployable bank with two persistent push directions after close."""
    if length < 20:
        raise ValueError("macro length must be at least 20")
    if not np.isfinite(float(push_scale)) or not 0.0 < float(push_scale) <= 1.0:
        raise ValueError("push_scale must be finite and in (0, 1]")
    directions = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    approach_end = 4
    descend_end = min(length - 8, approach_end + max(6, length // 4))
    close_end = min(length - 4, descend_end + max(4, length // 12))
    split = close_end + max(1, (length - close_end) // 2)
    candidates = [np.zeros((length, 5), dtype=np.float32)]
    for approach in directions:
        for push_a in directions:
            for push_b in directions:
                for gain in (0.5, 0.25):
                    sequence = np.zeros((length, 5), dtype=np.float32)
                    sequence[:approach_end, :2] = approach
                    sequence[approach_end:descend_end, 2] = -1.0
                    sequence[descend_end:close_end, 4] = 1.0
                    sequence[close_end:split, :2] = float(gain * push_scale) * push_a
                    sequence[split:, :2] = float(gain * push_scale) * push_b
                    candidates.append(sequence)
    return np.stack(candidates, axis=0)


def _capture_rgb_approach_branch(
    *, seed: int, task_id: int, approach_steps: int, descent_steps: int,
    pixel_action_mapping: np.ndarray | None, approach_gain: float,
    approach_threshold: float,
) -> dict[str, Any]:
    """Collect a calibration branch using only the frozen RGB approach law."""
    env = _make_env()
    try:
        obs, info = env.reset(
            seed=int(seed), options={"task_id": int(task_id), "render_goal": True},
        )
        goal_image = np.asarray(env.unwrapped._cur_goal_rendered, dtype=np.uint8).copy()
        previous = np.asarray(obs, dtype=np.uint8).copy()
        previous_action = np.zeros(5, dtype=np.float32)
        approach_trace: list[dict[str, Any]] = []
        contact_trace: list[float] = []
        block_goal_distance_trace: list[float] = []
        terminated = False
        truncated = False
        for _ in range(int(descent_steps)):
            before = np.asarray(obs, dtype=np.uint8).copy()
            descent = np.zeros(5, dtype=np.float32)
            descent[2] = -1.0
            obs, _reward, terminated, truncated, _info = env.step(descent)
            previous = before
            previous_action = descent.copy()
            if terminated or truncated:
                raise RuntimeError("RGB descent terminated before calibration branch")
        for _ in range(int(approach_steps)):
            before = np.asarray(obs, dtype=np.uint8).copy()
            approach_trace.append(_approach_observation(before))
            if approach_threshold > 0.0 and approach_trace[-1]["approach_error_norm"] is not None \
                    and approach_trace[-1]["approach_error_norm"] <= approach_threshold:
                break
            action = _rgb_approach_action(before, pixel_action_mapping, approach_gain)
            obs, _reward, terminated, truncated, info = env.step(action)
            previous = before
            previous_action = action.copy()
            contact_trace.append(float(np.max(np.asarray(
                info.get("proprio/gripper_contact", [0.0]), dtype=np.float32,
            ))))
            block = np.asarray(info.get("privileged/block_0_pos", [np.nan, np.nan]), dtype=np.float64).reshape(-1)
            if block.size >= 2:
                block_goal_distance_trace.append(float(np.linalg.norm(
                    block[:2] - np.asarray(_goal_position(env)[:2], dtype=np.float64),
                )))
            if terminated or truncated:
                raise RuntimeError("RGB approach terminated before calibration branch")
        raw = env.unwrapped
        return {
            "seed": int(seed),
            "task_id": int(task_id),
            "variation_values": copy.deepcopy(raw.variation_space.value),
            "state": np.concatenate([raw.data.qpos.copy(), raw.data.qvel.copy()]).tolist(),
            "current_image": np.asarray(obs, dtype=np.uint8).copy(),
            "previous_image": previous,
            "goal_image": goal_image,
            "previous_action": previous_action,
            "goal": _goal_position(env)[:2].copy(),
            "approach_trace": approach_trace,
            "contact_trace": contact_trace,
            "block_goal_distance_trace": block_goal_distance_trace,
        }
    finally:
        env.close()


def _branch_rgb_macro_effects(
    branch: dict[str, Any], bank: np.ndarray,
) -> np.ndarray:
    """Return macro effects from realized RGB successors only.

    This calibration target deliberately avoids simulator block position and
    task-cost labels.  It uses the same rendered goal image and the final RGB
    observation after each macro, so it can test whether the online RGB
    feature family transfers without a privileged teacher.
    """
    env = _make_env()
    try:
        goal = _target_red_centroid(
            np.asarray(branch["goal_image"], dtype=np.uint8),
        )[:2].astype(np.float32)
        effects: list[np.ndarray] = []
        for sequence in bank:
            obs, _info = env.reset(
                seed=int(branch["seed"]),
                options={
                    "task_id": int(branch["task_id"]),
                    "render_goal": True,
                    "variation_values": copy.deepcopy(branch["variation_values"]),
                    "state": np.asarray(branch["state"], dtype=np.float64),
                },
            )
            terminated = False
            truncated = False
            for action in sequence:
                obs, _reward, terminated, truncated, _step_info = env.step(action)
                if terminated or truncated:
                    break
            block = _target_red_centroid(
                np.asarray(obs, dtype=np.uint8),
            )[:2].astype(np.float32)
            if np.linalg.norm(block) <= 0.0 or np.linalg.norm(goal) <= 0.0:
                effects.append(np.zeros(2, dtype=np.float32))
            else:
                effects.append(goal - block)
        return np.asarray(effects, dtype=np.float32)
    finally:
        env.close()


def _episode(
    *,
    seed: int,
    task_id: int,
    bank: np.ndarray,
    fit: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: str,
    action_block: int,
    macro_cycles: int,
    approach_steps: int,
    descent_steps: int,
    pixel_action_mapping: np.ndarray | None,
    push_feedback: bool,
    push_feedback_threshold: float,
    effect_adaptation: bool,
    reclose_each_cycle: bool,
    stop_on_rgb_regression: bool,
    push_backtrack: bool,
    stop_push_on_regression: bool,
    post_approach_descend_steps: int,
    post_approach_close_steps: int,
    approach_gain: float,
    approach_threshold: float,
    contact_probe: bool,
    contact_probe_steps: int,
    contact_aware_features: bool,
    mode_conditioned_reclose: bool,
    min_cycle_rgb_progress: float,
) -> dict[str, Any]:
    env = _make_env()
    try:
        obs, info = env.reset(
            seed=int(seed), options={"task_id": int(task_id), "render_goal": True},
        )
        goal = np.asarray(env.unwrapped._cur_goal_rendered, dtype=np.uint8).copy()
        previous = np.asarray(obs, dtype=np.uint8).copy()
        previous_action = np.zeros(5, dtype=np.float32)
        approach_trace: list[dict[str, Any]] = []
        contact_trace: list[float] = []
        post_contact_actions = 0
        effect_actions: list[np.ndarray] = []
        effect_deltas: list[np.ndarray] = []
        contact_probe_trace: list[dict[str, Any]] = []
        block_goal_distance_trace: list[float] = []
        terminated = False
        truncated = False
        for _ in range(int(descent_steps)):
            before = np.asarray(obs, dtype=np.uint8).copy()
            descent = np.zeros(5, dtype=np.float32)
            descent[2] = -1.0
            obs, _reward, terminated, truncated, info = env.step(descent)
            previous = before
            previous_action = descent.copy()
            contact_trace.append(float(np.max(np.asarray(
                info.get("proprio/gripper_contact", [0.0]), dtype=np.float32,
            ))))
            block = np.asarray(info.get("privileged/block_0_pos", [np.nan, np.nan]), dtype=np.float64).reshape(-1)
            if block.size >= 2:
                block_goal_distance_trace.append(float(np.linalg.norm(
                    block[:2] - np.asarray(_goal_position(env)[:2], dtype=np.float64),
                )))
            if terminated or truncated:
                break
        if terminated or truncated:
            success = np.asarray(info.get("success", 0), dtype=np.float64).reshape(-1)
            return {"seed": int(seed), "task_id": int(task_id), "selected_macro_indices": [],
                    "success": bool(success[-1] > 0.5) if len(success) else False,
                    "final_success_info": success.tolist(), "steps": int(descent_steps),
                    "approach_trace": approach_trace,
                    "contact_trace": contact_trace,
                    "block_goal_distance_trace": block_goal_distance_trace,
                    "contact_observed": bool(any(value > 0.5 for value in contact_trace)),
                    "zero_macro_selected": False}
        # Optional mechanism arm: test the alternate phase ordering
        # align(XY) -> descend -> close.  The policy sees only the RGB
        # approach window; contact is recorded solely as an evaluator label.
        for _ in range(int(post_approach_descend_steps)):
            action = np.zeros(5, dtype=np.float32)
            action[2] = -1.0
            before = np.asarray(obs, dtype=np.uint8).copy()
            obs, _reward, terminated, truncated, info = env.step(action)
            previous = before
            previous_action = action.copy()
            post_contact_actions += 1
            contact_trace.append(float(np.max(np.asarray(
                info.get("proprio/gripper_contact", [0.0]), dtype=np.float32,
            ))))
            if terminated or truncated:
                break
        if not (terminated or truncated):
            for _ in range(int(post_approach_close_steps)):
                action = np.zeros(5, dtype=np.float32)
                action[4] = 1.0
                before = np.asarray(obs, dtype=np.uint8).copy()
                obs, _reward, terminated, truncated, info = env.step(action)
                previous = before
                previous_action = action.copy()
                post_contact_actions += 1
                contact_trace.append(float(np.max(np.asarray(
                    info.get("proprio/gripper_contact", [0.0]), dtype=np.float32,
                ))))
                if terminated or truncated:
                    break
        if terminated or truncated:
            success = np.asarray(info.get("success", 0), dtype=np.float64).reshape(-1)
            return {"seed": int(seed), "task_id": int(task_id), "selected_macro_indices": [],
                    "success": bool(success[-1] > 0.5) if len(success) else False,
                    "final_success_info": success.tolist(), "steps": int(approach_steps + post_contact_actions),
                    "approach_trace": approach_trace,
                    "contact_trace": contact_trace,
                    "block_goal_distance_trace": block_goal_distance_trace,
                    "contact_observed": bool(any(value > 0.5 for value in contact_trace)),
                    "post_contact_actions": int(post_contact_actions),
                    "zero_macro_selected": False}
        for _ in range(int(approach_steps)):
            before = np.asarray(obs, dtype=np.uint8).copy()
            approach_trace.append(_approach_observation(before))
            if approach_threshold > 0.0 and approach_trace[-1]["approach_error_norm"] is not None \
                    and approach_trace[-1]["approach_error_norm"] <= approach_threshold:
                break
            action = _rgb_approach_action(before, pixel_action_mapping, approach_gain)
            obs, _reward, terminated, truncated, info = env.step(action)
            previous = before
            previous_action = action.copy()
            contact_trace.append(float(np.max(np.asarray(
                info.get("proprio/gripper_contact", [0.0]), dtype=np.float32,
            ))))
            block = np.asarray(info.get("privileged/block_0_pos", [np.nan, np.nan]), dtype=np.float64).reshape(-1)
            if block.size >= 2:
                block_goal_distance_trace.append(float(np.linalg.norm(
                    block[:2] - np.asarray(_goal_position(env)[:2], dtype=np.float64),
                )))
            if terminated or truncated:
                break
        if terminated or truncated:
            success = np.asarray(info.get("success", 0), dtype=np.float64).reshape(-1)
            return {"seed": int(seed), "task_id": int(task_id), "selected_macro_indices": [],
                    "success": bool(success[-1] > 0.5) if len(success) else False,
                    "final_success_info": success.tolist(), "steps": int(approach_steps),
                    "approach_trace": approach_trace,
                    "contact_trace": contact_trace,
                    "block_goal_distance_trace": block_goal_distance_trace,
                    "zero_macro_selected": False}
        if contact_probe:
            # RGB-only contact validation: execute one short push pulse in
            # each declared direction and retain only observed target-centroid
            # displacement.  This is an excitation phase, not a success
            # label; no simulator contact or object state is read here.
            close_indices = np.flatnonzero(bank[1, :, 4] > 0.5)
            push_start = int(close_indices[-1]) + 1 if len(close_indices) else 0
            for probe_index in range(1, min(5, len(bank))):
                action = np.asarray(bank[probe_index, push_start], dtype=np.float32).copy()
                for _ in range(int(contact_probe_steps)):
                    before = np.asarray(obs, dtype=np.uint8).copy()
                    before_block = _target_red_centroid(before)[:2]
                    obs, _reward, terminated, truncated, info = env.step(action)
                    after = np.asarray(obs, dtype=np.uint8).copy()
                    after_block = _target_red_centroid(after)[:2]
                    delta = after_block - before_block
                    valid = bool(np.linalg.norm(before_block) > 0.0 and np.linalg.norm(after_block) > 0.0)
                    if valid:
                        effect_actions.append(action[:2].astype(np.float64))
                        effect_deltas.append(delta.astype(np.float64))
                    contact_probe_trace.append({
                        "candidate_index": int(probe_index),
                        "valid_rgb_target": valid,
                        "rgb_delta": delta.astype(np.float64).tolist(),
                    })
                    previous = before
                    previous_action = action.copy()
                    contact_trace.append(float(np.max(np.asarray(
                        info.get("proprio/gripper_contact", [0.0]), dtype=np.float32,
                    ))))
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    break
        selected: list[int] = []
        score_trace: list[list[float]] = []
        push_feedback_reversals = 0
        push_backtracks = 0
        push_gain_trace: list[float] = []
        stopped_push_for_regression = False
        push_error_trace: list[float] = []
        effect_trace: list[dict[str, Any]] = []
        support_probe_forced = 0
        effect_probe_retries = 0
        stopped_for_rgb_regression = False
        cycle_rgb_error_trace: list[dict[str, Any]] = []
        mode_reclose_trace: list[dict[str, Any]] = []
        terminated = False
        truncated = False
        for _cycle in range(int(macro_cycles)):
            branch = {
                "current_image": np.asarray(obs, dtype=np.uint8).copy(),
                "previous_image": previous.copy(),
                "goal_image": goal.copy(),
                "previous_action": previous_action.copy(),
            }
            scores = np.linalg.norm(
                _predict(
                    _physical_features(
                        [branch], bank, device, contact_aware_features,
                    ),
                    fit,
                ),
                axis=1,
            )
            score_trace.append(scores.astype(np.float64).tolist())
            index = int(np.argmin(scores))
            if effect_adaptation:
                # A non-weak rank-one map is still insufficient for a 2-D
                # action-effect decision.  The previous gate only tested map
                # norm, so it could repeatedly execute one direction and
                # report ``insufficient_support`` forever.  Force an
                # orthogonal probe for either failure mode.
                probe_required, effect_rank = _effect_probe_required(
                    effect_actions, effect_deltas,
                )
                probe_index = _support_probe_candidate_index(
                    bank, effect_actions,
                    force=probe_required and effect_probe_retries < 2,
                )
                if probe_index is not None:
                    index = probe_index
                    support_probe_forced += 1
                    if probe_required:
                        effect_probe_retries += 1
            selected.append(index)
            cycle_error_before, cycle_error_before_valid = _rgb_block_goal_error(
                np.asarray(obs, dtype=np.uint8), goal,
            )
            push_sign = 1.0
            push_gain = 1.0
            previous_push_error: float | None = None
            close_indices = np.flatnonzero(bank[index, :, 4] > 0.5)
            push_start = int(close_indices[-1]) + 1 if len(close_indices) else 0
            cycle_reclose = bool(reclose_each_cycle)
            mode_reason = "fixed_reclose" if reclose_each_cycle else "persistent"
            mode_progress: float | None = None
            if mode_conditioned_reclose and _cycle > 0:
                cycle_reclose, mode_reason, mode_progress = _mode_conditioned_reclose_decision(
                    cycle_rgb_error_trace, min_cycle_rgb_progress,
                )
            action_start = 0 if _cycle == 0 or cycle_reclose else push_start
            mode_reclose_trace.append({
                "cycle": int(_cycle),
                "reclose": bool(_cycle == 0 or cycle_reclose),
                "reason": mode_reason,
                "rgb_progress": mode_progress,
            })
            for action_index, action in enumerate(bank[index]):
                if action_index < action_start:
                    continue
                for _ in range(int(action_block)):
                    before = np.asarray(obs, dtype=np.uint8).copy()
                    runtime_action = np.asarray(action, dtype=np.float32).copy()
                    if push_feedback and action_index >= push_start and np.linalg.norm(runtime_action[:2]) > 1.0e-8:
                        current_error, current_valid = _rgb_block_goal_error(before, goal)
                        if current_valid:
                            if (
                                previous_push_error is not None
                                and current_error > previous_push_error + float(push_feedback_threshold)
                            ):
                                push_sign *= -1.0
                                push_feedback_reversals += 1
                                if push_backtrack:
                                    push_gain = max(0.25, 0.5 * push_gain)
                                    push_backtracks += 1
                                if stop_push_on_regression:
                                    stopped_push_for_regression = True
                                    break
                            runtime_action[:2] *= float(push_sign * push_gain)
                            push_gain_trace.append(float(push_gain))
                            push_error_trace.append(current_error)
                    if effect_adaptation and action_index >= push_start:
                        block_before = _target_red_centroid(before)[:2]
                        goal_xy = _target_red_centroid(goal)[:2]
                        candidate_actions = np.asarray(
                            [candidate[action_index, :2] for candidate in bank[1:]],
                            dtype=np.float32,
                        )
                        desired = goal_xy - block_before
                        adapted, effect_diag = _online_effect_action(
                            effect_actions, effect_deltas, desired, candidate_actions,
                        )
                        effect_trace.append(effect_diag)
                        if adapted is not None:
                            runtime_action[:2] = adapted
                    next_obs, _reward, terminated, truncated, info = env.step(runtime_action)
                    previous = before
                    previous_action = runtime_action.copy()
                    obs = next_obs
                    block_before = _target_red_centroid(before)[:2]
                    block_after = _target_red_centroid(np.asarray(obs, dtype=np.uint8))[:2]
                    if (
                        action_index >= push_start
                        and np.linalg.norm(block_before) > 0.0
                        and np.linalg.norm(block_after) > 0.0
                    ):
                        effect_actions.append(runtime_action[:2].astype(np.float64))
                        effect_deltas.append((block_after - block_before).astype(np.float64))
                    contact_trace.append(float(np.max(np.asarray(
                        info.get("proprio/gripper_contact", [0.0]), dtype=np.float32,
                    ))))
                    block = np.asarray(info.get("privileged/block_0_pos", [np.nan, np.nan]), dtype=np.float64).reshape(-1)
                    if block.size >= 2:
                        block_goal_distance_trace.append(float(np.linalg.norm(
                            block[:2] - np.asarray(_goal_position(env)[:2], dtype=np.float64),
                        )))
                    if push_feedback and action_index >= push_start:
                        next_error, next_valid = _rgb_block_goal_error(
                            np.asarray(obs, dtype=np.uint8), goal,
                        )
                        if next_valid:
                            previous_push_error = next_error
                    if terminated or truncated:
                        break
                if stopped_push_for_regression:
                    break
                if terminated or truncated:
                    break
            if stopped_push_for_regression:
                break
            if terminated or truncated:
                break
            cycle_error_after, cycle_error_after_valid = _rgb_block_goal_error(
                np.asarray(obs, dtype=np.uint8), goal,
            )
            cycle_rgb_error_trace.append({
                "before": cycle_error_before if cycle_error_before_valid else None,
                "after": cycle_error_after if cycle_error_after_valid else None,
                "valid": bool(cycle_error_before_valid and cycle_error_after_valid),
            })
            if (
                stop_on_rgb_regression
                and cycle_error_before_valid
                and cycle_error_after_valid
                and cycle_error_after > cycle_error_before + float(push_feedback_threshold)
            ):
                stopped_for_rgb_regression = True
                break
        success = np.asarray(info.get("success", 0), dtype=np.float64).reshape(-1)
        time_value = np.asarray(info.get("time", 0)).reshape(-1)
        return {
            "seed": int(seed),
            "task_id": int(task_id),
            "selected_macro_indices": selected,
            "success": bool(success[-1] > 0.5) if len(success) else False,
            "final_success_info": success.tolist(),
            "steps": int(time_value[-1]) if len(time_value) else 0,
            "approach_trace": approach_trace,
            "score_trace": score_trace,
            "contact_trace": contact_trace,
            "block_goal_distance_trace": block_goal_distance_trace,
            "push_feedback_reversals": int(push_feedback_reversals),
            "push_backtrack": bool(push_backtrack),
            "push_backtracks": int(push_backtracks),
            "push_gain_trace": push_gain_trace,
            "stop_push_on_regression": bool(stop_push_on_regression),
            "stopped_push_for_regression": bool(stopped_push_for_regression),
            "push_error_trace": push_error_trace,
            "effect_adaptation": bool(effect_adaptation),
            "effect_trace": effect_trace,
            "effect_history_rows": len(effect_actions),
            "contact_probe": bool(contact_probe),
            "contact_probe_rows": len(contact_probe_trace),
            "contact_probe_trace": contact_probe_trace,
            "support_probe_forced": int(support_probe_forced),
            "effect_probe_retries": int(effect_probe_retries),
            "stop_on_rgb_regression": bool(stop_on_rgb_regression),
            "stopped_for_rgb_regression": bool(stopped_for_rgb_regression),
            "cycle_rgb_error_trace": cycle_rgb_error_trace,
            "mode_conditioned_reclose": bool(mode_conditioned_reclose),
            "min_cycle_rgb_progress": float(min_cycle_rgb_progress),
            "mode_reclose_trace": mode_reclose_trace,
            "contact_observed": bool(any(value > 0.5 for value in contact_trace)),
            "post_contact_actions": int(post_contact_actions),
            "zero_macro_selected": bool(any(index == 0 for index in selected)),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261900)
    parser.add_argument("--calibration-roots", type=int, default=40)
    parser.add_argument("--test-roots", type=int, default=8)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--macro-cycles", type=int, default=1)
    parser.add_argument("--approach-steps", type=int, default=20)
    parser.add_argument("--approach-gain", type=float, default=1.0)
    parser.add_argument(
        "--approach-threshold", type=float, default=0.0,
        help="stop RGB approach once normalized effector-block error is below this value; 0 disables",
    )
    parser.add_argument("--descent-steps", type=int, default=15)
    parser.add_argument("--probe-steps", type=int, default=4)
    parser.add_argument(
        "--calibration-target",
        choices=("privileged_task", "rgb_goal"),
        default="privileged_task",
        help="calibration target; rgb_goal uses only realized RGB successors",
    )
    parser.add_argument("--macro-length", type=int, default=20)
    parser.add_argument("--push-scale", type=float, default=1.0)
    parser.add_argument(
        "--factorized-bank", action="store_true",
        help="use an explicit independent approach/push candidate bank",
    )
    parser.add_argument(
        "--multiphase-bank", action="store_true",
        help="use independent approach plus two persistent push phases",
    )
    parser.add_argument("--disable-push-feedback", action="store_true")
    parser.add_argument("--push-feedback-threshold", type=float, default=1.0e-3)
    parser.add_argument(
        "--push-backtrack", action="store_true",
        help="halve RGB push gain after a progress regression while reversing sign",
    )
    parser.add_argument(
        "--stop-push-on-regression", action="store_true",
        help="terminate the current push pulse immediately after an RGB regression",
    )
    parser.add_argument("--effect-adaptation", action="store_true")
    parser.add_argument(
        "--reclose-each-cycle", action="store_true",
        help="re-run descend/close on every cycle instead of holding contact",
    )
    parser.add_argument(
        "--mode-conditioned-reclose", action="store_true",
        help="reclose after insufficient RGB goal-error progress; RGB-only gate",
    )
    parser.add_argument(
        "--min-cycle-rgb-progress", type=float, default=1.0e-3,
        help="minimum RGB goal-error decrease required to retain persistent contact",
    )
    parser.add_argument(
        "--allow-rgb-regression", action="store_true",
        help="disable the default fail-closed cycle progress gate",
    )
    parser.add_argument(
        "--post-approach-descend-steps", type=int, default=0,
        help="mechanism arm: descend after the RGB XY approach window",
    )
    parser.add_argument(
        "--post-approach-close-steps", type=int, default=0,
        help="mechanism arm: close after post-approach descent",
    )
    parser.add_argument(
        "--contact-probe", action="store_true",
        help="RGB-only orthogonal push pulses after close to identify target effect support",
    )
    parser.add_argument("--contact-probe-steps", type=int, default=1)
    parser.add_argument(
        "--contact-aware-features", action="store_true",
        help="append RGB effector and effector-block contact chart to ranker features",
    )
    args = parser.parse_args()
    cuda_runtime = None
    if args.device.startswith("cuda"):
        cuda_runtime = require_cuda_allocation(args.device)
    if args.calibration_roots < 8 or args.test_roots < 1:
        raise ValueError("need at least 8 calibration roots and one test root")
    if args.approach_steps < 1 or args.action_block < 1 or args.macro_cycles < 1:
        raise ValueError("approach-steps, action-block, and macro-cycles must be positive")
    if args.descent_steps < 0 or args.probe_steps < 1:
        raise ValueError("descent-steps must be non-negative and probe-steps positive")
    if args.post_approach_descend_steps < 0 or args.post_approach_close_steps < 0:
        raise ValueError("post-approach phase lengths must be non-negative")
    if args.min_cycle_rgb_progress < 0.0 or not np.isfinite(args.min_cycle_rgb_progress):
        raise ValueError("min-cycle-rgb-progress must be finite and non-negative")
    if not 0.0 < args.approach_gain <= 1.0 or not np.isfinite(args.approach_gain):
        raise ValueError("approach-gain must be finite and in (0, 1]")
    if args.approach_threshold < 0.0 or not np.isfinite(args.approach_threshold):
        raise ValueError("approach-threshold must be finite and non-negative")
    if args.contact_probe_steps < 1:
        raise ValueError("contact-probe-steps must be positive")
    if args.push_feedback_threshold < 0.0 or not np.isfinite(args.push_feedback_threshold):
        raise ValueError("push-feedback-threshold must be finite and non-negative")
    if args.macro_length < 20:
        raise ValueError("macro-length must be at least 20")
    if args.factorized_bank and args.multiphase_bank:
        raise ValueError("factorized-bank and multiphase-bank are mutually exclusive")
    if not 0.0 < args.push_scale <= 1.0 or not np.isfinite(args.push_scale):
        raise ValueError("push-scale must be finite and in (0, 1]")
    if args.multiphase_bank:
        bank = _multiphase_factorized_online_macro_bank(args.macro_length, args.push_scale)
    elif args.factorized_bank:
        bank = _factorized_online_macro_bank(args.macro_length, args.push_scale)
    else:
        bank = _online_macro_bank(args.macro_length, args.push_scale)
    pixel_action_mapping, probe_diagnostics = _fit_rgb_pixel_action_mapping(
        seed=args.seed,
        tasks=[int(task) for task in args.tasks],
        roots=args.calibration_roots,
        descent_steps=args.descent_steps,
        probe_steps=args.probe_steps,
    )
    calibration = [
        _capture_rgb_approach_branch(
            seed=args.seed + i, task_id=int(args.tasks[i % len(args.tasks)]),
            approach_steps=args.approach_steps,
            descent_steps=args.descent_steps,
            pixel_action_mapping=pixel_action_mapping,
            approach_gain=args.approach_gain,
            approach_threshold=args.approach_threshold,
        )
        for i in range(args.calibration_roots)
    ]
    features = _physical_features(
        calibration, bank, args.device, args.contact_aware_features,
    )
    target_builder = (
        _branch_macro_effects
        if args.calibration_target == "privileged_task"
        else _branch_rgb_macro_effects
    )
    targets = torch.as_tensor(
        np.concatenate([target_builder(branch, bank) for branch in calibration], axis=0),
        device=args.device, dtype=torch.float32,
    )
    fit = _fit(features, targets)
    test = [
        _episode(
            seed=args.seed + 1000 + i,
            task_id=int(args.tasks[i % len(args.tasks)]),
            bank=bank, fit=fit, device=args.device,
            action_block=args.action_block, macro_cycles=args.macro_cycles,
            approach_steps=args.approach_steps,
            descent_steps=args.descent_steps,
            pixel_action_mapping=pixel_action_mapping,
            push_feedback=not args.disable_push_feedback,
            push_feedback_threshold=args.push_feedback_threshold,
            effect_adaptation=args.effect_adaptation,
            reclose_each_cycle=args.reclose_each_cycle,
            stop_on_rgb_regression=not args.allow_rgb_regression,
            push_backtrack=args.push_backtrack,
            stop_push_on_regression=args.stop_push_on_regression,
            post_approach_descend_steps=args.post_approach_descend_steps,
            post_approach_close_steps=args.post_approach_close_steps,
            approach_gain=args.approach_gain,
            approach_threshold=args.approach_threshold,
            contact_probe=args.contact_probe,
            contact_probe_steps=args.contact_probe_steps,
            contact_aware_features=args.contact_aware_features,
            mode_conditioned_reclose=args.mode_conditioned_reclose,
            min_cycle_rgb_progress=args.min_cycle_rgb_progress,
        )
        for i in range(args.test_roots)
    ]
    payload = {
        "schema": "live_cube_rgb_macro_ranker.v1",
        "protocol": {
            "device": args.device,
            "calibration_roots": args.calibration_roots,
            "test_roots": args.test_roots,
            "root_disjoint": True,
            "runtime_input": "current RGB, previous RGB, rendered RGB goal, previous action",
            "runtime_state_access": False,
            "bank": (
                f"{args.macro_length}-step multiphase factorized approach/push macro"
                if args.multiphase_bank
                else
                f"{args.macro_length}-step factorized approach/push macro"
                if args.factorized_bank
                else f"{args.macro_length}-step four-direction macro"
            ),
            "bank_factorization": (
                "independent approach/two persistent push directions"
                if args.multiphase_bank
                else "independent approach/push directions"
                if args.factorized_bank
                else "shared approach/push direction"
            ),
            "candidate_count": int(len(bank)),
            "push_scale": args.push_scale,
            "action_block": args.action_block,
            "macro_cycles": args.macro_cycles,
            "approach_steps": args.approach_steps,
            "approach_gain": args.approach_gain,
            "approach_threshold": args.approach_threshold,
            "contact_probe": args.contact_probe,
            "contact_probe_steps": args.contact_probe_steps,
            "contact_aware_features": args.contact_aware_features,
            "descent_steps": args.descent_steps,
            "probe_steps": args.probe_steps,
            "rgb_probe_calibration": probe_diagnostics,
            "push_feedback": not args.disable_push_feedback,
            "push_feedback_threshold": args.push_feedback_threshold,
            "push_backtrack": args.push_backtrack,
            "stop_push_on_regression": args.stop_push_on_regression,
            "effect_adaptation": args.effect_adaptation,
            "reclose_each_cycle": args.reclose_each_cycle,
            "mode_conditioned_reclose": args.mode_conditioned_reclose,
            "min_cycle_rgb_progress": args.min_cycle_rgb_progress,
            "stop_on_rgb_regression": not args.allow_rgb_regression,
            "post_approach_descend_steps": args.post_approach_descend_steps,
            "post_approach_close_steps": args.post_approach_close_steps,
            "fit": "calibration roots only",
            "calibration_target": args.calibration_target,
            "cuda_runtime": cuda_runtime,
        },
        "test": test,
        "aggregate": {
            "roots": len(test),
            "success_rate": float(np.mean([row["success"] for row in test])),
            "selected_macro_histogram": np.bincount(
                [index for row in test for index in row["selected_macro_indices"]],
                minlength=len(bank),
            ).astype(int).tolist(),
            "zero_macro_rate": float(np.mean([
                row["zero_macro_selected"] for row in test
            ])),
            "approach_valid_rate": float(np.mean([
                all(item["block_valid"] and item["effector_valid"]
                    for item in row["approach_trace"])
                for row in test
            ])),
            "contact_observed_rate": float(np.mean([
                row.get("contact_observed", False) for row in test
            ])),
        },
        "claim_boundary": (
            "RGB-only macro-ranker online transfer diagnostic; no universal "
            "world-model or causal certificate claim"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
