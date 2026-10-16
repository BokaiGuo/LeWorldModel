"""Post-contact hybrid Cube consequence for the residual negative result.

The experiment holds the successful RGB approach/close protocol fixed and
changes only the controller used after contact is established.  This isolates
whether frozen LeWM temporal features provide an incremental action-transfer
benefit after the previously diagnosed support failure has been repaired.

Compared arms:

* ``latent_static``: current RGB plus action, no previous frame;
* ``latent_temporal``: current/previous RGB plus previous action;
* ``latent_temporal_pixel_guarded``: temporal latent proposal with a fixed RGB
  successor-error guard that falls back to pixel feedback after a worsening
  step;
* ``rgb_pixel_adaptive``: the already validated RGB short-pulse controller.

Calibration is root-disjoint from deployment and uses no deployment outcome.
The default branch constructor is RGB-only.  A separate ``privileged_world``
branch-isolation mode is available when the scientific question is strictly
post-contact transfer: it uses simulator geometry only to construct a shared
contact starting state, while all three controller arms remain RGB-only.  That
mode must not be reported as evidence for RGB visual approach.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Import PyTorch before configuring the headless renderer so the managed CUDA
# runtime keeps the host device visible.
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import stable_worldmodel  # noqa: F401 - registers environments
import mujoco
from ogbench.manipspace import lie

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_cube_rgb_observability import (  # noqa: E402
    _target_red_centroid,
    _vec,
)
from experiments.audit_cube_temporal_action_effect import (  # noqa: E402
    build_feature_families,
)
from experiments.audit_cube_temporal_action_ranking import (  # noqa: E402
    _branch_observed_effects,
    _candidate_bank,
    _fit_readout,
    _predict,
)
from experiments.checkpoint_compat import load_compatible_pretrained  # noqa: E402
from experiments.run_cube_rgb_visual_servo import (  # noqa: E402
    GRIPPER_CLOSE_ACTION,
    _action_from_pixel_error,
    _capture_initial,
    _fit_rgb_action_map,
    _image_xy,
    _make_env,
    _predict_linear,
    _relative_approach_features,
    _relative_push_features,
    _reset,
)


def _latent_action(
    *, current: np.ndarray, previous: np.ndarray, goal: np.ndarray,
    previous_action: np.ndarray, candidates: np.ndarray, model: torch.nn.Module,
    fit: Any, device: str,
    batch_size: int, temporal: bool, family_name: str | None = None,
) -> tuple[np.ndarray, float]:
    """Rank the fixed action bank using only deployment-visible inputs."""
    n_candidates = len(candidates)
    current_batch = np.repeat(current[None], n_candidates, axis=0)
    previous_batch = np.repeat(previous[None], n_candidates, axis=0)
    goal_batch = np.repeat(goal[None], n_candidates, axis=0)
    previous_actions = np.repeat(previous_action[None], n_candidates, axis=0)
    valid = np.ones(n_candidates, dtype=np.float32)
    families, event_score = build_feature_families(
        current_batch,
        previous_batch,
        goal_batch,
        candidates,
        previous_actions,
        valid,
        device=device,
        model=model,
        batch_size=batch_size,
    )
    name = family_name or ("latent_temporal" if temporal else "latent_static")
    if isinstance(fit, dict) and fit.get("kind") == "event_gated_residual":
        base_prediction = _predict(families["latent_static"], fit["base_fit"])
        residual_features = families["latent_residual_compact"]
        residual_prediction = _predict(residual_features, fit["residual_fit"])
        standardized = (
            residual_features.to(dtype=torch.float64) - fit["residual_mean"]
        ) / fit["residual_scale"].clamp_min(1.0e-4)
        support_score = torch.sqrt(torch.mean(standardized.square(), dim=1))
        gate = (
            (event_score.to(dtype=torch.float64) > float(fit["event_threshold"]))
            & (support_score <= float(fit["support_threshold"]))
        )
        prediction = base_prediction + gate[:, None].to(base_prediction.dtype) * residual_prediction
    else:
        prediction = _predict(families[name], fit)
    scores = torch.linalg.vector_norm(prediction, dim=1)
    index = int(scores.argmin().item())
    return np.asarray(candidates[index], dtype=np.float32), float(scores[index].item())


def _pixel_guard_should_fallback(
    last_error: float | None,
    current_error: float | None,
    already_triggered: bool,
    threshold: float,
) -> bool:
    """Return the irreversible RGB fallback decision for one push step."""
    if already_triggered or last_error is None or current_error is None:
        return bool(already_triggered)
    if threshold < 0.0:
        raise ValueError("pixel guard threshold must be non-negative")
    return bool(current_error > last_error + threshold)


def _teleport_pinch_xy_to_block(raw: Any, block_world: np.ndarray) -> dict[str, Any]:
    """Construct a clean post-contact state with simulator-side IK.

    This helper is intentionally restricted to the privileged branch
    constructor.  It never runs in a controller arm and its returned
    diagnostics are evaluator-side metadata only.  The target keeps the
    current pinch orientation and Z coordinate, changing only XY to the
    physical target block center.  Using the environment's own
    pinch-to-attachment transform avoids the action-frame mismatch that made
    the older ``privileged_world`` approach fail before contact.
    """
    block_world = np.asarray(block_world, dtype=np.float64).reshape(3)
    pinch_id = int(raw._pinch_site_id)
    current_pos = np.asarray(raw.data.site_xpos[pinch_id], dtype=np.float64).copy()
    current_rot = lie.SO3.from_matrix(
        np.asarray(raw.data.site_xmat[pinch_id], dtype=np.float64).reshape(3, 3)
    )
    target_pos = current_pos.copy()
    target_pos[:2] = block_world[:2]
    target_pos[2] = float(np.clip(
        target_pos[2],
        float(np.asarray(raw._workspace_bounds)[0, 2]),
        float(np.asarray(raw._workspace_bounds)[1, 2]),
    ))
    target_pose = lie.SE3.from_rotation_and_translation(
        rotation=current_rot,
        translation=target_pos,
    )
    target_attach = target_pose @ raw._T_pa
    arm_ids = np.asarray(raw._arm_joint_ids, dtype=np.int64)
    qpos_target = raw._ik.solve(
        pos=target_attach.translation(),
        quat=target_attach.rotation().wxyz,
        curr_qpos=np.asarray(raw.data.qpos[arm_ids], dtype=np.float64),
    )
    raw.data.qpos[arm_ids] = np.asarray(qpos_target, dtype=np.float64)
    raw.data.qvel[arm_ids] = 0.0
    raw.data.ctrl[np.asarray(raw._arm_actuator_ids, dtype=np.int64)] = qpos_target
    mujoco.mj_forward(raw.model, raw.data)
    achieved = np.asarray(raw.data.site_xpos[pinch_id], dtype=np.float64)
    return {
        "target_xy_m": target_pos[:2].tolist(),
        "target_z_m": float(target_pos[2]),
        "xy_error_before_m": float(np.linalg.norm(current_pos[:2] - block_world[:2])),
        "xy_error_after_m": float(np.linalg.norm(achieved[:2] - block_world[:2])),
        "z_error_after_m": float(abs(achieved[2] - target_pos[2])),
        "ik_target_position_m": target_pos.tolist(),
    }


def _capture_contact_branch(
    initial: dict[str, Any], rgb_model: dict[str, np.ndarray], *,
    descend_steps: int, close_steps: int, approach_threshold: float,
    world_action_gain: float, approach_standoff_m: float, max_steps: int,
    branch_controller: str = "rgb",
) -> dict[str, Any]:
    """Run the shared RGB contact protocol once and freeze its branch state.

    Separate MuJoCo renderers can produce small RGB differences even when
    qpos/qvel are identical.  Re-running the approach independently for each
    arm therefore breaks paired intervention.  This helper constructs one
    shared branch, then exports the exact current/previous frames and state to
    all arms; the policy comparison starts only after the close action.
    """
    if branch_controller not in {"rgb", "privileged_world", "privileged_ik_contact"}:
        raise ValueError(f"unknown branch controller: {branch_controller}")
    env = _make_env()
    try:
        obs, info, goal_image = _reset(env, initial)
        previous = obs.copy()
        previous_action = np.zeros(5, dtype=np.float32)
        phase, phase_steps = "descend", 0
        approach_errors: list[float] = []
        ik_contact_report: dict[str, Any] | None = None
        for _ in range(max_steps):
            goal_xy, goal_valid = _image_xy(goal_image, effector=False)
            block_xy, block_valid = _image_xy(
                obs, effector=False,
                target_reference_y=float(goal_xy[1]) if goal_valid else None,
            )
            effector_xy, effector_valid = _image_xy(obs, effector=True)
            action = np.zeros(5, dtype=np.float32)
            if phase == "descend":
                action[2] = -1.0
                phase_steps += 1
                if phase_steps >= descend_steps:
                    phase, phase_steps = "approach", 0
            elif phase == "approach":
                if branch_controller == "privileged_ik_contact":
                    # This is a branch-construction-only intervention.  It
                    # deliberately removes the previously diagnosed RGB
                    # approach bottleneck by solving a simulator-side IK
                    # target whose XY equals the physical cube center while
                    # preserving the current height and orientation.  No
                    # state derived here is passed to any controller arm.
                    raw = env.unwrapped
                    block_world = _vec(info, "privileged/block_0_pos", 3).copy()
                    ik_contact_report = _teleport_pinch_xy_to_block(raw, block_world)
                    action[4] = GRIPPER_CLOSE_ACTION
                    phase = "close"
                    phase_steps = 1
                    approach_errors.append(float(ik_contact_report["xy_error_after_m"]))
                else:
                    if branch_controller == "privileged_world":
                        # This branch-isolation mode deliberately uses
                        # simulator geometry only to establish a common
                        # post-contact state.  The state, frames, and
                        # previous close action are frozen before any
                        # controller arm is evaluated.
                        world_error = (
                            _vec(info, "privileged/block_0_pos", 3)[:2]
                            - _vec(info, "proprio/effector_pos", 3)[:2]
                        ).astype(np.float32)
                    else:
                        if not (goal_valid and block_valid and effector_valid):
                            raise RuntimeError("shared RGB contact branch lost a required landmark")
                        world_error = _predict_linear(
                            _relative_approach_features(effector_xy, block_xy, goal_xy),
                            rgb_model["relative_approach_chart"],
                        )
                        goal_delta = _predict_linear(
                            _relative_push_features(block_xy, goal_xy),
                            rgb_model["relative_push_chart"],
                        )
                        goal_norm = float(np.linalg.norm(goal_delta))
                        if goal_norm > 1.0e-6 and approach_standoff_m > 0.0:
                            world_error = world_error - (
                                float(approach_standoff_m) * goal_delta / goal_norm
                            )
                    error_norm = float(np.linalg.norm(world_error))
                    approach_errors.append(error_norm)
                    if error_norm <= approach_threshold:
                        phase, phase_steps = "close", 0
                    else:
                        inverse = np.linalg.pinv(
                            np.asarray(rgb_model["world_action_mapping_per_step"], dtype=np.float64),
                            rcond=1.0e-4,
                        )
                        action[:2] = np.clip(
                            float(world_action_gain) * (world_error @ inverse), -1.0, 1.0,
                        ).astype(np.float32)
            elif phase == "close":
                action[4] = GRIPPER_CLOSE_ACTION
                phase_steps += 1
                if phase_steps >= close_steps:
                    phase, phase_steps = "push", 0
            else:
                break
            next_obs, _, terminated, truncated, next_info = env.step(action)
            previous, obs, previous_action, info = (
                obs,
                np.asarray(next_obs, dtype=np.uint8).copy(),
                action.copy(),
                next_info,
            )
            if terminated or truncated:
                raise RuntimeError("shared RGB contact branch terminated before push")
            if phase == "push":
                raw = env.unwrapped
                state = np.concatenate([raw.data.qpos.copy(), raw.data.qvel.copy()])
                return {
                    "seed": int(initial["seed"]),
                    "task_id": int(initial["task_id"]),
                    "variation_values": copy.deepcopy(raw.variation_space.value),
                    "state": state.tolist(),
                    "goal": np.asarray(initial["goal"], dtype=np.float32)[:2].copy().tolist(),
                    "current_image": obs.copy(),
                    "previous_image": previous.copy(),
                    "goal_image": goal_image.copy(),
                    "previous_action": previous_action.copy(),
                    "shared_prelude_steps": int(len(approach_errors) + descend_steps + close_steps),
                    "approach_error_trace": approach_errors,
                    "branch_controller": branch_controller,
                    "ik_contact_report": copy.deepcopy(ik_contact_report),
                }
        # Never let a failed branch become ``None`` and contaminate the
        # calibration fit.  The branch protocol is part of the estimand, so a
        # root that cannot establish RGB contact must be reported explicitly
        # and the run must fail before producing a result artifact.
        raise RuntimeError(
            "shared RGB contact branch did not reach push phase "
            f"for seed={initial['seed']} task={initial['task_id']} "
            f"after {max_steps} steps; phase={phase} "
            f"approach_trace_len={len(approach_errors)} "
            f"last_error_m={approach_errors[-1] if approach_errors else None}"
        )
    finally:
        env.close()


def _snapshot_replay_state(raw: Any) -> dict[str, Any]:
    """Snapshot MuJoCo state that affects the next one-step transition."""
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    physics = np.empty(mujoco.mj_stateSize(raw.model, spec), dtype=np.float64)
    mujoco.mj_getState(raw.model, raw.data, physics, spec)
    arrays = {}
    for name in (
        "ctrl", "qfrc_applied", "xfrc_applied", "qacc_warmstart",
        "mocap_pos", "mocap_quat", "userdata", "plugin_state",
    ):
        value = getattr(raw.data, name, None)
        if value is not None:
            arrays[name] = np.asarray(value).copy()
    return {"physics": physics, "arrays": arrays}


def _restore_replay_state(raw: Any, snapshot: dict[str, Any]) -> None:
    """Restore a snapshot and rebuild observation-side derived state."""
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    mujoco.mj_setState(raw.model, raw.data, snapshot["physics"], spec)
    for name, value in snapshot["arrays"].items():
        getattr(raw.data, name)[:] = value
    raw._reset_next_step = False
    raw.pre_step()
    mujoco.mj_forward(raw.model, raw.data)
    raw.post_step()


def _collect_sequence_calibration(
    branches: list[dict[str, Any]], candidates: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Collect matched history frames and labels with bounded reset count.

    For each root, candidate transitions are replayed from a complete
    MuJoCo-data snapshot.  Each previous-action successor receives one normal
    state reset before its candidate bank, matching the original label
    protocol while avoiding one reset per candidate.
    """
    examples: list[dict[str, Any]] = list(branches)
    effects: list[np.ndarray] = []
    env = _make_env()
    try:
        for branch in branches:
            _reset(env, branch)
            raw = env.unwrapped
            branch_snapshot = _snapshot_replay_state(raw)

            def replay_candidates(snapshot: dict[str, Any]) -> np.ndarray:
                root_effects: list[np.ndarray] = []
                for action in candidates:
                    _restore_replay_state(raw, snapshot)
                    _, _, _, _, next_info = env.step(
                        np.asarray(action, dtype=np.float32),
                    )
                    next_block = _vec(next_info, "privileged/block_0_pos", 3)[:2]
                    root_effects.append(
                        np.asarray(branch["goal"], dtype=np.float32) - next_block
                    )
                return np.asarray(root_effects, dtype=np.float32)

            effects.append(replay_candidates(branch_snapshot))
            for previous_action in candidates:
                _restore_replay_state(raw, branch_snapshot)
                next_obs, _, terminated, truncated, _ = env.step(
                    np.asarray(previous_action, dtype=np.float32),
                )
                if terminated or truncated:
                    raise RuntimeError(
                        "calibration previous-action branch terminated "
                        f"for seed={branch['seed']} task={branch['task_id']}"
                    )
                after_state = np.concatenate([
                    raw.data.qpos.copy(), raw.data.qvel.copy(),
                ]).astype(np.float64)
                examples.append({
                    "seed": int(branch["seed"]),
                    "task_id": int(branch["task_id"]),
                    "variation_values": copy.deepcopy(branch["variation_values"]),
                    "state": after_state.tolist(),
                    "goal": np.asarray(branch["goal"], dtype=np.float32).copy().tolist(),
                    "current_image": np.asarray(next_obs, dtype=np.uint8).copy(),
                    "previous_image": np.asarray(branch["current_image"], dtype=np.uint8).copy(),
                    "goal_image": np.asarray(branch["goal_image"], dtype=np.uint8).copy(),
                    "previous_action": np.asarray(previous_action, dtype=np.float32).copy(),
                    "shared_prelude_steps": int(branch["shared_prelude_steps"] + 1),
                    "approach_error_trace": branch["approach_error_trace"],
                    "branch_controller": branch.get("branch_controller", "unknown"),
                })
                # The old protocol reset from the after-action qpos/qvel
                # before labeling its next candidate.  Retain that semantic
                # boundary while reusing the compiled environment.
                after_branch = {
                    **branch,
                    "state": after_state.tolist(),
                }
                _reset(env, after_branch)
                after_snapshot = _snapshot_replay_state(raw)
                effects.append(replay_candidates(after_snapshot))
    finally:
        env.close()
    return examples, np.concatenate(effects, axis=0).astype(np.float32)


def _fit_post_contact_readouts(
    branches: list[dict[str, Any]], candidates: np.ndarray,
    *, model: torch.nn.Module, device: str, batch_size: int,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dict[str, Any],
    dict[str, Any],
]:
    """Fit static/temporal readouts on the exact post-contact protocol.

    This removes the generic-history mismatch diagnosed in the first hybrid
    panel: every calibration example starts from an RGB-established branch,
    uses the close action as ``previous_action``, and is labeled by the same
    one-step candidate intervention used at deployment.
    """
    n_candidates = len(candidates)

    # Include the initial post-contact frame and one fixed candidate-action
    # transition per root.  This is a calibration-only support expansion,
    # not an online selector and not a test-outcome-dependent augmentation.
    calibration_examples, effects = _collect_sequence_calibration(branches, candidates)
    current = np.repeat(
        np.asarray([b["current_image"] for b in calibration_examples]),
        n_candidates, axis=0,
    )
    previous = np.repeat(
        np.asarray([b["previous_image"] for b in calibration_examples]),
        n_candidates, axis=0,
    )
    goals = np.repeat(
        np.asarray([b["goal_image"] for b in calibration_examples]),
        n_candidates, axis=0,
    )
    actions = np.tile(candidates, (len(calibration_examples), 1))
    previous_actions = np.repeat(
        np.asarray([b["previous_action"] for b in calibration_examples]),
        n_candidates, axis=0,
    )
    valid = np.ones(len(calibration_examples) * n_candidates, dtype=np.float32)
    families, event_score = build_feature_families(
        current,
        previous,
        goals,
        actions,
        previous_actions,
        valid,
        device=device,
        model=model,
        batch_size=batch_size,
    )
    target = torch.as_tensor(effects, device=device, dtype=torch.float32)
    static_fit = _fit_readout(families["latent_static"], target)
    temporal_fit = _fit_readout(families["latent_temporal"], target)
    compact_fit = _fit_readout(families["latent_temporal_compact"], target)
    residual_features = families["latent_residual_compact"]
    base_prediction = _predict(families["latent_static"], static_fit).detach()
    residual_target = target - base_prediction
    residual_fit = _fit_readout(residual_features, residual_target)
    residual_mean, residual_scale, _ = residual_fit
    residual_standardized = (
        residual_features.to(dtype=torch.float64) - residual_mean
    ) / residual_scale.clamp_min(1.0e-4)
    residual_support = torch.sqrt(torch.mean(residual_standardized.square(), dim=1))
    event_threshold = float(torch.quantile(event_score, 0.75).item())
    support_threshold = float(torch.quantile(residual_support, 0.95).item())
    gated_fit = {
        "kind": "event_gated_residual",
        "base_fit": static_fit,
        "residual_fit": residual_fit,
        "residual_mean": residual_mean,
        "residual_scale": residual_scale,
        "event_threshold": event_threshold,
        "support_threshold": support_threshold,
    }
    return static_fit, temporal_fit, compact_fit, gated_fit, {
        "roots": int(len(branches)),
        "expanded_examples": int(len(calibration_examples)),
        "frames": int(len(calibration_examples) * n_candidates),
        "previous_action_protocol": (
            "initial post-contact close action plus one fixed candidate-bank "
            "previous-action transition per calibration root"
        ),
        "candidate_bank_shared": True,
        "families": {
            "latent_static": int(families["latent_static"].shape[1]),
            "latent_temporal": int(families["latent_temporal"].shape[1]),
            "latent_temporal_compact": int(families["latent_temporal_compact"].shape[1]),
            "latent_residual_compact": int(families["latent_residual_compact"].shape[1]),
        },
        "compact_temporal_definition": (
            "fixed 24-bin adaptive average pooling of current/goal/goal-delta/"
            "innovation latents plus action, previous close action, and validity"
        ),
        "gated_residual_definition": (
            "static base plus latent_residual_compact only when event score exceeds "
            "the calibration 0.75 quantile and residual support is within the "
            "calibration 0.95 quantile"
        ),
        "event_threshold_calibration_only": event_threshold,
        "support_threshold_calibration_only": support_threshold,
        "mean_target_norm": float(np.linalg.norm(effects, axis=1).mean()),
        "target_norm_std": float(np.linalg.norm(effects, axis=1).std()),
    }


def _run_arm_unpaired_legacy(
    *, initials: list[dict[str, Any]], model: torch.nn.Module,
    latent_fit: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    static_fit: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rgb_model: dict[str, np.ndarray], candidates: np.ndarray, device: str,
    batch_size: int, method: str, max_steps: int, descend_steps: int,
    close_steps: int, push_steps: int, approach_threshold: float,
    world_action_gain: float, approach_standoff_m: float, push_pixel_gain: float,
) -> dict[str, Any]:
    """Same protocol as ``_run_arm`` with one MuJoCo step per loop iteration."""
    if method not in {"latent_static", "latent_temporal", "rgb_pixel_adaptive"}:
        raise ValueError(f"unknown method: {method}")
    episodes: list[dict[str, Any]] = []
    for initial in initials:
        env = _make_env()
        try:
            obs, info, goal_image = _reset(env, initial)
            previous = obs.copy()
            previous_action = np.zeros(5, dtype=np.float32)
            phase, phase_steps = "descend", 0
            actions: list[list[float]] = []
            phase_trace: list[str] = []
            distances: list[float] = []
            contact_trace: list[float] = []
            block_trace: list[list[float]] = []
            effector_trace: list[list[float]] = []
            push_scores: list[float] = []
            approach_errors: list[float] = []
            push_errors: list[float] = []
            adaptation_sign = 1.0
            last_push_error: float | None = None

            for _ in range(max_steps):
                goal_xy, goal_valid = _image_xy(goal_image, effector=False)
                block_xy, block_valid = _image_xy(
                    obs, effector=False,
                    target_reference_y=float(goal_xy[1]) if goal_valid else None,
                )
                effector_xy, effector_valid = _image_xy(obs, effector=True)
                action = np.zeros(5, dtype=np.float32)
                if phase == "descend":
                    action[2] = -1.0
                    phase_steps += 1
                    if phase_steps >= descend_steps:
                        phase, phase_steps = "approach", 0
                elif phase == "approach":
                    if not (goal_valid and block_valid and effector_valid):
                        phase, phase_steps = "fallback", 0
                    else:
                        world_error = _predict_linear(
                            _relative_approach_features(effector_xy, block_xy, goal_xy),
                            rgb_model["relative_approach_chart"],
                        )
                        goal_delta = _predict_linear(
                            _relative_push_features(block_xy, goal_xy),
                            rgb_model["relative_push_chart"],
                        )
                        goal_norm = float(np.linalg.norm(goal_delta))
                        if goal_norm > 1.0e-6 and approach_standoff_m > 0.0:
                            world_error = world_error - (
                                float(approach_standoff_m) * goal_delta / goal_norm
                            )
                        error_norm = float(np.linalg.norm(world_error))
                        approach_errors.append(error_norm)
                        if error_norm <= approach_threshold:
                            phase, phase_steps = "close", 0
                        else:
                            inverse = np.linalg.pinv(
                                np.asarray(rgb_model["world_action_mapping_per_step"], dtype=np.float64),
                                rcond=1.0e-4,
                            )
                            action[:2] = np.clip(
                                float(world_action_gain) * (world_error @ inverse), -1.0, 1.0,
                            ).astype(np.float32)
                elif phase == "close":
                    action[4] = GRIPPER_CLOSE_ACTION
                    phase_steps += 1
                    if phase_steps >= close_steps:
                        phase, phase_steps = "push", 0
                elif phase == "push":
                    if method == "latent_static":
                        action[:], score = _latent_action(
                            current=obs, previous=previous, goal=goal_image,
                            previous_action=previous_action, candidates=candidates,
                            model=model, fit=static_fit, device=device,
                            batch_size=batch_size, temporal=False,
                        )
                        push_scores.append(score)
                    elif method == "latent_temporal":
                        action[:], score = _latent_action(
                            current=obs, previous=previous, goal=goal_image,
                            previous_action=previous_action, candidates=candidates,
                            model=model, fit=latent_fit, device=device,
                            batch_size=batch_size, temporal=True,
                        )
                        push_scores.append(score)
                    elif goal_valid and block_valid:
                        pixel_error = np.asarray(goal_xy - block_xy, dtype=np.float32)
                        error_norm = float(np.linalg.norm(pixel_error))
                        push_errors.append(error_norm)
                        if last_push_error is not None and error_norm > last_push_error + 1.0e-3:
                            adaptation_sign *= -1.0
                        action[:2] = np.clip(
                            adaptation_sign * _action_from_pixel_error(
                                pixel_error, rgb_model["pixel_action_mapping"], push_pixel_gain,
                            ), -1.0, 1.0,
                        ).astype(np.float32)
                        last_push_error = error_norm
                    phase_steps += 1
                    if phase_steps >= push_steps:
                        phase, phase_steps = "done", 0
                elif phase in {"done", "fallback"}:
                    action[:] = 0.0

                phase_trace.append(phase)
                actions.append(action.tolist())
                contact_trace.append(float(np.max(np.asarray(
                    info.get("proprio/gripper_contact", [0.0]), dtype=np.float32,
                ))))
                block_trace.append(_vec(info, "privileged/block_0_pos", 3).tolist())
                effector_trace.append(_vec(info, "proprio/effector_pos", 3).tolist())
                next_obs, _, terminated, truncated, next_info = env.step(action)
                block = _vec(next_info, "privileged/block_0_pos", 3)[:2]
                distances.append(float(np.linalg.norm(block - np.asarray(initial["goal"])[:2])))
                previous, obs, previous_action, info = (
                    obs,
                    np.asarray(next_obs, dtype=np.uint8).copy(),
                    action.copy(),
                    next_info,
                )
                if terminated or truncated:
                    break

            final_block = _vec(info, "privileged/block_0_pos", 3)[:2]
            episodes.append({
                "seed": int(initial["seed"]),
                "task_id": int(initial["task_id"]),
                "steps": len(actions),
                "final_goal_distance_m": float(np.linalg.norm(final_block - np.asarray(initial["goal"])[:2])),
                "best_goal_distance_m": float(np.min(distances)) if distances else None,
                "actions": actions,
                "phase_trace": phase_trace,
                "contact_trace_evaluator_only": contact_trace,
                "contact_steps_evaluator_only": int(np.sum(np.asarray(contact_trace) > 1.0e-5)),
                "block_trace_evaluator_only": block_trace,
                "effector_trace_evaluator_only": effector_trace,
                "approach_error_trace_runtime": approach_errors,
                "push_error_trace_runtime": push_errors,
                "push_score_trace_runtime": push_scores,
                "adaptation_sign_final": float(adaptation_sign),
            })
        finally:
            env.close()
    return {
        "method": method,
        "episodes": episodes,
        "mean_final_goal_distance_m": float(np.mean([r["final_goal_distance_m"] for r in episodes])),
        "mean_best_goal_distance_m": float(np.mean([r["best_goal_distance_m"] for r in episodes])),
    }


def _run_branch_arm_on_env(
    *, branches: list[dict[str, Any]], model: torch.nn.Module,
    latent_fit: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    compact_fit: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    gated_fit: dict[str, Any],
    static_fit: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rgb_model: dict[str, np.ndarray], candidates: np.ndarray, device: str,
    batch_size: int, method: str, push_steps: int, push_pixel_gain: float,
    pixel_guard_threshold: float,
    env: Any,
) -> dict[str, Any]:
    """Evaluate one controller from already paired post-contact branches."""
    if method not in {
        "latent_static", "latent_temporal", "latent_temporal_compact",
        "latent_gated_residual", "latent_temporal_pixel_guarded", "rgb_pixel_adaptive",
    }:
        raise ValueError(f"unknown method: {method}")
    episodes: list[dict[str, Any]] = []
    for branch in branches:
        try:
            _, info, _ = _reset(env, branch)
            raw = env.unwrapped
            state = np.asarray(branch["state"], dtype=np.float64)
            nq = int(raw.data.qpos.size)
            raw.data.qpos[:] = state[:nq]
            raw.data.qvel[:] = state[nq:nq + int(raw.data.qvel.size)]
            mujoco.mj_forward(raw.model, raw.data)
            obs = np.asarray(branch["current_image"], dtype=np.uint8).copy()
            previous = np.asarray(branch["previous_image"], dtype=np.uint8).copy()
            goal_image = np.asarray(branch["goal_image"], dtype=np.uint8).copy()
            previous_action = np.asarray(branch["previous_action"], dtype=np.float32).copy()
            actions: list[list[float]] = []
            phase_trace: list[str] = []
            distances: list[float] = []
            contact_trace: list[float] = []
            block_trace: list[list[float]] = []
            effector_trace: list[list[float]] = []
            push_scores: list[float] = []
            push_errors: list[float] = []
            adaptation_sign = 1.0
            last_push_error: float | None = None
            pixel_guard_triggered = False
            pixel_guard_trace: list[bool] = []
            goal_xy, goal_valid = _image_xy(goal_image, effector=False)

            for _ in range(push_steps):
                block_xy, block_valid = _image_xy(
                    obs,
                    effector=False,
                    target_reference_y=float(goal_xy[1]) if goal_valid else None,
                )
                action = np.zeros(5, dtype=np.float32)
                pixel_error = None
                error_norm = None
                if goal_valid and block_valid:
                    pixel_error = np.asarray(goal_xy - block_xy, dtype=np.float32)
                    error_norm = float(np.linalg.norm(pixel_error))
                    if method == "latent_temporal_pixel_guarded":
                        pixel_guard_triggered = _pixel_guard_should_fallback(
                            last_push_error,
                            error_norm,
                            pixel_guard_triggered,
                            pixel_guard_threshold,
                        )
                        push_errors.append(error_norm)
                if method == "latent_static":
                    action[:], score = _latent_action(
                        current=obs, previous=previous, goal=goal_image,
                        previous_action=previous_action, candidates=candidates,
                        model=model, fit=static_fit, device=device,
                        batch_size=batch_size, temporal=False,
                    )
                    push_scores.append(score)
                elif method == "latent_temporal":
                    action[:], score = _latent_action(
                        current=obs, previous=previous, goal=goal_image,
                        previous_action=previous_action, candidates=candidates,
                        model=model, fit=latent_fit, device=device,
                        batch_size=batch_size, temporal=True,
                    )
                    push_scores.append(score)
                elif method == "latent_temporal_compact":
                    action[:], score = _latent_action(
                        current=obs, previous=previous, goal=goal_image,
                        previous_action=previous_action, candidates=candidates,
                        model=model, fit=compact_fit, device=device,
                        batch_size=batch_size, temporal=True,
                        family_name="latent_temporal_compact",
                    )
                    push_scores.append(score)
                elif method == "latent_gated_residual":
                    action[:], score = _latent_action(
                        current=obs, previous=previous, goal=goal_image,
                        previous_action=previous_action, candidates=candidates,
                        model=model, fit=gated_fit, device=device,
                        batch_size=batch_size, temporal=True,
                        family_name="latent_residual_compact",
                    )
                    push_scores.append(score)
                elif method == "latent_temporal_pixel_guarded" and not pixel_guard_triggered:
                    action[:], score = _latent_action(
                        current=obs, previous=previous, goal=goal_image,
                        previous_action=previous_action, candidates=candidates,
                        model=model, fit=latent_fit, device=device,
                        batch_size=batch_size, temporal=True,
                    )
                    push_scores.append(score)
                elif goal_valid and block_valid:
                    assert pixel_error is not None and error_norm is not None
                    if last_push_error is not None and error_norm > last_push_error + 1.0e-3:
                        adaptation_sign *= -1.0
                    action[:2] = np.clip(
                        adaptation_sign * _action_from_pixel_error(
                            pixel_error, rgb_model["pixel_action_mapping"], push_pixel_gain,
                        ), -1.0, 1.0,
                    ).astype(np.float32)
                elif method == "latent_temporal_pixel_guarded":
                    action[:], score = _latent_action(
                        current=obs, previous=previous, goal=goal_image,
                        previous_action=previous_action, candidates=candidates,
                        model=model, fit=latent_fit, device=device,
                        batch_size=batch_size, temporal=True,
                    )
                    push_scores.append(score)

                if method == "latent_temporal_pixel_guarded":
                    if error_norm is not None:
                        last_push_error = error_norm
                    pixel_guard_trace.append(bool(pixel_guard_triggered))
                elif method == "rgb_pixel_adaptive" and error_norm is not None:
                    last_push_error = error_norm

                phase_trace.append("push")
                actions.append(action.tolist())
                contact_trace.append(float(np.max(np.asarray(
                    info.get("proprio/gripper_contact", [0.0]), dtype=np.float32,
                ))))
                block_trace.append(_vec(info, "privileged/block_0_pos", 3).tolist())
                effector_trace.append(_vec(info, "proprio/effector_pos", 3).tolist())
                next_obs, _, terminated, truncated, next_info = env.step(action)
                block = _vec(next_info, "privileged/block_0_pos", 3)[:2]
                distances.append(float(np.linalg.norm(block - np.asarray(branch["goal"])[:2])))
                previous, obs, previous_action, info = (
                    obs,
                    np.asarray(next_obs, dtype=np.uint8).copy(),
                    action.copy(),
                    next_info,
                )
                if terminated or truncated:
                    break

            final_block = _vec(info, "privileged/block_0_pos", 3)[:2]
            episodes.append({
                "seed": int(branch["seed"]),
                "task_id": int(branch["task_id"]),
                "steps": len(actions),
                "shared_prelude_steps": int(branch["shared_prelude_steps"]),
                "final_goal_distance_m": float(np.linalg.norm(final_block - np.asarray(branch["goal"])[:2])),
                "best_goal_distance_m": float(np.min(distances)) if distances else None,
                "actions": actions,
                "phase_trace": phase_trace,
                "contact_trace_evaluator_only": contact_trace,
                "contact_steps_evaluator_only": int(np.sum(np.asarray(contact_trace) > 1.0e-5)),
                "block_trace_evaluator_only": block_trace,
                "effector_trace_evaluator_only": effector_trace,
                "push_error_trace_runtime": push_errors,
                "push_score_trace_runtime": push_scores,
                "adaptation_sign_final": float(adaptation_sign),
                "pixel_guard_triggered": bool(pixel_guard_triggered),
                "pixel_guard_trace": pixel_guard_trace,
                "pixel_guard_threshold": float(pixel_guard_threshold),
            })
        finally:
            # The wrapper owns the shared env and closes it once after all
            # root-disjoint branches have been evaluated.
            pass
    return {
        "method": method,
        "episodes": episodes,
        "mean_final_goal_distance_m": float(np.mean([r["final_goal_distance_m"] for r in episodes])),
        "mean_best_goal_distance_m": float(np.mean([r["best_goal_distance_m"] for r in episodes])),
    }


def _run_branch_arm(**kwargs: Any) -> dict[str, Any]:
    """Run one arm with one reusable MuJoCo environment."""
    env = _make_env()
    try:
        return _run_branch_arm_on_env(env=env, **kwargs)
    finally:
        env.close()


def _paired_summary(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_method = {
        name: {row["seed"]: row for row in arm["episodes"]}
        for name, arm in arms.items()
    }
    seeds = list(by_method["latent_static"])
    out: dict[str, Any] = {"seed_order": seeds}
    for left, right in (
        ("latent_temporal", "latent_static"),
        ("latent_temporal_compact", "latent_static"),
        ("latent_temporal_compact", "latent_temporal"),
        ("latent_temporal_compact", "rgb_pixel_adaptive"),
        ("latent_gated_residual", "latent_static"),
        ("latent_gated_residual", "latent_temporal"),
        ("latent_gated_residual", "rgb_pixel_adaptive"),
        ("latent_temporal", "rgb_pixel_adaptive"),
        ("latent_static", "rgb_pixel_adaptive"),
        ("latent_temporal_pixel_guarded", "latent_temporal"),
        ("latent_temporal_pixel_guarded", "latent_static"),
        ("latent_temporal_pixel_guarded", "rgb_pixel_adaptive"),
    ):
        delta = np.asarray([
            by_method[left][seed]["final_goal_distance_m"]
            - by_method[right][seed]["final_goal_distance_m"]
            for seed in seeds
        ], dtype=np.float64)
        key = f"{left}_minus_{right}"
        out[key] = {
            "deltas_m": delta.tolist(),
            "mean_delta_m": float(delta.mean()),
            "wins": int(np.sum(delta < 0.0)),
            "losses": int(np.sum(delta > 0.0)),
            "ties": int(np.sum(delta == 0.0)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/cube/lewm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261220)
    parser.add_argument("--calibration-roots", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--descend-steps", type=int, default=20)
    parser.add_argument("--close-steps", type=int, default=5)
    parser.add_argument("--push-steps", type=int, default=25)
    parser.add_argument("--approach-threshold", type=float, default=0.015)
    parser.add_argument("--world-action-gain", type=float, default=0.20)
    parser.add_argument("--approach-standoff-m", type=float, default=0.0)
    parser.add_argument("--push-pixel-gain", type=float, default=0.20)
    parser.add_argument(
        "--pixel-guard-threshold", type=float, default=0.5,
        help=(
            "fixed RGB pixel-error increase required to switch the temporal latent "
            "proposal to pixel feedback; calibrated independently of test outcomes"
        ),
    )
    parser.add_argument(
        "--candidate-bank", choices=("legacy", "lateral4", "lateral8"),
        default="legacy",
        help=(
            "post-contact action support; legacy preserves the historical bank, "
            "lateral banks exclude z/gripper actions"
        ),
    )
    parser.add_argument(
        "--branch-controller",
        choices=("rgb", "privileged_world", "privileged_ik_contact"),
        default="rgb",
        help=(
            "contact branch law; privileged_world and privileged_ik_contact are "
            "isolation-only modes and are not RGB visual-approach evidence"
        ),
    )
    parser.add_argument("--probe-steps", type=int, default=4)
    parser.add_argument("--chart-ridge", type=float, default=1.0e-2)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for hybrid Cube experiment but unavailable")
    if args.calibration_roots < 8 or args.episodes < 4:
        raise ValueError("need at least 8 calibration roots and 4 test episodes")
    if args.max_steps <= args.descend_steps + args.close_steps:
        raise ValueError("max-steps must leave room for post-contact push")
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        "[purpose] post-contact hybrid Cube consequence; "
        f"device={args.device} calibration={args.calibration_roots} episodes={args.episodes}",
    )

    model = load_compatible_pretrained(args.policy, device=args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    calibration_initials = [
        _capture_initial(args.seed + index, int(args.tasks[index % len(args.tasks)]))
        for index in range(args.calibration_roots)
    ]
    test_initials = [
        _capture_initial(args.seed + 1000 + index, int(args.tasks[index % len(args.tasks)]))
        for index in range(args.episodes)
    ]

    rgb_model, rgb_report = _fit_rgb_action_map(
        calibration_initials,
        args.probe_steps,
        args.descend_steps,
        args.device,
        args.chart_ridge,
    )
    candidates = _candidate_bank(args.candidate_bank)
    calibration_branches = [
        _capture_contact_branch(
            initial,
            rgb_model,
            descend_steps=args.descend_steps,
            close_steps=args.close_steps,
            approach_threshold=args.approach_threshold,
            world_action_gain=args.world_action_gain,
            approach_standoff_m=args.approach_standoff_m,
            max_steps=args.max_steps,
            branch_controller=args.branch_controller,
        )
        for initial in calibration_initials
    ]
    branches = [
        _capture_contact_branch(
            initial,
            rgb_model,
            descend_steps=args.descend_steps,
            close_steps=args.close_steps,
            approach_threshold=args.approach_threshold,
            world_action_gain=args.world_action_gain,
            approach_standoff_m=args.approach_standoff_m,
            max_steps=args.max_steps,
            branch_controller=args.branch_controller,
        )
        for initial in test_initials
    ]
    static_fit, temporal_fit, compact_fit, gated_fit, readout_report = _fit_post_contact_readouts(
        calibration_branches,
        candidates,
        model=model,
        device=args.device,
        batch_size=args.batch_size,
    )

    arms = {
        name: _run_branch_arm(
            branches=branches,
            model=model,
            latent_fit=temporal_fit,
            compact_fit=compact_fit,
            gated_fit=gated_fit,
            static_fit=static_fit,
            rgb_model=rgb_model,
            candidates=candidates,
            device=args.device,
            batch_size=args.batch_size,
            method=name,
            push_steps=args.push_steps,
            push_pixel_gain=args.push_pixel_gain,
            pixel_guard_threshold=args.pixel_guard_threshold,
        )
        for name in (
            "latent_static", "latent_temporal", "latent_temporal_compact",
            "latent_gated_residual", "latent_temporal_pixel_guarded",
            "rgb_pixel_adaptive",
        )
    }
    report = {
        "schema": "cube_hybrid_contact_latent.v5",
        "protocol": {
            "environment": "swm/OGBCube-v0",
            "checkpoint": args.policy,
            "checkpoint_frozen": True,
            "device": args.device,
            "calibration_roots": args.calibration_roots,
            "test_episodes": args.episodes,
            "root_disjoint": True,
            "same_initial_state_pairing": True,
            "paired_branch_construction": "one shared RGB approach/close trajectory per test root; exact qpos/qvel and current/previous/goal frames frozen before controller comparison",
            "runtime_input": "RGB current/previous, rendered RGB goal, previous action, fixed action bank",
            "runtime_privileged_state": False,
            "shared_contact_protocol": "fixed descent -> RGB world-space approach -> gripper close -> controller-specific push",
            "branch_controller": args.branch_controller,
            "branch_controller_claim_boundary": (
                "privileged branch construction only; no privileged state is passed to any arm"
                if args.branch_controller in {"privileged_world", "privileged_ik_contact"} else
                "RGB-only branch construction"
            ),
            "privileged_ik_contact_role": (
                "simulator-side IK aligns pinch XY to physical block and preserves current Z/orientation; construction only"
                if args.branch_controller == "privileged_ik_contact" else None
            ),
            "descend_steps": args.descend_steps,
            "close_steps": args.close_steps,
            "push_steps": args.push_steps,
            "approach_threshold_m": args.approach_threshold,
            "world_action_gain": args.world_action_gain,
            "approach_standoff_m": args.approach_standoff_m,
            "push_pixel_gain": args.push_pixel_gain,
            "pixel_guard_threshold": args.pixel_guard_threshold,
            "readout_fit": "calibration roots only; no test outcomes",
            "candidate_bank": candidates.tolist(),
            "candidate_bank_name": args.candidate_bank,
            "evaluator_state_role": "paired reset and final distance/contact scoring only",
        },
        "calibration": {
            "roots": int(args.calibration_roots),
            "rgb_action_map": rgb_report,
            "post_contact_readout": readout_report,
        },
        "shared_branches": [
            {
                "seed": int(branch["seed"]),
                "task_id": int(branch["task_id"]),
                "shared_prelude_steps": int(branch["shared_prelude_steps"]),
                "approach_error_trace": branch["approach_error_trace"],
                "ik_contact_report": branch.get("ik_contact_report"),
            }
            for branch in branches
        ],
        "arms": arms,
        "paired": _paired_summary(arms),
        "claim_boundary": (
            "Post-contact incremental consequence test. The exploratory "
            "latent_temporal_pixel_guarded arm uses frozen temporal latent ranking "
            "until a fixed RGB pixel-error increase triggers feedback fallback; any "
            "positive result is a hybrid closed-loop repair, not evidence of latent-only "
            "action transfer, universal world-model transfer, causal latent dynamics, "
            "or general commutator certification."
        ),
    }
    (args.output / "hybrid_contact_latent.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report["paired"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
