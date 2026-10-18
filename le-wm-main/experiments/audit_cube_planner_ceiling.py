"""Privileged candidate-bank ceilings for the OGB-Cube RGB protocol.

This audit answers a prerequisite question for interpreting the RGB-only
macro ranker: can the declared action bank solve the task at all from the
same initial roots? Each candidate is replayed from an identical saved
simulator state and scored with the post-hoc privileged cube position. The
privileged score is used only to compute an oracle ceiling; it is never an
input to a deployable policy.

The fixed bank mirrors the online four-direction bank. The expanded bank
adds diagonal directions and low-gain variants, separating a ranker failure
from a support/reachability failure. This is a ceiling diagnostic, not RGB
planning evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers environments

from experiments.audit_cube_action_interventions import (
    _capture_initial_condition,
    _contact,
    _cube_position,
    _effector_position,
    _make_env,
)
from experiments.audit_cube_macro_action_ranking import _macro_bank
from experiments.run_live_cube_rgb_macro_ranker import (
    _fit_rgb_pixel_action_mapping,
    _rgb_block_goal_error,
    _online_macro_bank,
    _rgb_approach_action,
)


def _make_rgb_env() -> gym.Env:
    return gym.make(
        "swm/OGBCube-v0",
        ob_type="pixels",
        render_mode="rgb_array",
        terminate_at_goal=False,
    )


def _expanded_bank(length: int = 40) -> np.ndarray:
    """Build an evaluation-only bank with diagonal and factorized directions.

    The historical expanded bank ties the approach direction to the push
    direction.  That is a support bottleneck for Cube: reaching contact and
    moving the object toward the goal are generally different directions.
    The factorized component below deliberately separates those phases.  It
    is a privileged ceiling intervention only; it is not fed to the learned
    ranker without a separately calibrated action-support protocol.
    """
    base = _macro_bank(length=int(length), direction_count=8, approach_steps=4).copy()
    close = base[1, :, 4] < -0.5
    base[1:, close, 4] = 1.0
    close_end = int(np.flatnonzero(close)[-1]) + 1
    base[1:, close_end:, :2] *= 0.25
    low_gain = base.copy()
    low_gain[1:, close_end:, :2] *= 0.5
    factorized = _factorized_bank(length=int(length), direction_count=8)
    # ``base`` already contains the explicit no-op candidate; avoid adding a
    # second zero row when extending it with the factorized family.
    return np.concatenate([base, low_gain[1:], factorized[1:]], axis=0)


def _factorized_bank(length: int = 40, direction_count: int = 8) -> np.ndarray:
    """Build approach/push-factorized macros for a support-only ceiling.

    Each candidate chooses its approach direction independently from its push
    direction.  The candidate family is intentionally explicit and finite so
    reachability can be audited without silently adding adaptive planning.
    """
    if length < 20 or direction_count not in (4, 8, 16):
        raise ValueError("length must be >=20 and direction_count must be 4, 8, or 16")
    angles = np.arange(direction_count, dtype=np.float32) * (2.0 * np.pi / direction_count)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    approach_end = min(length - 8, 4)
    descend_end = min(length - 8, approach_end + max(6, length // 4))
    close_end = min(length - 2, descend_end + max(4, length // 12))
    candidates: list[np.ndarray] = []
    # Keep zero as an explicit no-op control for parity with all other banks.
    candidates.append(np.zeros((length, 5), dtype=np.float32))
    for approach in directions:
        for push in directions:
            for gain in (1.0, 0.5, 0.25):
                sequence = np.zeros((length, 5), dtype=np.float32)
                sequence[:approach_end, :2] = approach
                sequence[approach_end:descend_end, 2] = -1.0
                sequence[descend_end:close_end, 4] = 1.0
                sequence[close_end:, :2] = gain * push
                candidates.append(sequence)
    return np.stack(candidates, axis=0)


def _multiphase_factorized_bank(length: int = 40, direction_count: int = 8) -> np.ndarray:
    """Build a finite bank with two independent persistent push phases.

    A single push ray can be insufficient even when approach and push are
    factorized: the object may need a second direction after the first
    contact-preserving displacement.  The bank keeps the same fixed prelude,
    then applies two push directions without reopening.  It is an evaluator
    ceiling intervention, not an online learned policy.
    """
    if length < 20 or direction_count not in (4, 8, 16):
        raise ValueError("length must be >=20 and direction_count must be 4, 8, or 16")
    angles = np.arange(direction_count, dtype=np.float32) * (2.0 * np.pi / direction_count)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    approach_end = min(length - 8, 4)
    descend_end = min(length - 8, approach_end + max(6, length // 4))
    close_end = min(length - 4, descend_end + max(4, length // 12))
    push_start = close_end
    split = push_start + max(1, (length - push_start) // 2)
    candidates: list[np.ndarray] = [np.zeros((length, 5), dtype=np.float32)]
    for approach in directions:
        for push_a in directions:
            for push_b in directions:
                for gain in (0.5, 0.25):
                    sequence = np.zeros((length, 5), dtype=np.float32)
                    sequence[:approach_end, :2] = approach
                    sequence[approach_end:descend_end, 2] = -1.0
                    sequence[descend_end:close_end, 4] = 1.0
                    sequence[push_start:split, :2] = gain * push_a
                    sequence[split:, :2] = gain * push_b
                    candidates.append(sequence)
    return np.stack(candidates, axis=0)


def _goal_conditioned_push_bank(goal: np.ndarray, length: int = 40) -> np.ndarray:
    """Build a privileged bank whose push directions are centered on the goal.

    The goal is used only by the evaluator-side reachability ceiling.  The
    approach directions remain a fixed eight-way set, while the push phase
    includes the goal ray and declared angular offsets.  This isolates
    directional discretization from learned ranking and is never a deployable
    RGB policy by itself.
    """
    goal = np.asarray(goal, dtype=np.float64).reshape(-1)
    if goal.size < 2 or not np.isfinite(goal[:2]).all() or length < 20:
        raise ValueError("goal must contain finite XY and length must be >=20")
    norm = float(np.linalg.norm(goal[:2]))
    if norm <= 1.0e-8:
        raise ValueError("goal direction must be nonzero")
    goal_angle = float(np.arctan2(goal[1], goal[0]))
    offsets = np.asarray([0.0, np.pi / 8.0, -np.pi / 8.0, np.pi / 4.0,
                          -np.pi / 4.0, np.pi / 2.0, -np.pi / 2.0, np.pi])
    push_angles = goal_angle + offsets
    approach_angles = np.arange(8, dtype=np.float64) * (2.0 * np.pi / 8.0)
    approach = np.stack([np.cos(approach_angles), np.sin(approach_angles)], axis=1)
    push = np.stack([np.cos(push_angles), np.sin(push_angles)], axis=1)
    approach_end = min(length - 8, 4)
    descend_end = min(length - 8, approach_end + max(6, length // 4))
    close_end = min(length - 2, descend_end + max(4, length // 12))
    candidates: list[np.ndarray] = [np.zeros((length, 5), dtype=np.float32)]
    for approach_direction in approach:
        for push_direction in push:
            for gain in (1.0, 0.5, 0.25):
                sequence = np.zeros((length, 5), dtype=np.float32)
                sequence[:approach_end, :2] = approach_direction
                sequence[approach_end:descend_end, 2] = -1.0
                sequence[descend_end:close_end, 4] = 1.0
                sequence[close_end:, :2] = float(gain) * push_direction
                candidates.append(sequence)
    return np.stack(candidates, axis=0)


def _reset_from_initial(env: gym.Env, initial: dict[str, Any]):
    return env.reset(
        seed=int(initial["seed"]),
        options={
            "task_id": int(initial["task_id"]),
            "render_goal": True,
            "variation_values": copy.deepcopy(initial["variation_values"]),
            "state": np.asarray(initial["state"], dtype=np.float64),
        },
    )


def _capture_rgb_decision_state(
    initial: dict[str, Any],
    pixel_action_mapping: np.ndarray | None,
    descent_steps: int,
    approach_steps: int,
    approach_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the RGB prelude once and freeze its exact simulator decision state."""
    env = _make_rgb_env()
    try:
        obs, _info = _reset_from_initial(env, initial)
        terminated = False
        truncated = False
        for _ in range(int(descent_steps)):
            action = np.zeros(5, dtype=np.float32)
            action[2] = -1.0
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        for _ in range(int(approach_steps)):
            if terminated or truncated:
                break
            if approach_model is None:
                action = _rgb_approach_action(np.asarray(obs, dtype=np.uint8), pixel_action_mapping)
            else:
                from experiments.run_cube_rgb_visual_servo import (
                    _action_from_world_error,
                    _predict_linear,
                    _relative_approach_features,
                )
                from experiments.audit_cube_rgb_observability import (
                    _largest_effector_centroid,
                    _target_red_centroid,
                )
                image = np.asarray(obs, dtype=np.uint8)
                effector = _largest_effector_centroid(image)[:2]
                target = _target_red_centroid(image)[:2]
                goal = _target_red_centroid(
                    np.asarray(env.unwrapped._cur_goal_rendered, dtype=np.uint8),
                )[:2]
                feature = _relative_approach_features(
                    effector, target, goal,
                    family=str(approach_model["approach_feature_family"]),
                )
                world_error = _predict_linear(
                    feature, approach_model["relative_approach_chart"],
                )
                action = np.zeros(5, dtype=np.float32)
                action[:2] = _action_from_world_error(
                    world_error, approach_model["world_action_mapping_per_step"], 1.0,
                )
            obs, _reward, terminated, truncated, _info = env.step(action)
        raw = env.unwrapped
        return {
            "seed": int(initial["seed"]),
            "task_id": int(initial["task_id"]),
            "variation_values": copy.deepcopy(raw.variation_space.value),
            "state": np.concatenate([raw.data.qpos.copy(), raw.data.qvel.copy()]).tolist(),
            "goal": list(initial["goal"]),
            "cube_position": _cube_position(_info).tolist(),
            "prelude_terminated": bool(terminated),
            "prelude_truncated": bool(truncated),
        }
    finally:
        env.close()


def _run_candidate(
    env: gym.Env,
    decision_state: dict[str, Any],
    sequence: np.ndarray,
    goal_tolerance_m: float,
    macro_cycles: int,
) -> dict[str, Any]:
    """Replay one candidate from a frozen RGB-prelude decision state."""
    _obs, info = _reset_from_initial(env, decision_state)
    start_cube = _cube_position(info).copy()
    start_effector = _effector_position(info).copy()
    contacts = [_contact(info)]
    terminated = False
    truncated = False
    cycle_goal_distances: list[float] = []
    for _cycle in range(int(macro_cycles)):
        for action in np.asarray(sequence, dtype=np.float32):
            if terminated or truncated:
                break
            if action.shape != (5,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.0 + 1e-7):
                raise ValueError("candidate action must be a finite normalized 5-vector")
            _obs, _reward, terminated, truncated, info = env.step(action)
            contacts.append(_contact(info))
        cycle_goal_distances.append(float(np.linalg.norm(
            _cube_position(info) - np.asarray(decision_state["goal"], dtype=np.float64)
        )))
        if terminated or truncated:
            break
    final_cube = _cube_position(info).copy()
    final_effector = _effector_position(info).copy()
    goal = np.asarray(decision_state["goal"], dtype=np.float64)
    last_distance = float(np.linalg.norm(final_cube - goal))
    distance = float(min(cycle_goal_distances)) if cycle_goal_distances else last_distance
    best_cycle = int(np.argmin(cycle_goal_distances)) + 1 if cycle_goal_distances else 0
    summary = {
        "final_goal_distance_m": distance,
        "last_goal_distance_m": last_distance,
        "best_cycle": best_cycle,
        "cycle_goal_distances_m": cycle_goal_distances,
        "cube_displacement_m": float(np.linalg.norm(final_cube - start_cube)),
        "effector_displacement_m": float(np.linalg.norm(final_effector - start_effector)),
        "contact_max": float(np.max(contacts)),
        "contact_steps": int(np.sum(np.asarray(contacts) > 1e-5)),
        "success": bool(distance <= float(goal_tolerance_m)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    return summary


def _run_candidate_adaptive_rgb(
    env: gym.Env,
    decision_state: dict[str, Any],
    sequence: np.ndarray,
    goal_tolerance_m: float,
    macro_cycles: int,
    *,
    push_feedback_threshold: float = 1.0e-3,
    reclose_each_cycle: bool = False,
    push_backtrack: bool = True,
) -> dict[str, Any]:
    """Replay one candidate with the online RGB feedback law.

    This is an evaluator ceiling for the *action protocol*, not a learned
    policy.  The control path reads only rendered RGB images and reproduces
    the sign-reversal/gain-backtrack rule used by ``lowgain_backtrack``.  The
    privileged goal distance is read only after each replay for scoring.
    """
    obs, info = _reset_from_initial(env, decision_state)
    start_cube = _cube_position(info).copy()
    goal_image = np.asarray(env.unwrapped._cur_goal_rendered, dtype=np.uint8).copy()
    terminated = False
    truncated = False
    cycle_goal_distances: list[float] = []
    rgb_regressions = 0
    push_backtracks = 0
    close_indices = np.flatnonzero(np.asarray(sequence)[:, 4] > 0.5)
    push_start = int(close_indices[-1]) + 1 if len(close_indices) else 0
    for cycle in range(int(macro_cycles)):
        push_gain = 1.0
        push_sign = 1.0
        cycle_before, before_valid = _rgb_block_goal_error(
            np.asarray(obs, dtype=np.uint8), goal_image,
        )
        previous_push_error: float | None = None
        action_start = 0 if cycle == 0 or reclose_each_cycle else push_start
        for action_index, action in enumerate(np.asarray(sequence, dtype=np.float32)):
            if terminated or truncated:
                break
            if action_index < action_start:
                continue
            runtime_action = np.asarray(action, dtype=np.float32).copy()
            if action_index >= push_start and np.linalg.norm(runtime_action[:2]) > 1.0e-8:
                current_error, current_valid = _rgb_block_goal_error(
                    np.asarray(obs, dtype=np.uint8), goal_image,
                )
                if current_valid and previous_push_error is not None and (
                    current_error > previous_push_error + float(push_feedback_threshold)
                ):
                    push_sign *= -1.0
                    rgb_regressions += 1
                    if push_backtrack:
                        push_gain = max(0.25, 0.5 * push_gain)
                        push_backtracks += 1
                runtime_action[:2] *= float(push_sign * push_gain)
                if current_valid:
                    previous_push_error = current_error
            obs, _reward, terminated, truncated, info = env.step(runtime_action)
            if action_index >= push_start and not (terminated or truncated):
                next_error, next_valid = _rgb_block_goal_error(
                    np.asarray(obs, dtype=np.uint8), goal_image,
                )
                if next_valid:
                    previous_push_error = next_error
        if terminated or truncated:
            break
        _cube = _cube_position(info)
        cycle_goal_distances.append(float(np.linalg.norm(
            _cube - np.asarray(decision_state["goal"], dtype=np.float64),
        )))
        cycle_after, after_valid = _rgb_block_goal_error(
            np.asarray(obs, dtype=np.uint8), goal_image,
        )
        if before_valid and after_valid and cycle_after > cycle_before + float(push_feedback_threshold):
            # Match the online fail-closed cycle gate used by the backtrack arm.
            break
    final_cube = _cube_position(info).copy()
    goal = np.asarray(decision_state["goal"], dtype=np.float64)
    distances = np.asarray(cycle_goal_distances, dtype=np.float64)
    distance = float(np.min(distances)) if len(distances) else float(np.linalg.norm(final_cube - goal))
    last_distance = float(np.linalg.norm(final_cube - goal))
    summary = {
        "final_goal_distance_m": distance,
        "last_goal_distance_m": last_distance,
        "best_cycle": int(np.argmin(distances)) + 1 if len(distances) else 0,
        "cycle_goal_distances_m": cycle_goal_distances,
        "cube_displacement_m": float(np.linalg.norm(final_cube - start_cube)),
        "contact_max": float(_contact(info)),
        "success": bool(distance <= float(goal_tolerance_m)),
        "rgb_regressions": int(rgb_regressions),
        "push_backtracks": int(push_backtracks),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    return summary


def _adaptive_oracle_for_bank(
    decision_state: dict[str, Any], bank: np.ndarray, goal_tolerance_m: float,
    macro_cycles: int,
) -> dict[str, Any]:
    """Compute a privileged ceiling for the exact RGB-backtrack protocol."""
    env = _make_rgb_env()
    try:
        candidates = [
            _run_candidate_adaptive_rgb(
                env, decision_state, sequence, goal_tolerance_m, macro_cycles,
            )
            for sequence in bank
        ]
    finally:
        env.close()
    distances = np.asarray([row["final_goal_distance_m"] for row in candidates], dtype=np.float64)
    selected = int(np.argmin(distances))
    return {
        "candidate_count": int(len(candidates)),
        "oracle_index": selected,
        "oracle": candidates[selected],
        "reachable": bool(any(row["success"] for row in candidates)),
        "success_count": int(sum(row["success"] for row in candidates)),
        "candidate_distances_m": distances.tolist(),
    }


def _oracle_for_bank(
    decision_state: dict[str, Any], bank: np.ndarray, goal_tolerance_m: float,
    macro_cycles: int,
) -> dict[str, Any]:
    env = _make_env()
    try:
        candidates = [
            _run_candidate(
                env, decision_state, sequence, goal_tolerance_m, macro_cycles,
            )
            for sequence in bank
        ]
    finally:
        env.close()
    distances = np.asarray([row["final_goal_distance_m"] for row in candidates], dtype=np.float64)
    selected = int(np.argmin(distances))
    return {
        "candidate_count": int(len(candidates)),
        "oracle_index": selected,
        "oracle": candidates[selected],
        "reachable": bool(any(row["success"] for row in candidates)),
        "success_count": int(sum(row["success"] for row in candidates)),
        "candidate_distances_m": distances.tolist(),
        "candidate_contact_max": [float(row["contact_max"]) for row in candidates],
    }


def _paired_root_record(
    initial: dict[str, Any], fixed_bank: np.ndarray, expanded_bank: np.ndarray,
    goal_tolerance_m: float, descent_steps: int, approach_steps: int,
    pixel_action_mapping: np.ndarray | None, macro_cycles: int,
) -> dict[str, Any]:
    print("  stage=rgb_prelude", flush=True)
    decision_state = _capture_rgb_decision_state(
        initial, pixel_action_mapping, descent_steps, approach_steps,
    )
    # Goal-conditioned candidates are evaluator-only.  Keep the fixed and
    # factorized banks in the same record so the support source is explicit.
    root_expanded_bank = np.concatenate([
        expanded_bank,
        _goal_conditioned_push_bank(
            np.asarray(decision_state["goal"], dtype=np.float64)
            - np.asarray(decision_state["cube_position"], dtype=np.float64),
            length=int(expanded_bank.shape[1]),
        )[1:],
    ], axis=0)
    factorized_bank = _factorized_bank(length=int(expanded_bank.shape[1]))
    multiphase_bank = _multiphase_factorized_bank(length=int(expanded_bank.shape[1]))
    print(f"  stage=privileged_fixed candidates={len(fixed_bank)}", flush=True)
    fixed = _oracle_for_bank(
        decision_state, fixed_bank, goal_tolerance_m, macro_cycles,
    )
    print(f"  stage=privileged_expanded candidates={len(root_expanded_bank)}", flush=True)
    expanded = _oracle_for_bank(
        decision_state, root_expanded_bank, goal_tolerance_m, macro_cycles,
    )
    print(f"  stage=privileged_factorized candidates={len(factorized_bank)}", flush=True)
    factorized = _oracle_for_bank(
        decision_state, factorized_bank, goal_tolerance_m, macro_cycles,
    )
    print(f"  stage=privileged_multiphase candidates={len(multiphase_bank)}", flush=True)
    multiphase = _oracle_for_bank(
        decision_state, multiphase_bank, goal_tolerance_m, macro_cycles,
    )
    print(f"  stage=rgb_adaptive_fixed candidates={len(fixed_bank)}", flush=True)
    adaptive_fixed = _adaptive_oracle_for_bank(
        decision_state, fixed_bank, goal_tolerance_m, macro_cycles,
    )
    print(f"  stage=rgb_adaptive_expanded candidates={len(root_expanded_bank)}", flush=True)
    adaptive_expanded = _adaptive_oracle_for_bank(
        decision_state, root_expanded_bank, goal_tolerance_m, macro_cycles,
    )
    return {
        "seed": int(initial["seed"]),
        "task_id": int(initial["task_id"]),
        "fixed": fixed,
        "expanded": expanded,
        "factorized": factorized,
        "multiphase": multiphase,
        "factorized_bank_candidate_count": int(factorized_bank.shape[0]),
        "expanded_bank_candidate_count": int(root_expanded_bank.shape[0]),
        "adaptive_fixed": adaptive_fixed,
        "adaptive_expanded": adaptive_expanded,
        "expanded_distance_improvement_m": float(
            fixed["oracle"]["final_goal_distance_m"] - expanded["oracle"]["final_goal_distance_m"]
        ),
        "reachability_recovered_by_expansion": bool(
            not fixed["reachable"] and expanded["reachable"]
        ),
        "reachability_recovered_by_factorized": bool(
            not fixed["reachable"] and factorized["reachable"]
        ),
        "reachability_recovered_by_multiphase": bool(
            not fixed["reachable"] and multiphase["reachable"]
        ),
        "adaptive_reachability_recovered_by_expansion": bool(
            not adaptive_fixed["reachable"] and adaptive_expanded["reachable"]
        ),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")
    fixed = [row["fixed"] for row in records]
    expanded = [row["expanded"] for row in records]
    summary = {
        "roots": len(records),
        "fixed_bank_reachability_rate": float(np.mean([row["reachable"] for row in fixed])),
        "expanded_bank_reachability_rate": float(np.mean([row["reachable"] for row in expanded])),
        "fixed_bank_oracle_distance_mean_m": float(np.mean([
            row["oracle"]["final_goal_distance_m"] for row in fixed
        ])),
        "expanded_bank_oracle_distance_mean_m": float(np.mean([
            row["oracle"]["final_goal_distance_m"] for row in expanded
        ])),
        "reachability_recovered_by_expansion_count": int(sum(
            row["reachability_recovered_by_expansion"] for row in records
        )),
        "claim_boundary": "privileged candidate-bank ceiling only; not RGB online planning, learned dynamics evidence, or a transfer certificate",
    }
    if all("factorized" in row for row in records):
        factorized = [row["factorized"] for row in records]
        summary.update({
            "factorized_bank_reachability_rate": float(np.mean([
                row["reachable"] for row in factorized
            ])),
            "factorized_bank_oracle_distance_mean_m": float(np.mean([
                row["oracle"]["final_goal_distance_m"] for row in factorized
            ])),
            "reachability_recovered_by_factorized_count": int(sum(
                row.get("reachability_recovered_by_factorized", False)
                for row in records
            )),
        })
    if all("multiphase" in row for row in records):
        multiphase = [row["multiphase"] for row in records]
        summary.update({
            "multiphase_bank_reachability_rate": float(np.mean([
                row["reachable"] for row in multiphase
            ])),
            "multiphase_bank_oracle_distance_mean_m": float(np.mean([
                row["oracle"]["final_goal_distance_m"] for row in multiphase
            ])),
            "reachability_recovered_by_multiphase_count": int(sum(
                row.get("reachability_recovered_by_multiphase", False)
                for row in records
            )),
        })
    if all("adaptive_fixed" in row and "adaptive_expanded" in row for row in records):
        summary.update({
            "adaptive_fixed_bank_reachability_rate": float(np.mean([
                row["adaptive_fixed"]["reachable"] for row in records
            ])),
            "adaptive_expanded_bank_reachability_rate": float(np.mean([
                row["adaptive_expanded"]["reachable"] for row in records
            ])),
            "adaptive_reachability_recovered_by_expansion_count": int(sum(
                row["adaptive_reachability_recovered_by_expansion"] for row in records
            )),
        })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261900)
    parser.add_argument("--test-roots", type=int, default=8)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--macro-length", type=int, default=20)
    parser.add_argument("--expanded-macro-length", type=int, default=40)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.05)
    parser.add_argument("--descent-steps", type=int, default=15)
    parser.add_argument("--approach-steps", type=int, default=8)
    parser.add_argument("--approach-calibration-roots", type=int, default=8)
    parser.add_argument("--probe-steps", type=int, default=4)
    parser.add_argument("--macro-cycles", type=int, default=5)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from a partial JSON checkpoint at --output",
    )
    args = parser.parse_args()
    if args.test_roots < 1 or args.macro_length < 20 or args.expanded_macro_length < 20:
        raise ValueError("test-roots must be positive and macro lengths must be at least 20")
    if not np.isfinite(args.goal_tolerance_m) or args.goal_tolerance_m <= 0.0:
        raise ValueError("goal-tolerance-m must be positive and finite")
    if args.descent_steps < 0 or args.approach_steps < 0:
        raise ValueError("descent-steps and approach-steps must be non-negative")
    if args.approach_calibration_roots < 1 or args.probe_steps < 1:
        raise ValueError("approach calibration roots and probe-steps must be positive")
    if args.macro_cycles < 1:
        raise ValueError("macro-cycles must be positive")
    pixel_action_mapping, probe_diagnostics = _fit_rgb_pixel_action_mapping(
        seed=args.seed,
        tasks=[int(task) for task in args.tasks],
        roots=args.approach_calibration_roots,
        descent_steps=args.descent_steps,
        probe_steps=args.probe_steps,
    )
    fixed_bank = _online_macro_bank(length=args.macro_length, push_scale=0.25)
    expanded_bank = _expanded_bank(length=args.expanded_macro_length)
    print(f"[purpose] privileged Cube candidate-bank ceiling; fixed={len(fixed_bank)} expanded={len(expanded_bank)} roots={args.test_roots}")
    records: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        partial = json.loads(args.output.read_text())
        if partial.get("schema") != "cube_planner_ceiling.v2.partial":
            raise ValueError("--resume requires a matching partial ceiling checkpoint")
        records = list(partial.get("records", []))
        print(f"[resume] loaded completed roots={len(records)}", flush=True)
    start_index = len(records)
    for index in range(start_index, args.test_roots):
        task_id = int(args.tasks[index % len(args.tasks)])
        initial = _capture_initial_condition(args.seed + 1000 + index, task_id)
        record = _paired_root_record(
            initial, fixed_bank, expanded_bank, args.goal_tolerance_m,
            args.descent_steps, args.approach_steps, pixel_action_mapping,
            args.macro_cycles,
        )
        records.append(record)
        print(f"root={index} task={task_id} fixed={record['fixed']['oracle']['final_goal_distance_m']:.6g} expanded={record['expanded']['oracle']['final_goal_distance_m']:.6g} fixed_reachable={record['fixed']['reachable']} expanded_reachable={record['expanded']['reachable']}", flush=True)
        # Persist after every root so a long host-side run can be resumed
        # without silently discarding already completed paired evaluations.
        partial = {
            "schema": "cube_planner_ceiling.v2.partial",
            "protocol": {
                "environment": "swm/OGBCube-v0",
                "root_disjoint": True,
                "seed": int(args.seed),
                "test_roots": int(args.test_roots),
                "macro_length": int(args.macro_length),
                "expanded_macro_length": int(args.expanded_macro_length),
                "macro_cycles": int(args.macro_cycles),
            },
            "records": records,
        }
        tmp = args.output.with_suffix(args.output.suffix + ".partial.tmp")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(partial, indent=2) + "\n")
        tmp.replace(args.output)
    payload = {
        "schema": "cube_planner_ceiling.v2",
        "protocol": {
            "environment": "swm/OGBCube-v0",
            "root_disjoint": True,
            "candidate_selection": "privileged post-hoc final goal distance",
            "runtime_policy_input": "none; evaluator-only oracle",
            "fixed_bank": "online four-direction bank with push_scale=0.25",
            "expanded_bank": (
                "eight-direction bank plus low-gain variants and factorized "
                "approach_push direction pairs plus root-local goal-direction offsets"
            ),
            "factorized_bank": "independent approach/push direction pairs with three push gains",
            "multiphase_bank": "independent approach plus two persistent push directions with two gains",
            "expanded_macro_length": int(args.expanded_macro_length),
            "prelude": "same RGB Jacobian descent/approach protocol as panel",
            "descent_steps": int(args.descent_steps),
            "approach_steps": int(args.approach_steps),
            "approach_calibration_roots": int(args.approach_calibration_roots),
            "probe_steps": int(args.probe_steps),
            "rgb_probe_diagnostics": probe_diagnostics,
            "macro_cycles": int(args.macro_cycles),
            "cycle_protocol": "repeat the same full macro and let privileged oracle choose best stop cycle",
            "goal_tolerance_m": float(args.goal_tolerance_m),
            "claim_boundary": "privileged candidate reachability ceiling only",
        },
        "records": records,
        "aggregate": _summarize(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
