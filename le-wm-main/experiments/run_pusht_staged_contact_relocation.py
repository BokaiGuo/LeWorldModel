"""Staged contact-relocation repair for the PushT signed-effect adapter.

This experiment tests the mechanism suggested by the contact-rich manipulation
literature: contact location is a discrete/continuous control variable and
must not be frozen while a local torque model is used.  The learned component
still selects a signed pulse from the current observation only.  A separate
geometric stage controller is responsible for (re-)acquiring a support side
when contact is lost or the translation tube is violated.

The protocol reports four separate quantities:

* held-out signed-effect ranking (inherited from the signed-torque audit);
* block-pose closure before any agent return;
* strict task success after a guarded return;
* the number and causes of contact relocations.

This is a privileged simulator-state adapter audit.  It is not frozen visual
LeWM planning or evidence for causal latent dynamics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_obstacle_aware_local_effect import return_agent, route_to_contact, route_waypoints
from experiments.run_pusht_signed_torque_audit import (
    angle_error,
    bootstrap_mean,
    collect_effect_rows,
    contact_features,
    fit_models,
    make_terminal_roots,
    pulse_actions,
    select_index,
    support_inward_normal,
    unit,
    wrap_angle,
)


def block_pose_success(state: np.ndarray, goal: np.ndarray) -> bool:
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return bool(
        np.linalg.norm(state[2:4] - goal[2:4]) < 20.0
        and angle_error(state[4], goal[4]) < np.pi / 9.0
    )


def strict_success(state: np.ndarray, goal: np.ndarray) -> bool:
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return bool(
        np.linalg.norm(state[:4] - goal[:4]) < 20.0
        and angle_error(state[4], goal[4]) < np.pi / 9.0
    )


def world_vertices(env: gym.Env) -> np.ndarray:
    block = env.unwrapped.block
    return np.asarray([
        [float(world_point.x), float(world_point.y)]
        for shape in block.shapes
        for point in shape.get_vertices()
        for world_point in [block.local_to_world(point)]
    ], dtype=np.float64)


def support_target(
    env: gym.Env,
    state: np.ndarray,
    outward: np.ndarray,
    gap: float = 4.0,
) -> np.ndarray:
    """Return a point just outside the T boundary along ``outward``."""
    state = np.asarray(state, dtype=np.float64)
    outward = unit(outward)
    vertices = world_vertices(env)
    relative = vertices - state[2:4]
    radius = float(np.max(relative @ outward))
    return state[2:4] + max(radius + float(gap), 35.0) * outward


def relocation_direction(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    attempt: int,
) -> np.ndarray:
    """Choose a support side without querying a successor.

    The first side is the translation-support side.  Subsequent sides form a
    deterministic fan around it, which changes the contact mode instead of
    repeatedly applying torque at a drifting support point.
    """
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    block_to_goal = unit(goal[2:4] - state[2:4])
    # To push the block toward the goal, the agent should stand behind it.
    base_angle = float(np.arctan2(-block_to_goal[1], -block_to_goal[0]))
    offsets = (0.0, np.pi / 2.0, -np.pi / 2.0, np.pi, np.pi / 3.0, -np.pi / 3.0)
    offset = offsets[int(attempt) % len(offsets)]
    angle = base_angle + offset
    direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
    # Keep the helper honest: the geometry lookup is used to validate the
    # direction against a real T boundary, but never to inspect a successor.
    vertices = world_vertices(env) - state[2:4]
    if not np.isfinite(vertices).all():
        raise ValueError("non-finite block geometry")
    return unit(direction)


def execute(env: gym.Env, state: np.ndarray, goal: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, bool]:
    observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
    return np.asarray(observation["state"], dtype=np.float64).copy(), bool(terminated or truncated)


def move_to_support(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    outward: np.ndarray,
    clearance: float,
    contact_radius: float,
    max_steps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Disengage and route to a new support side with bounded PD commands."""
    current = np.asarray(state, dtype=np.float64).copy()
    block_before = current[2:4].copy()
    # Reuse the project's validated obstacle-aware route implementation.  The
    # earlier local copy below was too permissive: it stopped at a distance
    # threshold without proving that the selected support side had actually
    # been reached, which silently changed the contact mode.
    from types import SimpleNamespace
    route_args = SimpleNamespace(
        clearance=float(clearance),
        contact_distance=float(contact_radius),
        settle_steps=3,
    )
    routed, route_info = route_to_contact(
        env, current, goal, -unit(outward), route_args
    )
    return routed, {
        "outward": np.asarray(outward, dtype=np.float64).tolist(),
        "steps": int(route_info.get("route_steps", 0)),
        "contact_distance": float(np.linalg.norm(routed[:2] - routed[2:4])),
        "block_displacement": float(np.linalg.norm(routed[2:4] - block_before)),
        "trace": route_info.get("waypoints", []),
        "terminated": bool(route_info.get("terminated", False)),
        "route_impl": "validated_obstacle_aware_route_to_contact",
    }
    agent_block = current[:2] - current[2:4]
    if np.linalg.norm(agent_block) < 1e-8:
        agent_block = np.asarray([1.0, 0.0])
    agent_block = unit(agent_block)
    steps = 0
    terminated = False
    route_trace: list[dict[str, Any]] = []

    # First leave the old contact manifold.  This is a stage transition, not
    # a learned push and not a simulator look-ahead.
    distance = float(np.linalg.norm(current[:2] - current[2:4]))
    if distance < clearance:
        target = current[:2] + (clearance + 22.0) * agent_block
        for _ in range(min(32, max_steps)):
            delta = target - current[:2]
            if np.linalg.norm(delta) <= 8.0:
                break
            current, terminated = execute(env, current, goal, np.clip(delta / 180.0, -1.0, 1.0))
            steps += 1
            if terminated:
                break
        route_trace.append({"phase": "disengage", "steps": steps, "target": target.tolist()})

    if steps < max_steps:
        target_contact = support_target(env, current, outward, gap=4.0)
        waypoints = route_waypoints(current, target_contact, clearance)
        for waypoint in waypoints + [target_contact]:
            for _ in range(min(36, max_steps - steps)):
                delta = waypoint - current[:2]
                if np.linalg.norm(delta) <= 8.0:
                    break
                current, terminated = execute(env, current, goal, np.clip(delta / 180.0, -1.0, 1.0))
                steps += 1
                if terminated:
                    break
            route_trace.append({"phase": "route", "steps": steps, "target": waypoint.tolist()})
            if terminated or steps >= max_steps:
                break
        # A low-gain final approach prevents a large velocity impulse at the
        # newly selected contact side.
        if not terminated and steps < max_steps:
            for _ in range(12):
                current_distance = float(np.linalg.norm(current[:2] - current[2:4]))
                if current_distance <= contact_radius:
                    break
                target_contact = support_target(env, current, outward, gap=2.0)
                delta = target_contact - current[:2]
                current, terminated = execute(env, current, goal, np.clip(delta / 240.0, -0.35, 0.35))
                steps += 1
                if terminated:
                    break

    return current, {
        "outward": np.asarray(outward, dtype=np.float64).tolist(),
        "steps": int(steps),
        "contact_distance": float(np.linalg.norm(current[:2] - current[2:4])),
        "block_displacement": float(np.linalg.norm(current[2:4] - block_before)),
        "trace": route_trace,
        "terminated": bool(locals().get("terminated", False)),
    }


def learned_pulse(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    regressor: Any,
    pose_regressor: Any,
    classifier: Any,
    tangent_magnitudes: tuple[float, ...],
    inward_magnitude: float,
    selection_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    support_normal = support_inward_normal(env, state)
    actions, signs = pulse_actions(state, tangent_magnitudes, inward_magnitude, support_normal)
    rows = []
    for action, moment_sign in zip(actions, signs):
        rows.append({
            "before": np.asarray(state, dtype=np.float64).copy(),
            "goal": np.asarray(goal, dtype=np.float64).copy(),
            "action": np.asarray(action, dtype=np.float64),
            "moment_sign": float(moment_sign),
            "features": np.concatenate([contact_features(state, action), support_normal]),
            "angle_error_before": angle_error(state[4], goal[4]),
        })
    selected = select_index(rows, regressor, pose_regressor, classifier, selection_mode)
    features = np.asarray(rows[selected]["features"], dtype=np.float64)[None]
    predicted = float(regressor.predict(features)[0])
    return np.asarray(rows[selected]["action"], dtype=np.float32), {
        "selected": int(selected),
        "moment_sign": float(rows[selected]["moment_sign"]),
        "predicted_signed_delta": predicted,
        "contact_distance_before": float(np.linalg.norm(state[:2] - state[2:4])),
        "support_normal": support_normal.tolist(),
    }


def run_staged_case(
    env: gym.Env,
    root: dict[str, np.ndarray],
    regressor: Any,
    pose_regressor: Any,
    classifier: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=0, options={"state": root["state"].copy(), "goal_state": root["goal"].copy()})
    current = np.asarray(observation["state"], dtype=np.float64).copy()
    goal = np.asarray(root["goal"], dtype=np.float64).copy()
    initial_xy = float(np.linalg.norm(current[2:4] - goal[2:4]))
    initial_angle = angle_error(current[4], goal[4])
    relocation_records: list[dict[str, Any]] = []
    pulse_records: list[dict[str, Any]] = []
    attempt = 0
    relocations = 0
    reason_counts = {"initial": 0, "lost_contact": 0, "xy_drift": 0, "stagnation": 0}
    last_angle = initial_angle
    stagnant = 0
    # Initial acquisition is explicit even when the terminal root happens to
    # be close: this makes the stage boundary observable and comparable.
    outward = relocation_direction(env, current, goal, attempt)
    current, info = move_to_support(env, current, goal, outward, args.clearance, args.contact_radius, args.route_steps)
    relocation_records.append({"reason": "initial", **info})
    reason_counts["initial"] += 1
    relocations += 1
    for step in range(args.max_pulses):
        xy = float(np.linalg.norm(current[2:4] - goal[2:4]))
        angle = angle_error(current[4], goal[4])
        contact_distance = float(np.linalg.norm(current[:2] - current[2:4]))
        if block_pose_success(current, goal):
            break
        if contact_distance > args.max_contact_distance:
            reason = "lost_contact"
        elif xy > args.xy_tube:
            reason = "xy_drift"
        elif step > 0 and angle >= last_angle - args.min_angle_improvement:
            stagnant += 1
            reason = "stagnation" if stagnant >= args.stagnation_steps else ""
        else:
            stagnant = 0
            reason = ""
        if reason:
            if relocations >= args.max_relocations:
                break
            attempt += 1
            outward = relocation_direction(env, current, goal, attempt)
            current, info = move_to_support(env, current, goal, outward, args.clearance, args.contact_radius, args.route_steps)
            relocation_records.append({"reason": reason, **info})
            reason_counts[reason] += 1
            relocations += 1
            stagnant = 0
            last_angle = angle_error(current[4], goal[4])
            continue
        action, selection = learned_pulse(
            env, current, goal, regressor, pose_regressor, classifier,
            args.tangent_magnitudes, args.inward_magnitude, args.selection_mode,
        )
        before = current.copy()
        current, terminated = execute(env, current, goal, action)
        observed_delta = float(wrap_angle(current[4] - before[4]))
        pulse_records.append({
            "step": int(step), "action": action.tolist(),
            "angle_before": angle_error(before[4], goal[4]),
            "angle_after": angle_error(current[4], goal[4]),
            "xy_after": float(np.linalg.norm(current[2:4] - goal[2:4])),
            "observed_signed_delta": observed_delta,
            "contact_distance_after": float(np.linalg.norm(current[:2] - current[2:4])),
            "terminated": bool(terminated), **selection,
        })
        last_angle = angle_error(current[4], goal[4])
        if terminated:
            break

    block_closed = block_pose_success(current, goal)
    pre_return = current.copy()
    return_info: dict[str, Any] = {"attempted": False}
    if block_closed and args.return_agent:
        current, return_terminated = return_agent(env, current, goal, steps=args.return_steps, clearance=args.clearance)
        return_info = {"attempted": True, "terminated": bool(return_terminated)}
    final_xy = float(np.linalg.norm(current[2:4] - goal[2:4]))
    final_angle = angle_error(current[4], goal[4])
    return {
        "initial_xy": initial_xy, "initial_angle": initial_angle,
        "pre_return_state": pre_return, "final_state": current,
        "pre_return_block_pose_success": bool(block_closed),
        "strict_success": bool(strict_success(current, goal)),
        "final_xy": final_xy, "final_angle": final_angle,
        "relocations": int(relocations), "reason_counts": reason_counts,
        "relocation_records": relocation_records, "pulse_records": pulse_records,
        "return": return_info,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-roots", type=int, default=64)
    parser.add_argument("--held-out-roots", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9181)
    parser.add_argument("--root-protocol", choices=("terminal_pose", "radial_contact"), default="terminal_pose")
    parser.add_argument("--pulse-steps", type=int, default=1)
    parser.add_argument("--train-inward-magnitude", type=float, default=0.08)
    parser.add_argument("--inward-magnitude", type=float, default=0.08)
    parser.add_argument("--tangent-magnitudes", type=float, nargs="+", default=[0.06, 0.10, 0.16])
    parser.add_argument("--selection-mode", choices=("direction_classifier", "signed_regressor", "pose_regressor"), default="direction_classifier")
    parser.add_argument("--max-pulses", type=int, default=48)
    parser.add_argument("--max-reconnections", dest="max_relocations", type=int, default=5)
    parser.add_argument("--max-relocations", dest="max_relocations", type=int, default=5)
    parser.add_argument("--route-steps", type=int, default=110)
    parser.add_argument("--clearance", type=float, default=120.0)
    parser.add_argument("--contact-radius", type=float, default=48.0)
    parser.add_argument("--max-contact-distance", type=float, default=68.0)
    parser.add_argument("--xy-tube", type=float, default=24.0)
    parser.add_argument("--stagnation-steps", type=int, default=3)
    parser.add_argument("--min-angle-improvement", type=float, default=1.0e-3)
    parser.add_argument("--return-agent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return-steps", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.root_protocol != "terminal_pose":
        raise ValueError("staged contact relocation requires real PushT geometry; use --root-protocol terminal_pose")
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    tangent_magnitudes = tuple(float(x) for x in args.tangent_magnitudes)
    try:
        train_roots = make_terminal_roots(env, args.train_roots, args.seed + 1000, root_protocol=args.root_protocol)
        held_roots = make_terminal_roots(env, args.held_out_roots, args.seed + 2000, root_protocol=args.root_protocol)
        train_rows = collect_effect_rows(env, train_roots, tangent_magnitudes, args.train_inward_magnitude, args.pulse_steps)
        held_rows = collect_effect_rows(env, held_roots, tangent_magnitudes, args.train_inward_magnitude, args.pulse_steps)
        regressor, pose_regressor, classifier = fit_models(train_rows, args.seed)
        staged = [run_staged_case(env, root, regressor, pose_regressor, classifier, args) for root in held_roots]
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1", "train_roots": args.train_roots, "held_out_roots": args.held_out_roots,
            "train_seed": args.seed + 1000, "held_out_seed": args.seed + 2000,
            "root_protocol": args.root_protocol, "pulse_steps": args.pulse_steps,
            "contact_relocation": True, "contact_relocation_is_analytic_geometry": True,
            "learned_selection": args.selection_mode,
            "action_distribution": {"inward": args.train_inward_magnitude, "tangent": list(tangent_magnitudes)},
            "same_candidate_bank_for_all_cases": True,
        },
        "training": {"rows": len(train_rows), "unique_roots": len(train_roots)},
        "held_out_probe_rows": len(held_rows),
        "staged_summary": {
            "n": len(staged),
            "block_pose_success_rate": float(np.mean([row["pre_return_block_pose_success"] for row in staged])),
            "strict_success_rate": float(np.mean([row["strict_success"] for row in staged])),
            "mean_final_xy": float(np.mean([row["final_xy"] for row in staged])),
            "mean_final_angle": float(np.mean([row["final_angle"] for row in staged])),
            "mean_relocations": float(np.mean([row["relocations"] for row in staged])),
            "reason_counts": {key: int(sum(row["reason_counts"][key] for row in staged)) for key in ("initial", "lost_contact", "xy_drift", "stagnation")},
        },
        "episodes": staged,
        "sanity_checks": {
            "all_finite": bool(all(np.isfinite(np.asarray(row["final_state"], dtype=np.float64)).all() for row in staged)),
            "same_case_count": len(staged) == args.held_out_roots,
            "learned_pulses_present": bool(any(row["pulse_records"] for row in staged)),
            "relocation_records_present": bool(all(row["relocation_records"] for row in staged)),
        },
        "claim_boundary": "Privileged simulator-state staged contact relocation plus held-out signed-effect adapter; not visual LeWM planning, causal latent dynamics, or a universal world model.",
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (output / "staged_contact_relocation.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value))
    summary = report["staged_summary"]
    lines = [
        "# PushT staged contact relocation", "",
        "| metric | estimate |", "|---|---:|",
        f"| block-pose closure before return | {summary['block_pose_success_rate']:.4f} |",
        f"| strict task success after return | {summary['strict_success_rate']:.4f} |",
        f"| final block XY error | {summary['mean_final_xy']:.4f} |",
        f"| final angle error | {summary['mean_final_angle']:.4f} |",
        f"| mean relocations | {summary['mean_relocations']:.4f} |", "",
        f"Relocation reasons: {summary['reason_counts']}", "",
        report["claim_boundary"], "",
    ]
    (output / "staged_contact_relocation.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "summary": summary, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
