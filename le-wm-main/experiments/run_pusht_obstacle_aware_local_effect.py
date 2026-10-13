"""Privileged PushT repair: obstacle-aware contact and local action effects.

The earlier oracle diagnosis exposed a concrete failure mode: a straight-line
approach to ``block - d * desired_direction`` can cross the block and create a
contact from the wrong side.  This experiment tests a minimally sufficient
repair under simulator-state access:

1. route the agent around an expanded block clearance region;
2. probe one-step action effects on the *actual* simulator state;
3. greedily execute the action with the best measured block-pose improvement;
4. return the agent only after block translation/orientation have converged.

This is an oracle planning diagnostic, not a frozen-LeWM or visual-policy
result.  Its purpose is to determine whether the negative result is caused by
contact acquisition/planning rather than by an impossible PushT task.
"""

from __future__ import annotations

import argparse
import json
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

from experiments.run_pusht_repaired_predictor_mpc import stable_contact_bootstrap

gym.logger.min_level = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8421)
    parser.add_argument("--bootstrap-steps", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--probe-directions", type=int, default=24)
    parser.add_argument("--clearance", type=float, default=140.0)
    parser.add_argument("--contact-distance", type=float, default=42.0)
    parser.add_argument("--probe-scale", type=float, default=0.28)
    parser.add_argument("--probe-horizon", type=int, default=1)
    parser.add_argument("--settle-steps", type=int, default=6)
    return parser.parse_args()


def angle_error(value: float, target: float) -> float:
    return float(abs((float(value) - float(target) + np.pi) % (2 * np.pi) - np.pi))


def pose_cost(state: np.ndarray, goal: np.ndarray) -> float:
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return float(np.linalg.norm(state[2:4] - goal[2:4]) + 55.0 * angle_error(state[4], goal[4]))


def block_xy_cost(state: np.ndarray, goal: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(state)[2:4] - np.asarray(goal)[2:4]))


def task_success(state: np.ndarray, goal: np.ndarray) -> bool:
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return bool(
        np.linalg.norm(state[:4] - goal[:4]) < 20.0
        and angle_error(state[4], goal[4]) < np.pi / 9.0
    )


def block_pose_success(state: np.ndarray, goal: np.ndarray) -> bool:
    """Check the block-only part of the strict PushT termination rule."""
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return bool(
        np.linalg.norm(state[2:4] - goal[2:4]) < 20.0
        and angle_error(state[4], goal[4]) < np.pi / 9.0
    )


def reset_step(
    env: gym.Env, state: np.ndarray, goal: np.ndarray, action: np.ndarray
) -> tuple[np.ndarray, bool]:
    env.reset(seed=0, options={"state": np.asarray(state).copy(), "goal_state": np.asarray(goal).copy()})
    observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
    return np.asarray(observation["state"], dtype=np.float64).copy(), bool(terminated or truncated)


def execute(
    env: gym.Env, state: np.ndarray, goal: np.ndarray, action: np.ndarray
) -> tuple[np.ndarray, bool]:
    return reset_step(env, state, goal, action)


def route_waypoints(
    state: np.ndarray,
    target_contact: np.ndarray,
    clearance: float,
) -> list[np.ndarray]:
    """Construct a polyline around the block rather than through it."""
    state = np.asarray(state, dtype=np.float64)
    block = state[2:4]
    agent = state[:2]
    toward = target_contact - block
    toward /= max(np.linalg.norm(toward), 1e-8)
    perpendicular = np.asarray([-toward[1], toward[0]], dtype=np.float64)
    # Two symmetric side routes; choose the shorter route after staying outside
    # the expanded block clearance disk.
    candidates = []
    for sign in (-1.0, 1.0):
        side = block + sign * perpendicular * clearance
        # ``toward`` points from the block to the intended contact side.  The
        # intermediate waypoint must remain on that same side; using ``-``
        # here sends the final segment through the block (the original bug).
        side2 = block + sign * perpendicular * (0.72 * clearance) + toward * (0.55 * clearance)
        candidates.append([side, side2, target_contact])
    def path_length(points: list[np.ndarray]) -> float:
        points = [agent] + points
        return float(sum(np.linalg.norm(points[i + 1] - points[i]) for i in range(len(points) - 1)))
    return min(candidates, key=path_length)


def move_toward(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    target: np.ndarray,
    steps: int,
    contact_distance: float,
    stop_on_contact: bool = False,
) -> tuple[np.ndarray, int, bool, bool]:
    current = np.asarray(state, dtype=np.float64).copy()
    entered_contact = False
    terminated = False
    for step in range(steps):
        delta = np.asarray(target, dtype=np.float64) - current[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= 8.0:
            break
        # Use a smaller target gain during routing so that the PD controller
        # does not carry a large velocity into the contact probe.
        action = np.clip(delta / 180.0, -1.0, 1.0)
        next_state, terminated = execute(env, current, goal, action)
        if np.linalg.norm(next_state[2:4] - current[2:4]) > 1e-3:
            entered_contact = True
        current = next_state
        if terminated or (stop_on_contact and np.linalg.norm(current[2:4] - current[:2]) <= contact_distance):
            # The latter only gates the caller's phase transition; it does not
            # claim that a useful push has occurred.
            if np.linalg.norm(current[2:4] - current[:2]) <= contact_distance:
                entered_contact = True
            break
    return current, step + 1 if 'step' in locals() else 0, entered_contact, terminated


def route_to_contact(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    desired_direction: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    desired_direction = np.asarray(desired_direction, dtype=np.float64)
    desired_direction /= max(np.linalg.norm(desired_direction), 1e-8)
    current = np.asarray(state, dtype=np.float64).copy()
    # Stable-contact bootstraps intentionally start near the object.  A route
    # planned from that state can drag the block before it ever reaches the
    # intended side.  First disengage along the current outward normal.
    outward = current[:2] - current[2:4]
    outward_norm = float(np.linalg.norm(outward))
    if outward_norm < 1e-6:
        outward = np.asarray([1.0, 0.0], dtype=np.float64)
        outward_norm = 1.0
    outward /= outward_norm
    retreat_target = current[:2] + outward * (args.clearance + 35.0)
    current, retreat_steps, retreat_contact, terminated = move_toward(
        env, current, goal, retreat_target, 28, args.contact_distance,
        stop_on_contact=False,
    )
    # Recompute the target using the post-retraction block pose.
    contact = np.asarray(current[2:4], dtype=np.float64) - args.contact_distance * desired_direction
    waypoints = route_waypoints(current, contact, args.clearance)
    used = []
    total_steps = retreat_steps
    used.append({
        "target": retreat_target.copy(),
        "steps": retreat_steps,
        "entered_contact": retreat_contact,
        "phase": "disengage",
    })
    for waypoint in waypoints:
        current, count, entered, terminated = move_toward(
            env, current, goal, waypoint, 40, args.contact_distance,
            stop_on_contact=False,
        )
        used.append({"target": waypoint.copy(), "steps": count, "entered_contact": entered})
        total_steps += count
        if terminated:
            break
    if not terminated:
        # Recompute the final contact point after the side route.  The T shape
        # is not circular, so its pose may change slightly even outside the
        # nominal clearance disk.
        current_contact = current[2:4] - args.contact_distance * desired_direction
        current, count, entered, terminated = move_toward(
            env, current, goal, current_contact, 18, args.contact_distance,
            stop_on_contact=True,
        )
        used.append({"target": current_contact.copy(), "steps": count, "entered_contact": entered, "recomputed": True})
        total_steps += count
    if not terminated:
        for _ in range(args.settle_steps):
            current, terminated = execute(env, current, goal, np.zeros(2, dtype=np.float64))
            total_steps += 1
            if terminated:
                break
    return current, {
        "contact_target": contact,
        "retreat_target": retreat_target,
        "retreat_steps": retreat_steps,
        "waypoints": used,
        "route_steps": total_steps,
        "terminated": terminated,
        "contact_distance": float(np.linalg.norm(current[2:4] - current[:2])),
    }


def probe_actions(args: argparse.Namespace) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, args.probe_directions, endpoint=False)
    actions = np.stack([np.cos(angles), np.sin(angles)], axis=1) * args.probe_scale
    return np.concatenate([actions, np.zeros((1, 2), dtype=np.float64)], axis=0)


def choose_local_effect(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    args: argparse.Namespace,
    phase: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    current = np.asarray(state, dtype=np.float64)
    candidates = probe_actions(args)
    rows = []
    for action in candidates:
        predicted, terminated = reset_step(env, current, goal, action)
        rows.append({
            "action": action.copy(),
            "next_state": predicted.copy(),
            "block_delta": (predicted[2:6] - current[2:6]).copy(),
            "cost": pose_cost(predicted, goal),
            "xy_cost": block_xy_cost(predicted, goal),
            "angle_error": angle_error(predicted[4], goal[4]),
            "terminated": terminated,
        })
    # Use one task-consistent block-pose objective in both phases.  The earlier
    # translation-only score could accept a large orientation regression and
    # leave the controller in an unrecoverable contact mode.
    scores = [row["cost"] for row in rows]
    selected = int(np.argmin(scores))
    chosen = rows[selected]
    return np.asarray(chosen["action"], dtype=np.float64), {
        "phase": phase,
        "n_candidates": len(rows),
        "selected_index": selected,
        "selected": chosen,
        "best_score": float(scores[selected]),
        "max_block_motion": max(float(np.linalg.norm(row["next_state"][2:4] - current[2:4])) for row in rows),
        "probe_rows": rows,
    }


def local_push(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]], bool]:
    current = np.asarray(state, dtype=np.float64).copy()
    records = []
    terminated = False
    best_cost = pose_cost(current, goal)
    stagnant = 0
    for step in range(args.max_steps):
        xy_error = block_xy_cost(current, goal)
        angle = angle_error(current[4], goal[4])
        phase = "translation" if xy_error > 24.0 else "orientation"
        action, probe = choose_local_effect(env, current, goal, args, phase)
        if probe["max_block_motion"] < 0.5:
            # All probes are outside contact.  Re-establish the desired normal
            # explicitly; choosing the least-bad free-space probe caused the
            # previous no-op loop.
            direction = goal[2:4] - current[2:4]
            direction /= max(np.linalg.norm(direction), 1e-8)
            target = current[2:4] - args.contact_distance * direction
            action = np.clip((target - current[:2]) / 100.0, -1.0, 1.0)
            probe["fallback"] = "reestablish_contact"
        else:
            probe["fallback"] = None
        next_state, terminated = execute(env, current, goal, action)
        improvement = best_cost - pose_cost(next_state, goal)
        if improvement < 0.05:
            stagnant += 1
        else:
            stagnant = 0
        records.append({
            "step": step,
            "phase": phase,
            "state_before": current.copy(),
            "action": action.copy(),
            "state_after": next_state.copy(),
            "probe_best_score": probe["best_score"],
            "probe_max_block_motion": probe["max_block_motion"],
            "probe_fallback": probe["fallback"],
            "block_delta": (next_state[2:6] - current[2:6]).copy(),
            "xy_error_before": xy_error,
            "angle_error_before": angle,
            "pose_cost_before": pose_cost(current, goal),
            "pose_cost_after": pose_cost(next_state, goal),
        })
        current = next_state
        best_cost = min(best_cost, pose_cost(current, goal))
        if terminated or task_success(current, goal):
            break
        if stagnant >= 18:
            break
    return current, records, terminated


def return_agent(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    steps: int = 45,
    clearance: float = 140.0,
) -> tuple[np.ndarray, bool]:
    current = np.asarray(state, dtype=np.float64).copy()
    terminated = False
    # Once block pose is closed, returning the agent becomes a constrained
    # maneuver: a return action that leaves the block termination region is
    # rejected.  The previous implementation scored detach candidates by
    # separation and a soft block-change penalty, which allowed a successful
    # push to be rotated back out of the termination region during return.
    preserve_block = block_pose_success(current, goal)
    route_clearance = max(float(clearance), 160.0) if preserve_block else float(clearance)
    # If block closure happens while the agent is still touching the T, a
    # nominally safe return can immediately re-contact it.  Probe short
    # outward manoeuvres first and choose one that increases separation while
    # preserving the just-closed block pose.
    if np.linalg.norm(current[:2] - current[2:4]) < clearance:
        base_block = current[2:5].copy()
        outward = current[:2] - current[2:4]
        outward /= max(np.linalg.norm(outward), 1e-8)
        base_angle = float(base_block[2])
        probe_offsets = (0.0, -np.pi / 4.0, np.pi / 4.0, -np.pi / 2.0,
                         np.pi / 2.0, np.pi)
        detach_candidates = []
        for offset in probe_offsets:
            c, s = np.cos(offset), np.sin(offset)
            direction = np.asarray([
                c * outward[0] - s * outward[1],
                s * outward[0] + c * outward[1],
            ])
            trial = current.copy()
            terminated_trial = False
            for _ in range(5):
                # Use a low-gain detach.  The previous 0.35 probe could leave
                # the block inside the hard pose margin while imparting enough
                # velocity that the immediately following return waypoint
                # destroyed closure.
                trial, terminated_trial = execute(
                    env, trial, goal, 0.10 * direction
                )
                if terminated_trial:
                    break
            block_change = float(np.linalg.norm(trial[2:4] - base_block[:2]))
            block_angle_change = angle_error(float(trial[4]), base_angle)
            separation = float(np.linalg.norm(trial[:2] - trial[2:4]))
            detach_candidates.append((
                separation - 3.0 * (block_change + 40.0 * block_angle_change),
                trial,
                {
                    "direction": direction,
                    "separation": separation,
                    "block_change": block_change,
                    "block_angle_change": block_angle_change,
                },
                terminated_trial,
            ))
        valid_detach = [
            item for item in detach_candidates
            if not preserve_block or block_pose_success(item[1], goal)
        ]
        if preserve_block and not valid_detach:
            # Keep the completed block pose as the authoritative state.  The
            # caller may report agent-return failure, but must not turn a
            # solved block phase into a failed task by destructive recovery.
            return np.asarray(state, dtype=np.float64).copy(), False
        _, current, detach_info, detach_terminated = max(
            valid_detach or detach_candidates, key=lambda item: item[0]
        )
        if detach_terminated:
            return current, True

    def move_preserving(target: np.ndarray, move_steps: int) -> tuple[np.ndarray, bool]:
        """Move toward a waypoint while hard-gating block-pose regression."""
        nonlocal current
        for _ in range(move_steps):
            delta = np.asarray(target, dtype=np.float64) - current[:2]
            if float(np.linalg.norm(delta)) <= 8.0:
                break
            action = np.clip(delta / 180.0, -1.0, 1.0)
            trial, trial_terminated = execute(env, current, goal, action)
            if preserve_block and not block_pose_success(trial, goal):
                return current, False
            current = trial
            if trial_terminated:
                return current, True
        return current, False

    # A direct return can cross the block and undo a successful block pose.
    # Treat the final agent target as another obstacle-avoidance query.
    outward = current[:2] - current[2:4]
    norm = float(np.linalg.norm(outward))
    if norm < 1e-6:
        outward = np.asarray([1.0, 0.0], dtype=np.float64)
        norm = 1.0
    outward /= norm
    retreat_target = current[:2] + outward * (route_clearance + 35.0)
    if preserve_block:
        current, terminated = move_preserving(
            retreat_target, max(12, steps // 3)
        )
    else:
        current, _, _, terminated = move_toward(
            env, current, goal, retreat_target, max(12, steps // 3), 42.0,
            stop_on_contact=False,
        )
    if terminated:
        return current, terminated
    waypoints = route_waypoints(current, goal[:2], route_clearance)
    for waypoint in waypoints:
        if preserve_block:
            current, terminated = move_preserving(
                waypoint, max(12, steps // 3)
            )
        else:
            current, _, _, terminated = move_toward(
                env, current, goal, waypoint, max(12, steps // 3), 42.0,
                stop_on_contact=False,
            )
        if terminated:
            break
    return current, terminated


def run_case(env: gym.Env, case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    initial = np.asarray(case["state"], dtype=np.float64).copy()
    goal = np.asarray(case["goal"], dtype=np.float64).copy()
    direction = goal[2:4] - initial[2:4]
    direction /= max(np.linalg.norm(direction), 1e-8)
    routed, route_info = route_to_contact(env, initial, goal, direction, args)
    pushed, records, push_terminated = local_push(env, routed, goal, args)
    final, return_terminated = return_agent(env, pushed, goal, clearance=args.clearance)
    return {
        "initial_state": initial,
        "goal": goal,
        "routed_state": routed,
        "pushed_state": pushed,
        "final_state": final,
        "route": route_info,
        "n_local_steps": len(records),
        "local_records": records,
        "route_block_xy_displacement": float(np.linalg.norm(routed[2:4] - initial[2:4])),
        "pushed_block_xy_distance": block_xy_cost(pushed, goal),
        "final_block_xy_distance": block_xy_cost(final, goal),
        "final_agent_xy_distance": float(np.linalg.norm(final[:2] - goal[:2])),
        "final_angle_error": angle_error(final[4], goal[4]),
        "final_pose_cost": pose_cost(final, goal),
        "success": task_success(final, goal),
        "block_pose_success": bool(block_xy_cost(final, goal) < 20.0 and angle_error(final[4], goal[4]) < np.pi / 9.0),
        "push_terminated": push_terminated,
        "return_terminated": return_terminated,
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        cases = [
            stable_contact_bootstrap(env, args.seed + 10000 + index, args.bootstrap_steps)
            for index in range(args.episodes)
        ]
        rows = [run_case(env, case, args) for case in cases]
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "episodes": args.episodes,
            "seed": args.seed,
            "bootstrap": "stable contact; simulator state used only to initialize diagnostic cases",
            "planner": "obstacle-aware polyline + one-step simulator action-effect probe + greedy execution",
            "probe_directions": args.probe_directions,
            "clearance": args.clearance,
            "contact_distance": args.contact_distance,
            "max_steps": args.max_steps,
            "wall_seconds": time.perf_counter() - started,
        },
        "summary": {
            "n": len(rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "block_pose_success_rate": float(np.mean([row["block_pose_success"] for row in rows])),
            "mean_route_block_displacement": float(np.mean([row["route_block_xy_displacement"] for row in rows])),
            "mean_pushed_block_xy_distance": float(np.mean([row["pushed_block_xy_distance"] for row in rows])),
            "mean_final_block_xy_distance": float(np.mean([row["final_block_xy_distance"] for row in rows])),
            "mean_final_agent_xy_distance": float(np.mean([row["final_agent_xy_distance"] for row in rows])),
            "mean_final_angle_error": float(np.mean([row["final_angle_error"] for row in rows])),
            "mean_final_pose_cost": float(np.mean([row["final_pose_cost"] for row in rows])),
        },
        "episodes": rows,
        "sanity_checks": {
            "all_finite": bool(all(np.all(np.isfinite(np.asarray(row["final_state"]))) for row in rows)),
            "same_case_count": len(rows) == args.episodes,
            "probe_rows_present": bool(all(len(row["local_records"]) > 0 for row in rows)),
        },
        "claim_boundary": (
            "Privileged simulator-state repair diagnostic. It tests whether safe contact acquisition and local action-effect "
            "feedback can repair the oracle planning failure. It is not evidence for frozen-LeWM observability, a visual "
            "online planner, learned physical causality, or generalization beyond the matched PushT cases."
        ),
    }
    (output / "obstacle_aware_local_effect.json").write_text(json.dumps(jsonable(report), indent=2, sort_keys=True))
    summary = report["summary"]
    lines = [
        "# PushT obstacle-aware local-effect planning",
        "",
        "| success | block-pose success | route block displacement | pushed block XY | final block XY | final agent XY | final angle |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |"
        % (summary["success_rate"], summary["block_pose_success_rate"], summary["mean_route_block_displacement"], summary["mean_pushed_block_xy_distance"], summary["mean_final_block_xy_distance"], summary["mean_final_agent_xy_distance"], summary["mean_final_angle_error"]),
        "",
        report["claim_boundary"],
        "",
    ]
    (output / "obstacle_aware_local_effect.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "summary": summary, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
