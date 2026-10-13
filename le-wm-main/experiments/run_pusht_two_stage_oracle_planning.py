"""Two-stage privileged PushT planning diagnosis.

The original endpoint objective couples block placement and agent return too
early.  In contact tasks this can reject a trajectory that reaches the target
block pose but leaves the agent temporarily far away.  This experiment uses
the same simulator CEM candidate generator in two modes:

* ``coupled`` ranks the whole trajectory by the final joint state;
* ``two_stage`` ranks block pose with short-prefix receding replanning, stops
  at block closure, and then executes a separate block-avoiding return.

Both arms are privileged simulator-state upper bounds.  The experiment is
designed to identify the planning bottleneck; it is not a frozen-LeWM result.
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

from experiments.run_pusht_repaired_predictor_mpc import (
    staged_proposal_sequences,
    stable_contact_bootstrap,
)
from experiments.run_pusht_obstacle_aware_local_effect import return_agent

gym.logger.min_level = 40


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--population", type=int, default=96)
    p.add_argument("--cem-iterations", type=int, default=4)
    p.add_argument("--plan-blocks", type=int, default=20)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument(
        "--replan-blocks",
        type=int,
        default=2,
        help="execute only this many action blocks before replanning the push phase",
    )
    p.add_argument(
        "--max-push-replans",
        type=int,
        default=24,
        help="maximum receding push replans for the two-stage arm",
    )
    p.add_argument("--return-steps", type=int, default=50)
    p.add_argument(
        "--block-cost-mode",
        choices=("weighted_pose", "task_margin"),
        default="weighted_pose",
        help="block-phase score; task_margin matches the conjunctive PushT termination rule",
    )
    p.add_argument("--seed", type=int, default=8421)
    p.add_argument("--bootstrap-steps", type=int, default=80)
    return p.parse_args()


def angle_error(value: float, target: float) -> float:
    return float(abs((float(value) - float(target) + np.pi) % (2 * np.pi) - np.pi))


def block_cost(state: np.ndarray, goal: np.ndarray) -> float:
    state = np.asarray(state, dtype=np.float64)
    return float(np.linalg.norm(state[2:4] - goal[2:4]) + 40.0 * angle_error(state[4], goal[4]))


def task_margin_block_cost(state: np.ndarray, goal: np.ndarray) -> float:
    xy_margin = np.linalg.norm(np.asarray(state)[2:4] - np.asarray(goal)[2:4]) / 20.0
    angle_margin = angle_error(np.asarray(state)[4], np.asarray(goal)[4]) / (np.pi / 9.0)
    return float(max(xy_margin, angle_margin) + 0.01 * (xy_margin + angle_margin))


def joint_cost(state: np.ndarray, goal: np.ndarray) -> float:
    state = np.asarray(state, dtype=np.float64)
    return float(np.linalg.norm(state[:4] - goal[:4]) + 40.0 * angle_error(state[4], goal[4]))


def block_pose_success(state: np.ndarray, goal: np.ndarray) -> bool:
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return bool(
        np.linalg.norm(state[2:4] - goal[2:4]) < 20.0
        and angle_error(state[4], goal[4]) < np.pi / 9.0
    )


def reset_and_rollout(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, bool]:
    env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
    observation = None
    terminated = False
    for action in np.asarray(actions, dtype=np.float32):
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    if observation is None:
        raise RuntimeError("rollout produced no observation")
    return np.asarray(observation["state"], dtype=np.float64).copy(), bool(terminated)


def cem_plan(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
    objective: str,
) -> dict[str, Any]:
    horizon = args.plan_blocks * args.action_block
    structured = staged_proposal_sequences(
        rng, state, goal, args.population, horizon, args.action_block, 0.20
    ).reshape(args.population, args.plan_blocks, args.action_block, 2)[:, :, 0]
    mean = structured.mean(axis=0)
    std = np.maximum(structured.std(axis=0), 0.25)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    if objective == "block" and args.block_cost_mode == "task_margin":
        objective_fn = task_margin_block_cost
    else:
        objective_fn = block_cost if objective == "block" else joint_cost
    for iteration in range(args.cem_iterations):
        candidates = np.clip(
            mean[None] + std[None] * rng.normal(size=(args.population, args.plan_blocks, 2)),
            -1.0,
            1.0,
        ).astype(np.float32)
        warm = min(16, args.population)
        candidates[:warm] = structured[:warm]
        values, finals = [], []
        for candidate in candidates:
            final, _ = reset_and_rollout(
                env, state, goal,
                np.repeat(candidate, args.action_block, axis=0),
            )
            values.append(objective_fn(final, goal))
            finals.append(final)
        values = np.asarray(values, dtype=np.float64)
        finals = np.asarray(finals, dtype=np.float64)
        elite_count = max(8, args.population // 8)
        elite_index = np.argsort(values)[:elite_count]
        candidate_best = int(elite_index[0])
        if best is None or values[candidate_best] < best[0]:
            best = (
                float(values[candidate_best]),
                candidates[candidate_best].copy(),
                finals[candidate_best].copy(),
            )
        elite = candidates[elite_index]
        mean = 0.30 * mean + 0.70 * elite.mean(axis=0)
        std = 0.30 * std + 0.70 * np.maximum(elite.std(axis=0), 0.08)
    if best is None:
        raise RuntimeError("CEM did not produce a candidate")
    return {
        "objective": objective,
        "objective_value": best[0],
        "actions": best[1],
        "predicted_terminal_state": best[2],
        "iterations": args.cem_iterations,
        "population": args.population,
    }


def move_to_target(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    target: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, bool]:
    env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
    observation = None
    terminated = False
    current = state.copy()
    for _ in range(steps):
        action = np.clip((np.asarray(target) - current[:2]) / 100.0, -1.0, 1.0)
        observation, _, terminated, truncated, _ = env.step(action.astype(np.float32))
        current = np.asarray(observation["state"], dtype=np.float64).copy()
        if terminated or truncated:
            break
    return current, terminated


def safe_return(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return around the block before approaching the goal agent position."""
    state = np.asarray(state, dtype=np.float64).copy()
    direct = goal[:2] - state[:2]
    direct /= max(np.linalg.norm(direct), 1e-8)
    perpendicular = np.asarray([-direct[1], direct[0]])
    waypoints = [state[2:4] + 100.0 * perpendicular, state[2:4] - 100.0 * perpendicular]
    waypoint = min(waypoints, key=lambda value: np.linalg.norm(value - state[:2]))
    first, first_terminated = move_to_target(env, state, goal, waypoint, steps // 2)
    second, second_terminated = move_to_target(env, first, goal, goal[:2], steps - steps // 2)
    return second, {
        "waypoint": waypoint,
        "first_terminated": bool(first_terminated),
        "second_terminated": bool(second_terminated),
        "return_steps": steps,
    }


def receding_block_push(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Optimize only the block, execute a short prefix, and replan.

    The old two-stage arm separated the endpoint objective from return but
    still committed to one long open-loop push.  This function makes
    recoverability explicit: block pose is the only active objective, the
    agent is replanned from the observed simulator state, and the push phase
    stops immediately when the conjunctive block criterion is met.
    """
    current = np.asarray(state, dtype=np.float64).copy()
    trace: list[dict[str, Any]] = []
    for replan in range(args.max_push_replans):
        if block_pose_success(current, goal):
            break
        plan = cem_plan(env, current, goal, rng, args, "block")
        execute_blocks = min(max(1, args.replan_blocks), len(plan["actions"]))
        actions = np.repeat(
            np.asarray(plan["actions"][:execute_blocks], dtype=np.float32),
            args.action_block,
            axis=0,
        )
        before = current.copy()
        current, terminated = reset_and_rollout(env, current, goal, actions)
        trace.append({
            "replan": replan,
            "executed_blocks": execute_blocks,
            "objective_value": float(plan["objective_value"]),
            "state_before": before,
            "state_after": current.copy(),
            "block_xy_before": float(np.linalg.norm(before[2:4] - goal[2:4])),
            "block_xy_after": float(np.linalg.norm(current[2:4] - goal[2:4])),
            "angle_before": angle_error(before[4], goal[4]),
            "angle_after": angle_error(current[4], goal[4]),
            "terminated": bool(terminated),
        })
        if terminated:
            break
    return current, trace


def evaluate_arm(
    env: gym.Env,
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    arm: str,
) -> dict[str, Any]:
    rows = []
    for index, case in enumerate(cases):
        rng = np.random.default_rng(args.seed + 1000 * index + (0 if arm == "coupled" else 500000))
        if arm == "two_stage":
            pushed, push_trace = receding_block_push(
                env, case["state"], case["goal"], rng, args
            )
            pushed_terminated = bool(push_trace and push_trace[-1]["terminated"])
            plan = {
                "objective": "receding_block",
                "objective_value": float(push_trace[-1]["objective_value"])
                if push_trace else float(block_cost(pushed, case["goal"])),
                "trace": push_trace,
            }
            final, return_terminated = return_agent(
                env, pushed, case["goal"], steps=args.return_steps, clearance=140.0
            )
            return_info = {
                "return_steps": args.return_steps,
                "return_controller": "bounded_detach_and_obstacle_aware_return",
                "return_terminated": bool(return_terminated),
            }
        else:
            plan = cem_plan(
                env, case["state"], case["goal"], rng, args, "joint"
            )
            pushed, pushed_terminated = reset_and_rollout(
                env, case["state"], case["goal"],
                np.repeat(plan["actions"], args.action_block, axis=0),
            )
            final, return_info = pushed, {"return_steps": 0}
        rows.append({
            "case": index,
            "push_objective": plan["objective"],
            "push_objective_value": float(plan["objective_value"]),
            "pushed_block_cost": block_cost(pushed, case["goal"]),
            "pushed_joint_cost": joint_cost(pushed, case["goal"]),
            "final_block_cost": block_cost(final, case["goal"]),
            "final_joint_cost": joint_cost(final, case["goal"]),
            "block_xy_distance": float(np.linalg.norm(final[2:4] - case["goal"][2:4])),
            "agent_xy_distance": float(np.linalg.norm(final[:2] - case["goal"][:2])),
            "angle_error": angle_error(final[4], case["goal"][4]),
            "success": bool(np.linalg.norm(final[:4] - case["goal"][:4]) < 20.0 and angle_error(final[4], case["goal"][4]) < np.pi / 9.0),
            "block_pose_success": bool(np.linalg.norm(final[2:4] - case["goal"][2:4]) < 20.0 and angle_error(final[4], case["goal"][4]) < np.pi / 9.0),
            "pushed_terminated": bool(pushed_terminated),
            "push_trace": plan.get("trace", []),
            "return_info": return_info,
        })
    return {
        "arm": arm,
        "episodes": rows,
        "summary": {
            "n": len(rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "block_pose_success_rate": float(np.mean([row["block_pose_success"] for row in rows])),
            "mean_final_block_cost": float(np.mean([row["final_block_cost"] for row in rows])),
            "mean_final_joint_cost": float(np.mean([row["final_joint_cost"] for row in rows])),
            "mean_block_xy_distance": float(np.mean([row["block_xy_distance"] for row in rows])),
            "mean_agent_xy_distance": float(np.mean([row["agent_xy_distance"] for row in rows])),
            "mean_angle_error": float(np.mean([row["angle_error"] for row in rows])),
        },
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    started = time.perf_counter()
    try:
        cases = [
            stable_contact_bootstrap(env, args.seed + 10000 + index, args.bootstrap_steps)
            for index in range(args.episodes)
        ]
        reports = [
            evaluate_arm(env, cases, args, "coupled"),
            evaluate_arm(env, cases, args, "two_stage"),
        ]
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "episodes": args.episodes,
            "population": args.population,
            "cem_iterations": args.cem_iterations,
            "plan_blocks": args.plan_blocks,
            "action_block": args.action_block,
            "return_steps": args.return_steps,
            "bootstrap": "stable contact; same cases reused by coupled and two-stage arms",
            "coupled_objective": "joint agent/block endpoint cost",
            "two_stage_objective": "receding block pose cost followed by bounded safe agent return",
            "block_cost_mode": args.block_cost_mode,
            "receding_push": True,
            "replan_blocks": args.replan_blocks,
            "max_push_replans": args.max_push_replans,
            "wall_seconds": time.perf_counter() - started,
        },
        "reports": reports,
        "sanity_checks": {
            "same_cases": True,
            "all_finite": bool(all(np.isfinite(row["final_joint_cost"]) for report_row in reports for row in report_row["episodes"])),
            "arms": [row["arm"] for row in reports] == ["coupled", "two_stage"],
        },
        "claim_boundary": (
            "Privileged simulator-state planning diagnosis. A two-stage success demonstrates that the "
            "joint endpoint objective was a planning bottleneck for this PushT protocol; it does not "
            "establish frozen-LeWM observability or a learned online planner."
        ),
    }
    (output / "two_stage_oracle_planning.json").write_text(json.dumps(jsonable(report), indent=2, sort_keys=True))
    lines = [
        "# PushT two-stage oracle planning diagnosis",
        "",
        "| arm | task success | block-pose success | block XY | agent XY | angle error | joint cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in reports:
        row = item["summary"]
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (item["arm"], row["success_rate"], row["block_pose_success_rate"], row["mean_block_xy_distance"], row["mean_agent_xy_distance"], row["mean_angle_error"], row["mean_final_joint_cost"]))
    lines += ["", report["claim_boundary"], ""]
    (output / "two_stage_oracle_planning.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "reports": [item["summary"] for item in reports], "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
