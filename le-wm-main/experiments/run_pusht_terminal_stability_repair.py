"""Repair terminal returns using an explicit post-return stability certificate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_obstacle_aware_local_effect import (  # noqa: E402
    block_pose_success,
    task_success,
)
from experiments.run_pusht_repaired_predictor_mpc import stable_contact_bootstrap  # noqa: E402
from experiments.run_pusht_receding_block_pose_planner import (  # noqa: E402
    plan_closure_preserving_return,
)

gym.logger.min_level = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-steps", type=int, default=80)
    parser.add_argument("--terminal-stability-steps", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=910000)
    parser.add_argument("--closure-return-population", type=int, default=256)
    parser.add_argument("--closure-return-iterations", type=int, default=2)
    parser.add_argument("--closure-return-horizon", type=int, default=8)
    parser.add_argument("--closure-return-rescue-population", type=int, default=512)
    parser.add_argument("--closure-return-rescue-iterations", type=int, default=8)
    parser.add_argument("--closure-return-rescue-restarts", type=int, default=2)
    parser.add_argument("--closure-return-deep-population", type=int, default=1024)
    parser.add_argument("--closure-return-deep-iterations", type=int, default=16)
    parser.add_argument("--closure-return-deep-horizon", type=int, default=24)
    parser.add_argument("--closure-return-deep-restarts", type=int, default=2)
    return parser.parse_args()


def make_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        population=args.closure_return_population,
        horizon=args.closure_return_horizon,
        action_block=1,
        noise=0.18,
        disable_force_fan=True,
        force_fan_on_stagnation=False,
        force_fan_near_goal_only=False,
        force_fan_xy_threshold=60.0,
        closure_return_population=args.closure_return_population,
        closure_return_iterations=args.closure_return_iterations,
        closure_return_horizon=args.closure_return_horizon,
        closure_return_rescue_population=args.closure_return_rescue_population,
        closure_return_rescue_iterations=args.closure_return_rescue_iterations,
        closure_return_rescue_restarts=args.closure_return_rescue_restarts,
        closure_return_deep_population=args.closure_return_deep_population,
        closure_return_deep_iterations=args.closure_return_deep_iterations,
        closure_return_deep_horizon=args.closure_return_deep_horizon,
        closure_return_deep_restarts=args.closure_return_deep_restarts,
        closure_return_path_xy_margin=1.0,
        closure_return_path_angle_margin=0.02,
        return_steps=45,
        return_clearance=140.0,
        terminal_stability_steps=args.terminal_stability_steps,
    )


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
    if args.terminal_stability_steps < 1:
        raise ValueError("terminal-stability-steps must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    planner_args = make_args(args)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    env.unwrapped.render = lambda: np.zeros((1, 1, 3), dtype=np.uint8)
    rows = []
    started = time.perf_counter()
    try:
        for input_path in args.inputs:
            source = Path(input_path)
            report = json.loads(source.read_text())
            source_seed = int(report["protocol"]["seed"])
            for episode, source_row in enumerate(report["reports"]["receding_block_pose"]):
                case = stable_contact_bootstrap(
                    env,
                    source_seed + 10000 + episode,
                    args.bootstrap_steps,
                )
                goal = np.asarray(case["goal"], dtype=np.float64)
                pre_return = np.asarray(source_row["pre_return_state"], dtype=np.float64)
                env.reset(seed=0, options={"state": pre_return.copy(), "goal_state": goal.copy()})
                repaired, info = plan_closure_preserving_return(
                    env,
                    pre_return,
                    goal,
                    np.random.default_rng(args.seed_offset + source_seed + episode),
                    planner_args,
                )
                rows.append({
                    "source": str(source),
                    "seed": source_seed,
                    "episode": episode,
                    "source_success": bool(source_row["success"]),
                    "source_final_stable_unknown": True,
                    "pre_return_block_pose": bool(block_pose_success(pre_return, goal)),
                    "repaired_strict": bool(task_success(repaired, goal)),
                    "repaired_block_pose": bool(block_pose_success(repaired, goal)),
                    "terminal_stability": info.get("terminal_stability"),
                    "stable": bool(
                        task_success(repaired, goal)
                        and block_pose_success(repaired, goal)
                        and info.get("terminal_stability", {}).get("stable", False)
                    ),
                    "repaired_state": repaired,
                    "return_info": info,
                })
    finally:
        env.close()
    report = {
        "protocol": {
            "inputs": [str(Path(value)) for value in args.inputs],
            "terminal_stability_steps": args.terminal_stability_steps,
            "planner": "closure-preserving terminal CEM with strict post-return hold certificate",
            "wall_seconds": time.perf_counter() - started,
        },
        "summary": {
            "n": len(rows),
            "source_success": int(sum(row["source_success"] for row in rows)),
            "pre_return_block_pose": int(sum(row["pre_return_block_pose"] for row in rows)),
            "repaired_strict": int(sum(row["repaired_strict"] for row in rows)),
            "repaired_block_pose": int(sum(row["repaired_block_pose"] for row in rows)),
            "stable": int(sum(row["stable"] for row in rows)),
            "stability_rate": float(np.mean([row["stable"] for row in rows])) if rows else None,
        },
        "rows": rows,
        "sanity_checks": {
            "all_finite": bool(all(np.isfinite(np.asarray(row["repaired_state"])).all() for row in rows)),
            "all_pre_return_block_pose": bool(all(row["pre_return_block_pose"] for row in rows)),
            "all_source_success": bool(all(row["source_success"] for row in rows)),
        },
        "claim_boundary": (
            "Privileged terminal-interface repair. The artifact tests whether a stable strict endpoint "
            "can be recovered from already block-closed states; it is not evidence for visual LeWM "
            "planning, learned physical causality, or general PushT performance."
        ),
    }
    (output / "terminal_stability_repair.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True)
    )
    summary = report["summary"]
    (output / "terminal_stability_repair.md").write_text(
        "\n".join([
            "# PushT terminal stability repair",
            "",
            "| n | source success | repaired strict | repaired block pose | stable terminal | stability rate |",
            "|---:|---:|---:|---:|---:|---:|",
            "| %d | %d | %d | %d | %d | %.3f |" % (
                summary["n"], summary["source_success"], summary["repaired_strict"],
                summary["repaired_block_pose"], summary["stable"], summary["stability_rate"] or 0.0,
            ),
            "",
            report["claim_boundary"],
            "",
        ])
    )
    print(json.dumps({"output": str(output), "summary": summary, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
