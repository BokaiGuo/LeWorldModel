"""Replay a strict terminal stability certificate on completed PushT audits.

This audit does not rerun the planner.  It takes the final states from two
already completed matched privileged planning runs, replays a zero-action hold
from each endpoint, and checks the strict joint criterion and block-pose
criterion after the hold.  It is an interface audit, not evidence for visual
LeWM planning or learned physical causality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
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

gym.logger.min_level = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hold-steps", type=int, default=1)
    parser.add_argument(
        "--bootstrap-steps",
        type=int,
        default=80,
        help="bootstrap length used by the source receding-planner protocol",
    )
    return parser.parse_args()


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
    if args.hold_steps < 1:
        raise ValueError("hold-steps must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    env.unwrapped.render = lambda: np.zeros((1, 1, 3), dtype=np.uint8)
    rows = []
    try:
        for input_path in args.inputs:
            source = Path(input_path)
            report = json.loads(source.read_text())
            source_seed = int(report["protocol"]["seed"])
            for index, row in enumerate(report["reports"]["receding_block_pose"]):
                state = np.asarray(row["final_state"], dtype=np.float64)
                case = stable_contact_bootstrap(
                    env,
                    source_seed + 10000 + index,
                    args.bootstrap_steps,
                )
                goal = np.asarray(case["goal"], dtype=np.float64)
                before_strict = bool(task_success(state, goal))
                before_block = bool(block_pose_success(state, goal))
                current = state.copy()
                hold_states = []
                terminated = False
                truncated = False
                for _ in range(args.hold_steps):
                    env.reset(seed=0, options={"state": current.copy(), "goal_state": goal.copy()})
                    observation, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
                    current = np.asarray(observation["state"], dtype=np.float64).copy()
                    hold_states.append(current.copy())
                rows.append({
                    "source": str(source),
                    "episode": int(index),
                    "skipped": False,
                    "source_success": bool(row["success"]),
                    "endpoint_strict": before_strict,
                    "endpoint_block_pose": before_block,
                    "hold_strict": bool(task_success(current, goal)),
                    "hold_block_pose": bool(block_pose_success(current, goal)),
                    "stable": bool(
                        before_strict
                        and before_block
                        and task_success(current, goal)
                        and block_pose_success(current, goal)
                    ),
                    "terminated": bool(terminated or truncated),
                    "endpoint_state": state,
                    "hold_state": current,
                    "hold_states": np.asarray(hold_states),
                })
    finally:
        env.close()
    checked = [row for row in rows if not row["skipped"]]
    report = {
        "protocol": {
            "inputs": [str(Path(value)) for value in args.inputs],
            "hold_steps": int(args.hold_steps),
            "criterion": "strict joint PushT endpoint plus block-pose preservation",
            "selection": "completed privileged receding-block-pose endpoints; no planner rerun",
        },
        "summary": {
            "n_rows": len(rows),
            "n_checked": len(checked),
            "n_skipped": len(rows) - len(checked),
            "source_success": int(sum(row["source_success"] for row in checked)),
            "endpoint_strict": int(sum(row["endpoint_strict"] for row in checked)),
            "hold_strict": int(sum(row["hold_strict"] for row in checked)),
            "stable": int(sum(row["stable"] for row in checked)),
            "stability_rate": float(np.mean([row["stable"] for row in checked])) if checked else None,
        },
        "rows": rows,
        "sanity_checks": {
            "all_checked_finite": bool(all(
                np.isfinite(np.asarray(row["hold_state"])).all() for row in checked
            )),
            "all_source_success_checked": bool(all(row["source_success"] for row in checked)),
            "no_unexpected_skips": bool(not any(row["skipped"] for row in rows)),
        },
        "claim_boundary": (
            "Terminal-interface stability audit for privileged simulator-state endpoints. "
            "It validates strict endpoint persistence under a short zero-action hold; "
            "it does not establish visual LeWM planning, learned physical causality, "
            "or general PushT performance."
        ),
    }
    (output / "terminal_stability_audit.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True)
    )
    summary = report["summary"]
    lines = [
        "# PushT terminal stability audit",
        "",
        "| checked | source strict success | endpoint strict | hold strict | stable | stability rate |",
        "|---:|---:|---:|---:|---:|---:|",
        "| %d | %d | %d | %d | %d | %.3f |"
        % (
            summary["n_checked"],
            summary["source_success"],
            summary["endpoint_strict"],
            summary["hold_strict"],
            summary["stable"],
            summary["stability_rate"] or 0.0,
        ),
        "",
        report["claim_boundary"],
        "",
    ]
    (output / "terminal_stability_audit.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "summary": summary, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
