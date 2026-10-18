"""Privileged ceiling for the exact RGB-backtrack action protocol.

This audit is intentionally narrower than ``audit_cube_planner_ceiling``:
the fixed online bank is replayed with the same RGB progress feedback,
sign-reversal, gain-backtrack, and fail-closed cycle gate used by the
``lowgain_backtrack`` arm.  A privileged evaluator chooses the best candidate
only after replay.  It therefore separates fixed-bank support from the
adaptive action protocol, but is not learned-world-model or RGB-policy
evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import stable_worldmodel  # noqa: F401 - registers environments

from experiments.audit_cube_planner_ceiling import (
    _adaptive_oracle_for_bank,
    _capture_initial_condition,
    _capture_rgb_decision_state,
    _fit_rgb_pixel_action_mapping,
    _goal_conditioned_push_bank,
)
from experiments.run_live_cube_rgb_macro_ranker import _online_macro_bank


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")
    distances = [row["adaptive"]["oracle"]["final_goal_distance_m"] for row in records]
    result = {
        "roots": len(records),
        "adaptive_fixed_bank_reachability_rate": float(np.mean([
            row["adaptive"]["reachable"] for row in records
        ])),
        "adaptive_fixed_bank_oracle_distance_mean_m": float(np.mean(distances)),
        "adaptive_fixed_bank_success_count": int(sum(
            row["adaptive"]["success_count"] for row in records
        )),
        "claim_boundary": (
            "privileged post-hoc ceiling for the RGB-backtrack action protocol; "
            "not learned-world-model evidence, online policy evidence, or a transfer certificate"
        ),
    }
    if all("adaptive_goal_conditioned" in row for row in records):
        goal_rows = [row["adaptive_goal_conditioned"] for row in records]
        result.update({
            "adaptive_goal_conditioned_bank_reachability_rate": float(np.mean([
                row["reachable"] for row in goal_rows
            ])),
            "adaptive_goal_conditioned_bank_oracle_distance_mean_m": float(np.mean([
                row["oracle"]["final_goal_distance_m"] for row in goal_rows
            ])),
            "adaptive_goal_conditioned_bank_success_count": int(sum(
                row["success_count"] for row in goal_rows
            )),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261900)
    parser.add_argument("--test-roots", type=int, default=8)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--calibration-roots", type=int, default=40)
    parser.add_argument("--descent-steps", type=int, default=15)
    parser.add_argument("--approach-steps", type=int, default=8)
    parser.add_argument("--probe-steps", type=int, default=4)
    parser.add_argument("--macro-cycles", type=int, default=8)
    parser.add_argument("--macro-length", type=int, default=20)
    args = parser.parse_args()
    if args.test_roots < 1 or args.calibration_roots < 8:
        raise ValueError("test-roots must be positive and calibration-roots must be >=8")
    if args.macro_length < 20 or args.macro_cycles < 1:
        raise ValueError("macro-length must be >=20 and macro-cycles must be positive")
    if not args.tasks:
        raise ValueError("tasks must not be empty")

    print(
        "[purpose] exact RGB-backtrack adaptive fixed-bank ceiling; "
        f"roots={args.test_roots} cycles={args.macro_cycles}",
        flush=True,
    )
    pixel_mapping, probe_diagnostics = _fit_rgb_pixel_action_mapping(
        seed=args.seed,
        tasks=[int(task) for task in args.tasks],
        roots=args.calibration_roots,
        descent_steps=args.descent_steps,
        probe_steps=args.probe_steps,
    )
    bank = _online_macro_bank(length=args.macro_length, push_scale=0.25)
    records: list[dict[str, Any]] = []
    for index in range(args.test_roots):
        task_id = int(args.tasks[index % len(args.tasks)])
        initial = _capture_initial_condition(args.seed + 1000 + index, task_id)
        decision_state = _capture_rgb_decision_state(
            initial, pixel_mapping, args.descent_steps, args.approach_steps,
        )
        adaptive = _adaptive_oracle_for_bank(
            decision_state, bank, 0.05, args.macro_cycles,
        )
        goal_direction = (
            np.asarray(decision_state["goal"], dtype=np.float64)[:2]
            - np.asarray(decision_state["cube_position"], dtype=np.float64)[:2]
        )
        goal_conditioned_bank = _goal_conditioned_push_bank(
            goal_direction, length=args.macro_length,
        )
        adaptive_goal_conditioned = _adaptive_oracle_for_bank(
            decision_state, goal_conditioned_bank, 0.05, args.macro_cycles,
        )
        record = {
            "seed": int(initial["seed"]),
            "task_id": task_id,
            "adaptive": adaptive,
            "adaptive_goal_conditioned": adaptive_goal_conditioned,
        }
        records.append(record)
        print(
            f"root={index} task={task_id} "
            f"reachable={adaptive['reachable']} "
            f"distance={adaptive['oracle']['final_goal_distance_m']:.6g} "
            f"goal_conditioned_reachable={adaptive_goal_conditioned['reachable']} "
            f"goal_conditioned_distance={adaptive_goal_conditioned['oracle']['final_goal_distance_m']:.6g}",
            flush=True,
        )
    payload = {
        "schema": "cube_adaptive_protocol_ceiling.v1",
        "protocol": {
            "environment": "swm/OGBCube-v0",
            "root_disjoint": True,
            "candidate_bank": "online four-direction bank with push_scale=0.25",
            "goal_conditioned_candidate_bank": (
                "evaluator-only eight approach directions x goal-relative push "
                "offsets x three gains"
            ),
            "feedback": "RGB block-goal progress sign reversal with gain backtrack",
            "cycle_gate": "fail closed on RGB cycle regression",
            "calibration_roots": int(args.calibration_roots),
            "test_roots": int(args.test_roots),
            "macro_cycles": int(args.macro_cycles),
            "macro_length": int(args.macro_length),
            "descent_steps": int(args.descent_steps),
            "approach_steps": int(args.approach_steps),
            "probe_steps": int(args.probe_steps),
            "rgb_probe_diagnostics": probe_diagnostics,
            "candidate_selection": "privileged post-hoc final goal distance",
            "runtime_control_input": "RGB only",
            "claim_boundary": (
                "privileged action-protocol ceiling only; goal-conditioned bank "
                "uses evaluator-side cube/goal geometry and is not deployable RGB "
                "planning or learned-world-model evidence"
            ),
        },
        "aggregate": _summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
