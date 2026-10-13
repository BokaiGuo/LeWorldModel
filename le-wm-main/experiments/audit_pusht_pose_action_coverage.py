"""Audit whether the action fan contains a one-step pose-closing action.

This is a capability audit, not a planner result.  For held-out contact
states and near-pose goals, it measures the oracle coverage of several fixed
action fans.  The simulator successor is used only to label coverage; no
learned selector or runtime verification is involved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_pusht_contact_mode_effective_motion import goal_for_state
from experiments.run_pusht_contact_history_velocity_diagnostic import collect_trajectory_data, mode_for_state
from experiments.run_pusht_direct_pose_decision_ablation import action_endpoint, label_endpoint
from experiments.run_pusht_object_centric_learned_planner import make_state_only_env, torque_action_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--held-out-states", type=int, default=32)
    parser.add_argument("--goals-per-state", type=int, default=6)
    parser.add_argument("--pool-transitions", type=int, default=4096)
    parser.add_argument("--directions", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=20261050)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = make_state_only_env()
    fans = {
        "small": [0.10, 0.15, 0.25],
        "medium": [0.10, 0.15, 0.25, 0.35],
        "large": [0.10, 0.15, 0.25, 0.35, 0.55],
    }
    try:
        pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 100000)
        states = []
        indices = np.flatnonzero(np.asarray(pool["modes"]) == "contact")
        rng = np.random.default_rng(args.seed + 1)
        rng.shuffle(indices)
        states = [np.asarray(pool["states"])[index].copy() for index in indices[:args.held_out_states]]
        goal_rng = np.random.default_rng(args.seed + 2)
        reports = []
        for horizon in args.horizons:
            for direction_count in args.directions:
                for fan_name, magnitudes in fans.items():
                    actions = torque_action_bank(direction_count, magnitudes=tuple(magnitudes))
                    rows = []
                    for state_index, state in enumerate(states):
                        for goal_index in range(args.goals_per_state):
                            goal = goal_for_state(state, state_index * args.goals_per_state + goal_index, goal_rng, closure_only=True)
                            labels = [
                                label_endpoint(
                                    state,
                                    goal,
                                    action_endpoint(env, state, action, horizon, args.seed + 300000 + horizon * 1000000 + state_index * 10000 + goal_index * 100 + action_index),
                                    require_contact=mode_for_state(state) == "contact",
                                )
                                for action_index, action in enumerate(actions)
                            ]
                            rows.append({
                                "oracle_count": int(sum(item["joint"] for item in labels)),
                                "pose_count": int(sum(item["pose"] for item in labels)),
                                "contact_count": int(sum(item["contact"] for item in labels)),
                                "motion_count": int(sum(item["motion"] for item in labels)),
                            })
                    reports.append({
                        "horizon": horizon,
                        "fan": fan_name,
                        "directions": direction_count,
                        "action_count": len(actions),
                        "oracle_coverage": float(np.mean([row["oracle_count"] > 0 for row in rows])),
                        "pose_coverage": float(np.mean([row["pose_count"] > 0 for row in rows])),
                        "contact_coverage": float(np.mean([row["contact_count"] > 0 for row in rows])),
                        "motion_coverage": float(np.mean([row["motion_count"] > 0 for row in rows])),
                        "mean_oracle_count": float(np.mean([row["oracle_count"] for row in rows])),
                        "mean_pose_count": float(np.mean([row["pose_count"] for row in rows])),
                        "n_groups": len(rows),
                    })
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "held_out_contact_states": len(states),
            "goals_per_state": args.goals_per_state,
            "independent_reset_stream": True,
            "horizons": [int(value) for value in args.horizons],
            "runtime_successor_verification": False,
            "wall_seconds": time.perf_counter() - started,
        },
        "reports": reports,
        "claim_boundary": "Held-out repeated-action action-fan coverage audit. It separates candidate-set insufficiency from learned ranking error; it is not evidence of planning success or world-model universality.",
        "sanity_checks": {"all_finite": bool(all(np.isfinite(value) for row in reports for value in row.values() if isinstance(value, (int, float)))), "same_states_and_goals_within_fan": True},
    }
    (output / "pose_action_coverage.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# PushT pose-action coverage audit", "", "| horizon | fan | directions | actions | oracle coverage | pose coverage | contact coverage | mean oracle count |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in reports:
        lines.append("| %d | %s | %d | %d | %.3f | %.3f | %.3f | %.3f |" % (row["horizon"], row["fan"], row["directions"], row["action_count"], row["oracle_coverage"], row["pose_coverage"], row["contact_coverage"], row["mean_oracle_count"]))
    lines += ["", report["claim_boundary"], ""]
    (output / "pose_action_coverage.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "reports": reports, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
