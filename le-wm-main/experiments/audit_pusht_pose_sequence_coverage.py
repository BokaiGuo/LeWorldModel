"""Audit two-action macro coverage for PushT pose closure.

The audit separates a richer action sequence set from a learned selector.  It
uses held-out contact states and near-pose goals, evaluates the same action
bank as the existing pure-learned sequence confirmation (25 primitive actions
and 625 ordered two-action pairs), and reports whether a goal-directed,
motion-producing, contact-preserving macro exists.  Simulator successors are
used only for the oracle label; no learned model or runtime verifier is used.
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
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[0.2, 0.4, 0.6])
    parser.add_argument("--seed", type=int, default=20261070)
    return parser.parse_args()


def summarize_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_groups": len(groups),
        "oracle_coverage": float(np.mean([row["oracle_count"] > 0 for row in groups])),
        "pose_coverage": float(np.mean([row["pose_count"] > 0 for row in groups])),
        "motion_coverage": float(np.mean([row["motion_count"] > 0 for row in groups])),
        "contact_coverage": float(np.mean([row["contact_count"] > 0 for row in groups])),
        "mean_oracle_count": float(np.mean([row["oracle_count"] for row in groups])),
        "median_oracle_count": float(np.median([row["oracle_count"] for row in groups])),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = make_state_only_env()
    try:
        pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 100000)
        indices = np.flatnonzero(np.asarray(pool["modes"]) == "contact")
        rng = np.random.default_rng(args.seed + 1)
        rng.shuffle(indices)
        states = [np.asarray(pool["states"])[index].copy() for index in indices[:args.held_out_states]]
        goal_rng = np.random.default_rng(args.seed + 2)
        actions = torque_action_bank(args.directions, magnitudes=tuple(args.magnitudes))
        sequences = np.asarray([[first, second] for first in actions for second in actions], dtype=np.float32)
        groups = {"primitive_one_step": [], "repeated_two_step": [], "all_two_step": []}
        for state_index, state in enumerate(states):
            for goal_index in range(args.goals_per_state):
                goal = goal_for_state(state, state_index * args.goals_per_state + goal_index, goal_rng, closure_only=True)
                labels_one = []
                labels_repeat = []
                labels_all = []
                for action_index, action in enumerate(actions):
                    endpoint = action_endpoint(env, state, action, 1, args.seed + 300000 + state_index * 10000 + goal_index * 100 + action_index)
                    labels_one.append(label_endpoint(state, goal, endpoint, require_contact=mode_for_state(state) == "contact"))
                    endpoint_repeat = action_endpoint(env, state, action, 2, args.seed + 400000 + state_index * 10000 + goal_index * 100 + action_index)
                    labels_repeat.append(label_endpoint(state, goal, endpoint_repeat, require_contact=mode_for_state(state) == "contact"))
                for sequence_index, sequence in enumerate(sequences):
                    env.reset(seed=args.seed + 500000 + state_index * 100000 + goal_index * 1000 + sequence_index, options={"state": state.copy(), "goal_state": state.copy()})
                    endpoint = state.copy()
                    for action in sequence:
                        observation, _, terminated, truncated, _ = env.step(action)
                        endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
                        if terminated or truncated:
                            break
                    labels_all.append(label_endpoint(state, goal, endpoint, require_contact=mode_for_state(state) == "contact"))
                groups["primitive_one_step"].append({
                    "oracle_count": int(sum(row["joint"] for row in labels_one)),
                    "pose_count": int(sum(row["pose"] for row in labels_one)),
                    "motion_count": int(sum(row["motion"] for row in labels_one)),
                    "contact_count": int(sum(row["contact"] for row in labels_one)),
                })
                groups["repeated_two_step"].append({
                    "oracle_count": int(sum(row["joint"] for row in labels_repeat)),
                    "pose_count": int(sum(row["pose"] for row in labels_repeat)),
                    "motion_count": int(sum(row["motion"] for row in labels_repeat)),
                    "contact_count": int(sum(row["contact"] for row in labels_repeat)),
                })
                groups["all_two_step"].append({
                    "oracle_count": int(sum(row["joint"] for row in labels_all)),
                    "pose_count": int(sum(row["pose"] for row in labels_all)),
                    "motion_count": int(sum(row["motion"] for row in labels_all)),
                    "contact_count": int(sum(row["contact"] for row in labels_all)),
                })
    finally:
        env.close()
    summaries = {name: summarize_groups(rows) for name, rows in groups.items()}
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "held_out_contact_states": len(states),
            "goals_per_state": args.goals_per_state,
            "primitive_action_count": len(actions),
            "ordered_two_action_sequence_count": len(sequences),
            "directions": args.directions,
            "magnitudes": [float(value) for value in args.magnitudes],
            "goal_regime": "near-pose closure",
            "simulator_successor_used_for_labels_only": True,
            "runtime_candidate_verification": False,
            "same_state_goal_groups_across_conditions": True,
            "wall_seconds": time.perf_counter() - started,
        },
        "summaries": summaries,
        "claim_boundary": "Held-out macro-action coverage audit. It tests whether a richer two-action candidate set contains pose-directed contact-compatible actions; it does not establish learned ranking, task success, visual planning, or universal world modeling.",
        "sanity_checks": {
            "all_finite": bool(all(np.isfinite(value) for summary in summaries.values() for value in summary.values() if isinstance(value, (int, float)))),
            "same_state_goal_groups_across_conditions": True,
        },
    }
    (output / "pose_sequence_coverage.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# PushT two-action macro coverage audit", "", "| candidate set | count | oracle coverage | pose coverage | contact coverage | mean oracle count | median oracle count |", "|---|---:|---:|---:|---:|---:|---:|"]
    counts = {"primitive_one_step": len(actions), "repeated_two_step": len(actions), "all_two_step": len(sequences)}
    for name in ("primitive_one_step", "repeated_two_step", "all_two_step"):
        summary = summaries[name]
        lines.append("| %s | %d | %.3f | %.3f | %.3f | %.3f | %.1f |" % (name, counts[name], summary["oracle_coverage"], summary["pose_coverage"], summary["contact_coverage"], summary["mean_oracle_count"], summary["median_oracle_count"]))
    lines += ["", report["claim_boundary"], ""]
    (output / "pose_sequence_coverage.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "summaries": summaries, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
