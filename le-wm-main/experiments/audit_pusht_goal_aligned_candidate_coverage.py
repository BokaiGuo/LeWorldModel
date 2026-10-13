"""Audit a goal/contact-frame candidate intervention for PushT.

The existing candidate intervention increases the density of an absolute-action
fan.  This audit tests a sharper alternative: construct an ordered two-action
macro bank from the current agent--block geometry and the current block-to-goal
direction, without querying any successor state.  The learned continuous ranker
is frozen at the historical absolute-action training distribution.  Simulator
successors are used only once to create labels for the matched coverage audit.

The estimand is deliberately diagnostic: whether a state/goal-conditioned
candidate interface contains a pose-improving, contact-compatible macro-action
more often than the historical fan, and whether that availability transfers to
the frozen ranker.  A positive offline result is not task-success evidence.
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

from experiments.audit_pusht_candidate_generation_intervention import (  # noqa: E402
    action_bank,
    label_sequence,
    sequence_endpoint,
)
from experiments.audit_pusht_contact_mode_effective_motion import (  # noqa: E402
    goal_for_state,
    select_balanced_states,
)
from experiments.run_pusht_contact_history_velocity_diagnostic import (  # noqa: E402
    collect_trajectory_data,
)
from experiments.run_pusht_learned_sequence_verifier import (  # noqa: E402
    choose_sequence,
    make_sequences,
)
from experiments.run_pusht_object_centric_learned_planner import (  # noqa: E402
    local_effect_action_bank,
    make_state_only_env,
)
from experiments.audit_pusht_candidate_training_shift import fit_sampled_ranker  # noqa: E402


ARMS = ("absolute_fan", "goal_contact_frame", "hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-states", type=int, default=16)
    parser.add_argument("--held-out-states", type=int, default=16)
    parser.add_argument("--pool-transitions", type=int, default=4096)
    parser.add_argument("--goals-per-state", type=int, default=4)
    parser.add_argument("--train-goals-per-state", type=int, default=4)
    parser.add_argument("--train-sequences", type=int, default=1200)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--trees", type=int, default=96)
    parser.add_argument("--directions", type=int, default=16)
    parser.add_argument("--scale", type=float, default=0.55)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261101)
    return parser.parse_args()


def key(sequence: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(sequence, dtype=np.float64).reshape(-1), 7).tolist())


def bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(len(values)),
    }


def candidate_sets(state: np.ndarray, goal: np.ndarray, directions: int, scale: float) -> dict[str, np.ndarray]:
    absolute = make_sequences(action_bank(8, (0.2, 0.4, 0.6)))
    geometry_actions = local_effect_action_bank(state, goal, directions=directions, scale=scale)
    geometry = make_sequences(geometry_actions)
    merged: dict[tuple[float, ...], np.ndarray] = {}
    for sequence in np.concatenate([absolute, geometry], axis=0):
        merged.setdefault(key(sequence), sequence)
    return {
        "absolute_fan": absolute,
        "goal_contact_frame": geometry,
        "hybrid": np.asarray(list(merged.values()), dtype=np.float32),
    }


def evaluate_arm(
    models: dict[str, Any],
    state: np.ndarray,
    goal: np.ndarray,
    sequences: np.ndarray,
    truth_by_key: dict[tuple[float, ...], dict[str, Any]],
) -> dict[str, Any]:
    truths = [truth_by_key[key(sequence)] for sequence in sequences]
    selected, info = choose_sequence(
        "sequence_continuous_rank", models, state, goal, sequences, "local"
    )
    order = np.asarray(info["predicted_order"], dtype=np.int64)
    actual_score = np.asarray([
        -row["angle_progress"]
        + 0.02 * max(-row["xy_progress"], 0.0)
        + 0.35 * (1.0 - row["contact"])
        + 0.10 * max(0.5 - row["motion_distance"], 0.0)
        for row in truths
    ], dtype=np.float64)
    joint = np.asarray([row["joint"] for row in truths], dtype=np.int64)
    return {
        "candidate_count": int(len(sequences)),
        "oracle_available": int(joint.any()),
        "oracle_count": int(joint.sum()),
        "selected_feasible": int(joint[selected]),
        "top8_hit": int(joint[order[: min(8, len(order))]].any()),
        "top32_hit": int(joint[order[: min(32, len(order))]].any()),
        "selected_true_score": float(actual_score[selected]),
        "oracle_best_score": float(actual_score.min()),
        "selected_rank_by_true_score": int(np.flatnonzero(np.argsort(actual_score) == selected)[0] + 1),
        "selected_angle_progress": float(truths[selected]["angle_progress"]),
        "selected_xy_progress": float(truths[selected]["xy_progress"]),
        "selected_contact": int(truths[selected]["contact"]),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = make_state_only_env()
    rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    try:
        train_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 100000)
        held_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 200000)
        train_states = select_balanced_states(train_pool, args.train_states, args.seed + 1, ("contact",))
        held_states = select_balanced_states(held_pool, args.held_out_states, args.seed + 2, ("contact",))
        train_actions = action_bank(8, (0.2, 0.4, 0.6))
        models = fit_sampled_ranker(
            env, train_states, train_actions, args.train_goals_per_state,
            args.train_sequences, args.seed + 300000, args.action_repeat, args.trees,
        )
        goal_rng = np.random.default_rng(args.seed + 400000)
        for state_index, state in enumerate(held_states):
            for goal_index in range(args.goals_per_state):
                goal = goal_for_state(
                    state, state_index * args.goals_per_state + goal_index,
                    goal_rng, closure_only=True,
                )
                candidates = candidate_sets(state, goal, args.directions, args.scale)
                union: dict[tuple[float, ...], np.ndarray] = {}
                for sequences in candidates.values():
                    for sequence in sequences:
                        union.setdefault(key(sequence), sequence)
                truth_by_key: dict[tuple[float, ...], dict[str, Any]] = {}
                previous_repeat = sequence_endpoint.action_repeat
                sequence_endpoint.action_repeat = args.action_repeat
                try:
                    for sequence_index, sequence in enumerate(union.values()):
                        endpoint = sequence_endpoint(
                            env, state, sequence,
                            args.seed + 500000 + state_index * 10000 + goal_index * 100 + sequence_index * 11,
                        )
                        truth_by_key[key(sequence)] = label_sequence(state, goal, endpoint)
                finally:
                    sequence_endpoint.action_repeat = previous_repeat
                for arm in ARMS:
                    row = evaluate_arm(models, state, goal, candidates[arm], truth_by_key)
                    row.update({"state_index": state_index, "goal_index": goal_index})
                    rows[arm].append(row)
    finally:
        env.close()

    metrics = (
        "candidate_count", "oracle_available", "oracle_count", "selected_feasible",
        "top8_hit", "top32_hit", "selected_true_score", "oracle_best_score",
        "selected_rank_by_true_score", "selected_angle_progress", "selected_xy_progress",
        "selected_contact",
    )
    summaries = {
        arm: {
            metric: bootstrap(
                np.asarray([row[metric] for row in values]),
                args.bootstrap, args.seed + 700 + arm_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }
        for arm_index, (arm, values) in enumerate(rows.items())
    }
    paired = {
        arm: {
            metric: bootstrap(
                np.asarray([new[metric] - old[metric] for old, new in zip(rows["absolute_fan"], rows[arm])]),
                args.bootstrap, args.seed + 1200 + arm_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }
        for arm_index, arm in enumerate(("goal_contact_frame", "hybrid"))
    }
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "train_contact_states": len(train_states),
            "held_out_contact_states": len(held_states),
            "held_out_state_goal_groups": len(rows["absolute_fan"]),
            "action_repeat": args.action_repeat,
            "ranker_training": "continuous ranker trained only on the absolute fan",
            "goal_contact_frame_bank": {
                "directions": args.directions,
                "scale": args.scale,
                "uses_current_state_and_goal_only": True,
                "runtime_successor_queries": False,
            },
            "simulator_successor_used_for_labels_only": True,
            "same_state_goal_groups_across_arms": True,
            "wall_seconds": time.perf_counter() - started,
        },
        "summaries": summaries,
        "paired_vs_absolute": paired,
        "rows": rows,
        "claim_boundary": (
            "Privileged simulator-state candidate-availability intervention. The geometry-conditioned "
            "bank tests whether current contact and goal geometry changes oracle availability and frozen "
            "ranking; it does not establish visual planning, physical latent causality, task success, "
            "or universal world-model capability."
        ),
        "sanity_checks": {
            "all_finite": bool(all(
                np.isfinite(value)
                for values in rows.values()
                for row in values
                for value in row.values()
                if isinstance(value, (int, float, np.integer, np.floating))
            )),
            "same_state_goal_groups_across_arms": True,
            "runtime_successor_queries": False,
            "simulator_successor_used_for_labels_only": True,
        },
    }
    (output / "goal_aligned_candidate_coverage.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# Goal/contact-frame candidate coverage audit", "",
        "| candidate arm | candidates | oracle available | selected feasible | top-8 hit | true rank | angle progress | XY progress |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        summary = summaries[arm]
        lines.append(
            "| %s | %.1f | %.3f | %.3f | %.3f | %.1f | %.4f | %.3f |" % (
                arm,
                summary["candidate_count"]["estimate"],
                summary["oracle_available"]["estimate"],
                summary["selected_feasible"]["estimate"],
                summary["top8_hit"]["estimate"],
                summary["selected_rank_by_true_score"]["estimate"],
                summary["selected_angle_progress"]["estimate"],
                summary["selected_xy_progress"]["estimate"],
            )
        )
    lines += ["", report["claim_boundary"], ""]
    (output / "goal_aligned_candidate_coverage.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "summaries": summaries, "paired_vs_absolute": paired, "sanity_checks": report["sanity_checks"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
