"""Test whether candidate-coverage gains transfer after ranker augmentation.

The preceding intervention changed the candidate set while freezing a ranker
trained on the historical fan.  This runner separates two explanations for
the resulting failure: insufficient coverage versus ranker covariate shift.
It fits a baseline ranker and a dense-fan-augmented ranker on the same
training states, then evaluates both on identical held-out state--goal groups
and identical baseline/dense candidate sets.  All simulator successors are
used only for labels; selection remains purely learned.
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
    candidate_sets,
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
    feature_vector,
    fit_models,
    make_sequences,
)
from experiments.run_pusht_object_centric_learned_planner import (  # noqa: E402
    make_state_only_env,
)


ARMS = (
    "baseline_ranker_on_baseline",
    "baseline_ranker_on_dense",
    "dense_ranker_on_baseline",
    "dense_ranker_on_dense",
)


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
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261096)
    return parser.parse_args()


def key(sequence: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(sequence, dtype=np.float64).reshape(-1), 7).tolist())


def fit_sampled_ranker(
    env: Any,
    states: list[np.ndarray],
    actions: np.ndarray,
    goals_per_state: int,
    max_sequences: int,
    seed: int,
    action_repeat: int,
    trees: int,
) -> dict[str, Any]:
    sequences = make_sequences(actions)
    rng = np.random.default_rng(seed)
    features = []
    label_names = (
        "pose", "turn", "motion", "contact", "joint",
        "angle_progress", "xy_progress", "motion_distance",
    )
    labels = {name: [] for name in label_names}
    for state_index, state in enumerate(states):
        goals = [
            goal_for_state(state, state_index * goals_per_state + goal_index, rng, closure_only=True)
            for goal_index in range(goals_per_state)
        ]
        for goal_index, goal in enumerate(goals):
            if len(sequences) > max_sequences:
                indices = rng.choice(len(sequences), size=max_sequences, replace=False)
                selected = sequences[np.sort(indices)]
            else:
                selected = sequences
            for sequence_index, sequence in enumerate(selected):
                env.reset(
                    seed=seed + state_index * 1000000 + goal_index * 10000 + sequence_index * 10,
                    options={"state": state.copy(), "goal_state": state.copy()},
                )
                endpoint = np.asarray(state, dtype=np.float64).copy()
                terminated = truncated = False
                for action in sequence:
                    for _ in range(action_repeat):
                        observation, _, terminated, truncated, _ = env.step(action)
                        endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
                        if terminated or truncated:
                            break
                    if terminated or truncated:
                        break
                features.append(feature_vector(state, sequence[0], sequence[1], np.zeros(2), goal, "local"))
                target = label_sequence(state, goal, endpoint)
                for name in labels:
                    labels[name].append(target[name])
    features_array = np.asarray(features, dtype=np.float64)
    labels_array = {
        name: np.asarray(values, dtype=np.float64 if name in ("angle_progress", "xy_progress", "motion_distance") else np.int64)
        for name, values in labels.items()
    }
    return fit_models(features_array, labels_array, trees, seed + 7)


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


def evaluate_arm(
    models: dict[str, Any],
    state: np.ndarray,
    goal: np.ndarray,
    sequences: np.ndarray,
    truth_by_key: dict[tuple[float, ...], dict[str, Any]],
) -> dict[str, Any]:
    truths = [truth_by_key[key(sequence)] for sequence in sequences]
    selected, info = choose_sequence("sequence_continuous_rank", models, state, goal, sequences, "local")
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
        baseline_actions = action_bank(8, (0.2, 0.4, 0.6))
        dense_actions = action_bank(16, (0.15, 0.3, 0.45, 0.6))
        baseline_models = fit_sampled_ranker(
            env, train_states, baseline_actions, args.train_goals_per_state,
            args.train_sequences, args.seed + 300000, args.action_repeat, args.trees,
        )
        dense_models = fit_sampled_ranker(
            env, train_states, dense_actions, args.train_goals_per_state,
            args.train_sequences, args.seed + 400000, args.action_repeat, args.trees,
        )
        goal_rng = np.random.default_rng(args.seed + 500000)
        for state_index, state in enumerate(held_states):
            for goal_index in range(args.goals_per_state):
                goal = goal_for_state(state, state_index * args.goals_per_state + goal_index, goal_rng, closure_only=True)
                candidates = candidate_sets(state, goal)
                evaluation_sets = {"baseline": candidates["baseline_fan"], "dense": candidates["dense_fan"]}
                union: dict[tuple[float, ...], np.ndarray] = {}
                for sequences in evaluation_sets.values():
                    for sequence in sequences:
                        union.setdefault(key(sequence), sequence)
                truth_by_key = {}
                for sequence_index, sequence in enumerate(union.values()):
                    endpoint = sequence_endpoint.action_repeat
                    sequence_endpoint.action_repeat = args.action_repeat
                    final = sequence_endpoint(env, state, sequence, args.seed + 600000 + state_index * 10000 + goal_index * 100 + sequence_index * 11)
                    sequence_endpoint.action_repeat = endpoint
                    truth_by_key[key(sequence)] = label_sequence(state, goal, final)
                arm_specs = {
                    "baseline_ranker_on_baseline": (baseline_models, "baseline"),
                    "baseline_ranker_on_dense": (baseline_models, "dense"),
                    "dense_ranker_on_baseline": (dense_models, "baseline"),
                    "dense_ranker_on_dense": (dense_models, "dense"),
                }
                for arm, (models, candidate_name) in arm_specs.items():
                    row = evaluate_arm(models, state, goal, evaluation_sets[candidate_name], truth_by_key)
                    row.update({"state_index": state_index, "goal_index": goal_index})
                    rows[arm].append(row)
    finally:
        env.close()

    summaries = {}
    for index, (arm, values) in enumerate(rows.items()):
        metrics = (
            "candidate_count", "oracle_available", "oracle_count", "selected_feasible",
            "top8_hit", "top32_hit", "selected_true_score", "oracle_best_score",
            "selected_rank_by_true_score", "selected_angle_progress", "selected_xy_progress",
            "selected_contact",
        )
        summaries[arm] = {
            metric: bootstrap(np.asarray([row[metric] for row in values]), args.bootstrap, args.seed + 700 + index * 100 + metric_index)
            for metric_index, metric in enumerate(metrics)
        }
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "train_contact_states": len(train_states),
            "held_out_contact_states": len(held_states),
            "held_out_state_goal_groups": len(rows[ARMS[0]]),
            "action_repeat": args.action_repeat,
            "train_sequences_per_state_goal": args.train_sequences,
            "ranker_training": "baseline fan versus dense fan, same training states and goals",
            "runtime_candidate_successor_verification": False,
            "simulator_successor_used_for_labels_only": True,
            "same_state_goal_groups_across_arms": True,
            "wall_seconds": time.perf_counter() - started,
        },
        "summaries": summaries,
        "rows": rows,
        "claim_boundary": (
            "Privileged simulator-state candidate-distribution intervention. This test "
            "separates candidate coverage from ranker covariate shift; it does not establish "
            "visual LeWM planning, physical latent causality, universal world-model capability, "
            "or task success."
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
            "runtime_candidate_successor_verification": False,
        },
    }
    (output / "candidate_training_shift.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# PushT candidate-distribution and ranker-shift audit", "",
        "| arm | oracle coverage | selected feasible | top-8 hit | true rank | angle progress | XY progress |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        summary = summaries[arm]
        lines.append("| %s | %.3f | %.3f | %.3f | %.1f | %.4f | %.3f |" % (
            arm,
            summary["oracle_available"]["estimate"],
            summary["selected_feasible"]["estimate"],
            summary["top8_hit"]["estimate"],
            summary["selected_rank_by_true_score"]["estimate"],
            summary["selected_angle_progress"]["estimate"],
            summary["selected_xy_progress"]["estimate"],
        ))
    lines += ["", report["claim_boundary"], ""]
    (output / "candidate_training_shift.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "summaries": summaries, "sanity_checks": report["sanity_checks"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
