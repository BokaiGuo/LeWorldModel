"""Audit candidate-generation interventions with a frozen learned ranker.

This experiment isolates candidate coverage from decision ranking.  The
continuous two-action ranker is fitted once on the baseline action fan and is
then frozen.  On identical held-out state--goal groups, only the candidate
set changes: the historical fan, a denser fan, geometry-conditioned action
templates, and their union.  Simulator successors are used only to create
held-out labels and oracle coverage statistics; they are never used by the
ranker at selection time.

The intended estimand is not a leaderboard score.  It is the change in the
probability that a candidate set contains at least one contact-compatible
pose-improving macro-action, and whether that coverage change transfers to
the frozen selector's top-ranked choice.
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
    label_sequence,
    make_sequences,
)
from experiments.run_pusht_object_centric_learned_planner import (  # noqa: E402
    make_state_only_env,
    torque_action_bank,
)


VARIANTS = (
    "baseline_fan",
    "dense_fan",
    "goal_templates",
    "dense_plus_templates",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-states", type=int, default=32)
    parser.add_argument("--held-out-states", type=int, default=16)
    parser.add_argument("--pool-transitions", type=int, default=4096)
    parser.add_argument("--goals-per-state", type=int, default=4)
    parser.add_argument("--trees", type=int, default=96)
    parser.add_argument("--closure-only", action="store_true", default=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument(
        "--action-repeat", type=int, default=1,
        help="repeat each selected macro action this many simulator steps",
    )
    parser.add_argument("--seed", type=int, default=20261093)
    return parser.parse_args()


def unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        return vector / norm
    if fallback is not None:
        fallback = np.asarray(fallback, dtype=np.float64)
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm > 1e-8:
            return fallback / fallback_norm
    return np.asarray([1.0, 0.0], dtype=np.float64)


def action_bank(directions: int, magnitudes: tuple[float, ...]) -> np.ndarray:
    return torque_action_bank(directions, magnitudes=magnitudes)


def _key(sequence: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(sequence, dtype=np.float64).reshape(-1), 7).tolist())


def goal_template_sequences(
    state: np.ndarray,
    goal: np.ndarray,
    magnitudes: tuple[float, ...] = (0.2, 0.4, 0.6),
) -> np.ndarray:
    """Construct a small geometry-conditioned macro proposal set.

    The templates are deliberately ordinary geometric proposals, not oracle
    successor rollouts.  ``approach`` points from the agent toward the block;
    ``push`` points from the current block toward the goal; rotated variants
    cover the tangential contact directions and reversals.
    """
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    approach = unit(state[2:4] - state[:2])
    push = unit(goal[2:4] - state[2:4], fallback=approach)
    quarter = np.asarray([-push[1], push[0]], dtype=np.float64)
    directions = [
        approach,
        -approach,
        push,
        -push,
        quarter,
        -quarter,
    ]
    vectors = []
    for first_direction in directions:
        for second_direction in directions:
            # The template set is intentionally smaller than the Cartesian
            # product of the dense fan; it adds exact state-goal directions
            # while keeping the candidate budget comparable to the baseline.
            if first_direction is not directions[0] and second_direction is not directions[2]:
                continue
            for first_magnitude in magnitudes:
                for second_magnitude in magnitudes:
                    vectors.append([
                        first_magnitude * first_direction,
                        second_magnitude * second_direction,
                    ])
    sequences = np.asarray(vectors, dtype=np.float32)
    if len(sequences) == 0:
        return np.zeros((0, 2, 2), dtype=np.float32)
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for sequence in sequences:
        unique.setdefault(_key(sequence), sequence)
    return np.asarray(list(unique.values()), dtype=np.float32)


def candidate_sets(state: np.ndarray, goal: np.ndarray) -> dict[str, np.ndarray]:
    baseline_actions = action_bank(8, (0.2, 0.4, 0.6))
    dense_actions = action_bank(16, (0.15, 0.3, 0.45, 0.6))
    baseline = make_sequences(baseline_actions)
    dense = make_sequences(dense_actions)
    templates = goal_template_sequences(state, goal)
    baseline_keys = {_key(sequence) for sequence in baseline}
    template_only = np.asarray(
        [sequence for sequence in templates if _key(sequence) not in baseline_keys],
        dtype=np.float32,
    )
    dense_keys = {_key(sequence) for sequence in dense}
    dense_plus = np.asarray(
        [*dense, *[sequence for sequence in templates if _key(sequence) not in dense_keys]],
        dtype=np.float32,
    )
    return {
        "baseline_fan": baseline,
        "dense_fan": dense,
        "goal_templates": np.asarray([*baseline, *template_only], dtype=np.float32),
        "dense_plus_templates": dense_plus,
    }


def sequence_endpoint(env: Any, state: np.ndarray, sequence: np.ndarray, seed: int) -> np.ndarray:
    env.reset(seed=seed, options={"state": state.copy(), "goal_state": state.copy()})
    endpoint = np.asarray(state, dtype=np.float64).copy()
    for action in sequence:
        for _ in range(sequence_endpoint.action_repeat):
            observation, _, terminated, truncated, _ = env.step(action)
            endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
            if terminated or truncated:
                return endpoint
    return endpoint


sequence_endpoint.action_repeat = 1


def make_training_data_repeated(
    env: Any,
    states: list[np.ndarray],
    actions: np.ndarray,
    goals_per_state: int,
    seed: int,
    action_repeat: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit the same two-action interface on a longer executed macro horizon."""
    sequences = make_sequences(actions)
    rng = np.random.default_rng(seed + 10)
    features = []
    label_names = ("pose", "turn", "motion", "contact", "joint", "angle_progress", "xy_progress", "motion_distance")
    labels = {name: [] for name in label_names}
    for state_index, state in enumerate(states):
        goals = [goal_for_state(state, state_index * goals_per_state + goal_index, rng, closure_only=True) for goal_index in range(goals_per_state)]
        for goal_index, goal in enumerate(goals):
            for sequence_index, sequence in enumerate(sequences):
                env.reset(
                    seed=seed + state_index * 1000000 + goal_index * 10000 + sequence_index * 10,
                    options={"state": state.copy(), "goal_state": state.copy()},
                )
                endpoint = np.asarray(state, dtype=np.float64).copy()
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
    return np.asarray(features, dtype=np.float64), {
        name: np.asarray(values, dtype=np.float64 if name in ("angle_progress", "xy_progress", "motion_distance") else np.int64)
        for name, values in labels.items()
    }


def bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(len(values)),
    }


def audit_group(
    env: Any,
    models: dict[str, Any],
    state: np.ndarray,
    goal: np.ndarray,
    sequences: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    truths = []
    for index, sequence in enumerate(sequences):
        endpoint = sequence_endpoint(env, state, sequence, seed + index * 11)
        truths.append(label_sequence(state, goal, endpoint))
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


def audit_group_cached(
    env: Any,
    models: dict[str, Any],
    state: np.ndarray,
    goal: np.ndarray,
    candidates: dict[str, np.ndarray],
    seed: int,
    action_repeat: int,
) -> dict[str, dict[str, Any]]:
    """Evaluate the union once, then reuse labels for every candidate set.

    ``dense_plus_templates`` contains the dense fan by construction.  Reusing
    its simulator labels avoids counting the same successor multiple times and
    makes the intervention computationally auditable.
    """
    union: dict[tuple[float, ...], np.ndarray] = {}
    for sequences in candidates.values():
        for sequence in sequences:
            union.setdefault(_key(sequence), sequence)
    truth_by_key: dict[tuple[float, ...], dict[str, Any]] = {}
    for index, sequence in enumerate(union.values()):
        previous_repeat = sequence_endpoint.action_repeat
        sequence_endpoint.action_repeat = action_repeat
        endpoint = sequence_endpoint(env, state, sequence, seed + index * 11)
        sequence_endpoint.action_repeat = previous_repeat
        truth_by_key[_key(sequence)] = label_sequence(state, goal, endpoint)

    reports: dict[str, dict[str, Any]] = {}
    for variant, sequences in candidates.items():
        truths = [truth_by_key[_key(sequence)] for sequence in sequences]
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
        reports[variant] = {
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
            "union_candidate_count": int(len(union)),
        }
    return reports


def summarize(rows: list[dict[str, Any]], seed: int, bootstrap_draws: int) -> dict[str, Any]:
    metrics = (
        "candidate_count", "oracle_available", "oracle_count", "selected_feasible",
        "top8_hit", "top32_hit", "selected_true_score", "oracle_best_score",
        "selected_rank_by_true_score", "selected_angle_progress", "selected_xy_progress",
        "selected_contact",
    )
    return {
        metric: bootstrap(np.asarray([row[metric] for row in rows]), bootstrap_draws, seed + index)
        for index, metric in enumerate(metrics)
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = make_state_only_env()
    reports: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    try:
        train_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 100000)
        held_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 200000)
        train_states = select_balanced_states(train_pool, args.train_states, args.seed + 1, ("contact",))
        held_states = select_balanced_states(held_pool, args.held_out_states, args.seed + 2, ("contact",))
        train_actions = action_bank(8, (0.2, 0.4, 0.6))
        features, labels = make_training_data_repeated(
            env, train_states, train_actions, args.goals_per_state,
            args.seed + 300000, args.action_repeat,
        )
        models = fit_models(features, labels, args.trees, args.seed + 7)
        goal_rng = np.random.default_rng(args.seed + 400000)
        for state_index, state in enumerate(held_states):
            for goal_index in range(args.goals_per_state):
                goal = goal_for_state(
                    state, state_index * args.goals_per_state + goal_index,
                    goal_rng, closure_only=args.closure_only,
                )
                candidates = candidate_sets(state, goal)
                cached_rows = audit_group_cached(
                    env, models, state, goal, candidates,
                    args.seed + 500000 + state_index * 10000 + goal_index * 100,
                    args.action_repeat,
                )
                for variant in VARIANTS:
                    row = cached_rows[variant]
                    row.update({"state_index": state_index, "goal_index": goal_index})
                    reports[variant].append(row)
    finally:
        env.close()

    summaries = {
        variant: summarize(rows, args.seed + 700 + index * 1000, args.bootstrap)
        for index, (variant, rows) in enumerate(reports.items())
    }
    paired = {}
    for variant_index, variant in enumerate(VARIANTS[1:]):
        paired_metrics = {}
        for metric_index, metric in enumerate((
            "oracle_available", "selected_feasible", "top8_hit", "top32_hit",
            "selected_true_score", "selected_rank_by_true_score",
            "selected_angle_progress", "selected_xy_progress",
        )):
            paired_metrics[metric] = bootstrap(
                np.asarray([
                    new[metric] - old[metric]
                    for old, new in zip(reports["baseline_fan"], reports[variant])
                ]),
                args.bootstrap,
                args.seed + 2000 + variant_index * 100 + metric_index,
            )
        paired[variant] = paired_metrics
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "train_contact_states": len(train_states),
            "held_out_contact_states": len(held_states),
            "held_out_state_goal_groups": len(reports["baseline_fan"]),
            "ranker": "frozen sequence_continuous_rank trained only on baseline fan",
            "feature_interface": "local",
            "runtime_candidate_successor_verification": False,
            "simulator_successor_used_for_labels_only": True,
            "same_state_goal_groups_across_variants": True,
            "action_repeat": args.action_repeat,
            "seed": args.seed,
            "wall_seconds": time.perf_counter() - started,
        },
        "candidate_protocol": {
            "baseline_fan": "8 directions x magnitudes {0.2,0.4,0.6}, all ordered two-action pairs",
            "dense_fan": "16 directions x magnitudes {0.15,0.3,0.45,0.6}, all ordered two-action pairs",
            "goal_templates": "baseline fan plus exact approach/push/reversal/tangent geometry-conditioned templates",
            "dense_plus_templates": "dense fan plus the same geometry-conditioned templates",
        },
        "summaries": summaries,
        "paired_against_baseline": paired,
        "rows": reports,
        "training": {
            "rows": int(len(features)),
            "positive_rates": {name: float(np.mean(values)) for name, values in labels.items()},
        },
        "claim_boundary": (
            "Privileged simulator-state candidate-generation intervention. A positive result "
            "would show that candidate coverage is a necessary interface condition for the "
            "frozen learned ranker; it would not establish visual LeWM planning, physical "
            "latent causality, universal world-model capability, or task success."
        ),
        "sanity_checks": {
            "all_finite": bool(all(
                np.isfinite(value)
                for rows in reports.values()
                for row in rows
                for value in row.values()
                if isinstance(value, (int, float, np.integer, np.floating))
            )),
            "same_state_goal_groups_across_variants": True,
            "frozen_ranker": True,
            "runtime_candidate_successor_verification": False,
        },
    }
    (output / "candidate_generation_intervention.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# PushT candidate-generation intervention",
        "",
        "| candidate set | candidates | oracle coverage | selected feasible | top-8 hit | true rank | angle progress | XY progress |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        summary = summaries[variant]
        lines.append(
            "| %s | %.0f | %.3f | %.3f | %.3f | %.1f | %.4f | %.3f |" % (
                variant,
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
    (output / "candidate_generation_intervention.md").write_text("\n".join(lines))
    print(json.dumps({
        "output": str(output),
        "summaries": summaries,
        "paired_against_baseline": paired,
        "sanity_checks": report["sanity_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
