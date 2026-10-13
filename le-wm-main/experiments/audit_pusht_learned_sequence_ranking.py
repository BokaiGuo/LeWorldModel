"""Held-out ranking audit for the learned PushT macro-action selectors.

The closed-loop runner measures final pose, but does not by itself identify
whether the selector ranked a useful macro-action near the top.  This audit
fits the same models on an independent training-state stream and evaluates
all ordered two-action sequences on disjoint held-out state--goal groups.
Successor simulation is used only to create held-out audit labels; the model
ranking receives current state, goal, and the macro-action features only.
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

from experiments.run_pusht_contact_history_velocity_diagnostic import (  # noqa: E402
    collect_trajectory_data,
    mode_for_state,
)
from experiments.run_pusht_learned_sequence_verifier import (  # noqa: E402
    CONTINUOUS_TARGETS,
    choose_sequence,
    feature_vector,
    fit_models,
    goal_for_state,
    label_sequence,
    make_sequences,
    make_training_data,
)
from experiments.run_pusht_object_centric_learned_planner import (  # noqa: E402
    make_state_only_env,
    torque_action_bank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-states", type=int, default=64)
    parser.add_argument("--held-out-states", type=int, default=32)
    parser.add_argument("--pool-transitions", type=int, default=4096)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[0.2, 0.4, 0.6])
    parser.add_argument("--goals-per-state", type=int, default=4)
    parser.add_argument("--trees", type=int, default=96)
    parser.add_argument("--feature-interface", choices=("local", "global_wall"), default="local")
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261091)
    return parser.parse_args()


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


def sequence_endpoint(env, state: np.ndarray, sequence: np.ndarray, seed: int) -> np.ndarray:
    env.reset(seed=seed, options={"state": state.copy(), "goal_state": state.copy()})
    endpoint = np.asarray(state, dtype=np.float64).copy()
    for action in sequence:
        observation, _, terminated, truncated, _ = env.step(action)
        endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
        if terminated or truncated:
            break
    return endpoint


def audit_group(
    env,
    models: dict[str, Any],
    variant: str,
    state: np.ndarray,
    goal: np.ndarray,
    sequences: np.ndarray,
    seed: int,
    feature_interface: str,
) -> dict[str, Any]:
    truths = []
    for index, sequence in enumerate(sequences):
        endpoint = sequence_endpoint(env, state, sequence, seed + index * 11)
        truths.append(label_sequence(state, goal, endpoint))
    selected, info = choose_sequence(variant, models, state, goal, sequences, feature_interface)
    order = np.asarray(info["predicted_order"], dtype=np.int64)
    joint = np.asarray([row["joint"] for row in truths], dtype=np.int64)
    actual_score = np.asarray([
        -row["angle_progress"]
        + 0.02 * max(-row["xy_progress"], 0.0)
        + 0.35 * (1.0 - row["contact"])
        + 0.10 * max(0.5 - row["motion_distance"], 0.0)
        for row in truths
    ], dtype=np.float64)
    return {
        "variant": variant,
        "oracle_available": int(joint.any()),
        "oracle_count": int(joint.sum()),
        "selected_feasible": int(joint[selected]),
        "top8_hit": int(joint[order[:8]].any()),
        "top32_hit": int(joint[order[:32]].any()),
        "selected_true_score": float(actual_score[selected]),
        "oracle_best_score": float(actual_score.min()),
        "selected_rank_by_true_score": int(np.flatnonzero(np.argsort(actual_score) == selected)[0] + 1),
        "selected_angle_progress": float(truths[selected]["angle_progress"]),
        "selected_xy_progress": float(truths[selected]["xy_progress"]),
        "selected_contact": int(truths[selected]["contact"]),
        "predicted_order": order.tolist(),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = make_state_only_env()
    actions = torque_action_bank(args.directions, magnitudes=tuple(args.magnitudes))
    sequences = make_sequences(actions)
    try:
        train_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 100000)
        held_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 200000)
        train_states = __import__(
            "experiments.audit_pusht_contact_mode_effective_motion",
            fromlist=["select_balanced_states"],
        ).select_balanced_states(train_pool, args.train_states, args.seed + 1, ("contact",))
        held_states = __import__(
            "experiments.audit_pusht_contact_mode_effective_motion",
            fromlist=["select_balanced_states"],
        ).select_balanced_states(held_pool, args.held_out_states, args.seed + 2, ("contact",))
        features, labels = make_training_data(
            env, train_states, actions, args.goals_per_state,
            args.seed + 300000, args.feature_interface,
        )
        models = fit_models(features, labels, args.trees, args.seed + 7)
        rng = np.random.default_rng(args.seed + 400000)
        reports = {name: [] for name in ("sequence_soft_contact", "sequence_continuous_rank")}
        for state_index, state in enumerate(held_states):
            for goal_index in range(args.goals_per_state):
                goal = goal_for_state(
                    state, state_index * args.goals_per_state + goal_index,
                    rng, closure_only=True,
                )
                for variant in reports:
                    reports[variant].append(audit_group(
                        env, models, variant, state, goal, sequences,
                        args.seed + 500000 + state_index * 10000 + goal_index * 100,
                        args.feature_interface,
                    ))
    finally:
        env.close()

    summaries = {}
    for index, (variant, rows) in enumerate(reports.items()):
        summaries[variant] = {
            metric: bootstrap(
                np.asarray([row[metric] for row in rows]), args.bootstrap, args.seed + 600 + index * 100 + metric_index,
            )
            for metric_index, metric in enumerate((
                "oracle_available", "selected_feasible", "top8_hit", "top32_hit",
                "selected_rank_by_true_score", "selected_angle_progress", "selected_xy_progress",
            ))
        }
        summaries[variant]["rows"] = rows
    continuous = reports["sequence_continuous_rank"]
    binary = reports["sequence_soft_contact"]
    paired = {
        metric: bootstrap(
            np.asarray([new[metric] - old[metric] for old, new in zip(binary, continuous)]),
            args.bootstrap, args.seed + 1000 + metric_index,
        )
        for metric_index, metric in enumerate((
            "selected_feasible", "top8_hit", "top32_hit", "selected_rank_by_true_score",
            "selected_angle_progress", "selected_xy_progress",
        ))
    }
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "train_contact_states": len(train_states),
            "held_out_contact_states": len(held_states),
            "held_out_state_goal_groups": len(continuous),
            "ordered_two_action_sequence_count": len(sequences),
            "feature_interface": args.feature_interface,
            "runtime_candidate_successor_verification": False,
            "wall_seconds": time.perf_counter() - started,
        },
        "summaries": summaries,
        "paired_continuous_minus_binary": paired,
        "claim_boundary": (
            "Held-out privileged simulator-state macro-action ranking audit. "
            "The audit tests decision ranking and candidate coverage; it does "
            "not establish visual planning, physical latent causality, or "
            "universal world-model capability."
        ),
        "sanity_checks": {
            "all_finite": bool(all(
                np.isfinite(row[key])
                for rows in reports.values()
                for row in rows
                for key in (
                    "selected_true_score", "oracle_best_score", "selected_angle_progress",
                    "selected_xy_progress",
                )
            )),
            "same_groups_and_sequences": True,
            "runtime_candidate_successor_verification": False,
        },
    }
    (output / "learned_sequence_ranking_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# Learned sequence ranking audit", "",
        "| selector | oracle available | selected feasible | top-8 hit | top-32 hit | true-rank | angle progress | XY progress |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in reports:
        m = summaries[variant]
        lines.append(
            "| %s | %.3f | %.3f | %.3f | %.3f | %.2f | %.4f | %.3f |" % (
                variant,
                m["oracle_available"]["estimate"],
                m["selected_feasible"]["estimate"],
                m["top8_hit"]["estimate"],
                m["top32_hit"]["estimate"],
                m["selected_rank_by_true_score"]["estimate"],
                m["selected_angle_progress"]["estimate"],
                m["selected_xy_progress"]["estimate"],
            )
        )
    lines += ["", report["claim_boundary"], ""]
    (output / "learned_sequence_ranking_audit.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "summaries": {k: {m: v for m, v in s.items() if m != "rows"} for k, s in summaries.items()}, "paired": paired, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
