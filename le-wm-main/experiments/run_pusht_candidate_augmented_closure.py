"""Closed-loop test of the candidate-augmented learned pose interface.

The offline training-shift audit found that a dense action fan becomes useful
only when the continuous ranker is trained on the same candidate distribution.
This runner tests that intervention in receding privileged state closure with
matched initial states and goals.  There is no simulator successor check at
selection time; simulator transitions are used only by the offline training
labels and by the final environment execution.
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
)
from experiments.audit_pusht_candidate_training_shift import fit_sampled_ranker  # noqa: E402
from experiments.audit_pusht_contact_mode_effective_motion import (  # noqa: E402
    goal_for_state,
    select_balanced_states,
)
from experiments.run_pusht_contact_history_velocity_diagnostic import (  # noqa: E402
    collect_trajectory_data,
)
from experiments.run_pusht_direct_pose_decision_ablation import angle_error  # noqa: E402
from experiments.run_pusht_learned_sequence_verifier import choose_sequence  # noqa: E402
from experiments.run_pusht_learned_sequence_verifier import feature_vector, positive_probability  # noqa: E402
from experiments.run_pusht_object_centric_learned_planner import make_state_only_env  # noqa: E402


VARIANTS = (
    "baseline_ranker_baseline_fan",
    "baseline_ranker_dense_fan",
    "dense_ranker_dense_fan",
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
    parser.add_argument("--trees", type=int, default=96)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--closure-cycles", type=int, default=8)
    parser.add_argument(
        "--score-mode", choices=("angle_primary", "margin_balanced"), default="angle_primary",
        help="continuous utility: historical angle-primary score or success-margin balanced score",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261098)
    return parser.parse_args()


def strict_success(state: np.ndarray, goal: np.ndarray) -> bool:
    return bool(
        np.linalg.norm(np.asarray(state)[2:4] - np.asarray(goal)[2:4]) < 20.0
        and angle_error(state[4], goal[4]) < np.pi / 9.0
    )


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


def run_episode(
    env: Any,
    models: dict[str, Any],
    candidate_variant: str,
    state: np.ndarray,
    goal: np.ndarray,
    action_repeat: int,
    cycles: int,
    seed: int,
    score_mode: str,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed, options={"state": state.copy(), "goal_state": goal.copy()})
    current = np.asarray(observation["state"], dtype=np.float64).copy()
    trace = []
    for cycle in range(cycles):
        candidates = candidate_sets(current, goal)
        sequences = candidates[candidate_variant]
        if score_mode == "angle_primary":
            selected, info = choose_sequence("sequence_continuous_rank", models, current, goal, sequences, "local")
        else:
            features = np.asarray([
                feature_vector(current, sequence[0], sequence[1], np.zeros(2), goal, "local")
                for sequence in sequences
            ], dtype=np.float64)
            predicted_angle = np.asarray(models["angle_progress"].predict(features), dtype=np.float64)
            predicted_xy = np.asarray(models["xy_progress"].predict(features), dtype=np.float64)
            predicted_motion = np.asarray(models["motion_distance"].predict(features), dtype=np.float64)
            predicted_contact = positive_probability(models["contact"], features)
            # Normalize both progress channels by the strict PushT margins.
            # This treats one threshold-sized angle improvement and one
            # threshold-sized XY improvement as comparable, while retaining
            # the learned contact/motion safeguards.
            score = (
                -predicted_angle / (np.pi / 9.0)
                -0.05 * predicted_xy
                +0.35 * (1.0 - predicted_contact)
                +0.10 * np.maximum(0.5 - predicted_motion, 0.0)
            )
            order = np.argsort(score, kind="mergesort")
            selected = int(order[0])
            info = {
                "selected_score": float(score[selected]),
                "selected_angle_progress": float(predicted_angle[selected]),
                "selected_xy_progress": float(predicted_xy[selected]),
            }
        selected_sequence = sequences[selected]
        terminated = truncated = False
        for action in selected_sequence:
            for _ in range(action_repeat):
                observation, _, terminated, truncated, _ = env.step(action)
                current = np.asarray(observation["state"], dtype=np.float64).copy()
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
        trace.append({
            "cycle": cycle,
            "candidate_count": len(sequences),
            "selected": int(selected),
            "selected_score": float(info["selected_score"]),
            "selected_angle_progress": float(info["selected_angle_progress"]),
            "selected_xy_progress": float(info["selected_xy_progress"]),
            "block_xy": float(np.linalg.norm(current[2:4] - goal[2:4])),
            "angle_error": float(angle_error(current[4], goal[4])),
        })
        if strict_success(current, goal) or terminated or truncated:
            break
    return {
        "final_xy": float(np.linalg.norm(current[2:4] - goal[2:4])),
        "final_angle": float(angle_error(current[4], goal[4])),
        "strict_success": int(strict_success(current, goal)),
        "cycles": len(trace),
        "trace": trace,
        "final_state": current.tolist(),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = make_state_only_env()
    rows: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
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
                specs = {
                    "baseline_ranker_baseline_fan": (baseline_models, "baseline_fan"),
                    "baseline_ranker_dense_fan": (baseline_models, "dense_fan"),
                    "dense_ranker_dense_fan": (dense_models, "dense_fan"),
                }
                for variant, (models, candidate_variant) in specs.items():
                    row = run_episode(
                        env, models, candidate_variant, state, goal, args.action_repeat,
                        args.closure_cycles, args.seed + 600000 + state_index * 10000 + goal_index * 100,
                        args.score_mode,
                    )
                    row.update({"state_index": state_index, "goal_index": goal_index, "variant": variant})
                    rows[variant].append(row)
    finally:
        env.close()

    summaries = {}
    for index, (variant, values) in enumerate(rows.items()):
        summaries[variant] = {
            metric: bootstrap(np.asarray([row[metric] for row in values]), args.bootstrap, args.seed + 800 + index * 100 + metric_index)
            for metric_index, metric in enumerate(("strict_success", "final_xy", "final_angle", "cycles"))
        }
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "train_contact_states": len(train_states),
            "held_out_contact_states": len(held_states),
            "held_out_episodes": len(rows[VARIANTS[0]]),
            "action_repeat": args.action_repeat,
            "closure_cycles": args.closure_cycles,
            "score_mode": args.score_mode,
            "ranker_training": "baseline fan versus dense fan; same training states and goals",
            "runtime_candidate_successor_verification": False,
            "same_initial_states_and_goals_across_variants": True,
            "wall_seconds": time.perf_counter() - started,
        },
        "summaries": summaries,
        "rows": rows,
        "claim_boundary": (
            "Privileged simulator-state learned macro-action closure. A positive result "
            "supports a candidate-distribution-matched interface repair; it does not establish "
            "visual LeWM planning, physical latent causality, universal world-model capability, "
            "or general PushT performance."
        ),
        "sanity_checks": {
            "all_finite": bool(all(
                np.isfinite(value)
                for values in rows.values()
                for row in values
                for value in row.values()
                if isinstance(value, (int, float, np.integer, np.floating))
            )),
            "same_initial_states_and_goals_across_variants": True,
            "runtime_candidate_successor_verification": False,
        },
    }
    (output / "candidate_augmented_closure.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# Candidate-augmented learned PushT closure", "", "| variant | strict success | final XY | final angle | cycles |", "|---|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        summary = summaries[variant]
        lines.append("| %s | %.3f | %.2f | %.3f | %.2f |" % (
            variant,
            summary["strict_success"]["estimate"],
            summary["final_xy"]["estimate"],
            summary["final_angle"]["estimate"],
            summary["cycles"]["estimate"],
        ))
    lines += ["", report["claim_boundary"], ""]
    (output / "candidate_augmented_closure.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "summaries": summaries, "sanity_checks": report["sanity_checks"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
