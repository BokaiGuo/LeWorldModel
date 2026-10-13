"""Held-out ranking audit for horizon-matched PushT mode selectors.

The closed-loop verifier can hide whether a failure came from candidate
ranking or from executing a correctly ranked action under contact-mode drift.
This audit evaluates the same one-step candidate bank on root-disjoint states,
using simulator successors only for held-out labels and never for selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_pusht_contact_mode_effective_motion import select_balanced_states  # noqa: E402
from experiments.run_pusht_contact_history_velocity_diagnostic import collect_trajectory_data, mode_for_state  # noqa: E402
from experiments.run_pusht_learned_sequence_verifier import (  # noqa: E402
    choose_sequence,
    fit_pairwise_angle_ranker,
    fit_models,
    goal_for_state,
    label_sequence,
    make_sequences,
    make_training_data,
)
from experiments.run_pusht_object_centric_learned_planner import make_state_only_env, torque_action_bank  # noqa: E402


VARIANTS = ("sequence_mode_receding", "sequence_mode_pairwise_receding", "sequence_mode_pairwise_staged_receding")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-states", type=int, default=24)
    parser.add_argument("--held-out-states", type=int, default=16)
    parser.add_argument("--goals-per-state", type=int, default=3)
    parser.add_argument("--pool-transitions", type=int, default=4096)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[0.2, 0.4, 0.6])
    parser.add_argument("--trees", type=int, default=96)
    parser.add_argument("--seed", type=int, default=9173)
    return parser.parse_args()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(np.float64)
    ry = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort").astype(np.float64)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def endpoint(env, state: np.ndarray, action: np.ndarray, seed: int) -> np.ndarray:
    env.reset(seed=seed, options={"state": state.copy(), "goal_state": state.copy()})
    observation, _, _, _, _ = env.step(action)
    return np.asarray(observation["state"], dtype=np.float64).copy()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = make_state_only_env()
    actions = torque_action_bank(args.directions, magnitudes=tuple(args.magnitudes))
    sequences = make_sequences(actions, length=1)
    try:
        train_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 100000)
        held_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 200000)
        train_states = select_balanced_states(train_pool, args.train_states, args.seed + 1, ("contact",))
        held_states = select_balanced_states(held_pool, args.held_out_states, args.seed + 2, ("contact",))
        train_features, train_labels = make_training_data(
            env, train_states, actions, args.goals_per_state,
            args.seed + 300000, "local", "path", sequence_length=1,
        )
        models = fit_models(train_features, train_labels, args.trees, args.seed + 7)
        models["angle_pairwise"] = fit_pairwise_angle_ranker(
            train_features, train_labels["angle_progress"], len(actions), args.trees, args.seed + 700,
        )
        rng = np.random.default_rng(args.seed + 400000)
        rows = []
        for state_index, state in enumerate(held_states):
            for goal_index in range(args.goals_per_state):
                goal = goal_for_state(state, state_index * args.goals_per_state + goal_index, rng, closure_only=True)
                endpoints = [endpoint(env, state, action[0], args.seed + 500000 + state_index * 10000 + goal_index * 100 + action_index) for action_index, action in enumerate(sequences)]
                truths = [label_sequence(state, goal, successor) for successor in endpoints]
                true_angle = np.asarray([row["angle_progress"] for row in truths], dtype=np.float64)
                true_xy = np.asarray([row["xy_progress"] for row in truths], dtype=np.float64)
                best_angle = int(np.argmax(true_angle))
                best_xy = int(np.argmax(true_xy))
                for variant in VARIANTS:
                    selected, info = choose_sequence(
                        variant, models, state, goal, sequences, "local",
                        mode_zero_policy="nonzero",
                    )
                    predicted_order = np.asarray(info["predicted_order"], dtype=np.int64)
                    # ``predicted_order`` is best-first.  Encode best as the
                    # largest score so that Spearman has the usual positive
                    # interpretation: higher predicted score means larger
                    # held-out angle progress.
                    predicted_score = np.empty(len(sequences), dtype=np.float64)
                    predicted_score[predicted_order] = -np.arange(len(sequences), dtype=np.float64)
                    rank = int(np.flatnonzero(np.argsort(-true_angle, kind="mergesort") == selected)[0] + 1)
                    rows.append({
                        "variant": variant,
                        "state_index": state_index,
                        "goal_index": goal_index,
                        "selected": selected,
                        "best_angle": best_angle,
                        "best_xy": best_xy,
                        "selected_angle_top1": int(selected == best_angle),
                        "selected_xy_top1": int(selected == best_xy),
                        "selected_angle_rank": rank,
                        "selected_angle_progress": float(true_angle[selected]),
                        "selected_xy_progress": float(true_xy[selected]),
                        "selected_contact": int(truths[selected]["contact"]),
                        "heldout_angle_spearman": spearman(predicted_score, true_angle),
                    })
    finally:
        env.close()
    summaries = {}
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        summaries[variant] = {
            metric: float(np.mean([row[metric] for row in subset]))
            for metric in ("selected_angle_top1", "selected_xy_top1", "selected_angle_rank", "selected_angle_progress", "selected_xy_progress", "selected_contact", "heldout_angle_spearman")
        }
        summaries[variant]["n"] = len(subset)
    report = {
        "protocol": {"train_states": len(train_states), "held_out_states": len(held_states), "goals_per_state": args.goals_per_state, "primitive_action_count": len(actions), "decision_horizon": 1, "runtime_candidate_successor_verification": False},
        "summaries": summaries,
        "rows": rows,
        "claim_boundary": "Root-disjoint one-step candidate-ranking audit in privileged simulator state. Successors create held-out labels only; this is not RGB visual planning or physical causal identification.",
        "sanity_checks": {"all_finite": bool(all(np.isfinite(value) for row in rows for value in row.values() if isinstance(value, (int, float)))), "same_candidates": True, "runtime_candidate_successor_verification": False},
    }
    (output / "horizon_matched_ranking.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# Horizon-matched held-out ranking audit", "", "| variant | angle top-1 | XY top-1 | angle rank | angle rho |", "|---|---:|---:|---:|---:|"]
    for variant, summary in summaries.items():
        lines.append(f"| {variant} | {summary['selected_angle_top1']:.3f} | {summary['selected_xy_top1']:.3f} | {summary['selected_angle_rank']:.2f} | {summary['heldout_angle_spearman']:.3f} |")
    lines += ["", report["claim_boundary"], ""]
    (output / "horizon_matched_ranking.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "summaries": summaries, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
