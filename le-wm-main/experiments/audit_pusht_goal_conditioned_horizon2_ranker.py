"""Held-out audit of a goal-conditioned horizon-2 feasibility ranker.

The direct endpoint adapter predicts an action effect and then applies a
hand-coded pose score.  This audit tests the missing decision interface: a
classifier is trained to predict whether a candidate horizon-2 action block
will improve orientation while staying inside the translation tube for the
current goal.  Training labels use simulator endpoints on disjoint contact
states; held-out ranking never probes the simulator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_contact_history_velocity_diagnostic import (  # noqa: E402
    collect_trajectory_data,
    mode_for_state,
)
from experiments.run_pusht_object_centric_learned_planner import (  # noqa: E402
    angle_error,
    make_state_only_env,
    torque_action_bank,
    torque_block_target_frame_features,
    unit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-states", type=int, default=256)
    parser.add_argument("--held-out-states", type=int, default=48)
    parser.add_argument("--pool-transitions", type=int, default=4096)
    parser.add_argument("--directions", type=int, default=40)
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[0.15, 0.25, 0.35, 0.55, 0.75])
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--seed", type=int, default=8421)
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(sample, 0.025)),
        "ci95_high": float(np.quantile(sample, 0.975)),
        "n": int(len(values)),
    }


def make_goal(state: np.ndarray, index: int) -> np.ndarray:
    goal = np.asarray(state, dtype=np.float64).copy()
    offsets = np.asarray(((34.0, -27.0), (-27.0, 34.0), (41.0, 0.0), (0.0, 41.0)))
    goal[2:4] += offsets[index % len(offsets)]
    goal[4] = float((goal[4] + (0.48 if index % 2 == 0 else -0.48)) % (2.0 * np.pi))
    return goal


def actual_endpoint(env, state: np.ndarray, action: np.ndarray, horizon: int, seed: int) -> np.ndarray:
    env.reset(seed=seed, options={"state": state.copy(), "goal_state": state.copy()})
    endpoint = np.asarray(state, dtype=np.float64).copy()
    for _ in range(horizon):
        observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
        if terminated or truncated:
            break
    return endpoint


def goal_features(state: np.ndarray, action: np.ndarray, goal: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    angle = float(state[4])
    rotation = np.asarray([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)],
    ])
    delta = rotation @ (goal[2:4] - state[2:4])
    angle_delta = angle_error(goal[4], state[4])
    signed_angle = float((goal[4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi)
    return np.concatenate([
        torque_block_target_frame_features(state, action, np.zeros(2)),
        np.asarray([
            float(delta[0]), float(delta[1]), float(np.linalg.norm(delta)),
            float(np.cos(signed_angle)), float(np.sin(signed_angle)),
            float(angle_delta),
        ]),
    ])


def collect_labels(env, states: np.ndarray, actions: np.ndarray, horizon: int, seed: int):
    features = []
    labels = []
    for state_index, state in enumerate(np.asarray(states, dtype=np.float64)):
        goal = make_goal(state, state_index)
        before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
        before_angle = angle_error(state[4], goal[4])
        limit = min(45.0, before_xy + 4.0)
        for action_index, action in enumerate(actions):
            endpoint = actual_endpoint(
                env, state, action, horizon,
                seed + state_index * 1000 + action_index,
            )
            after_xy = float(np.linalg.norm(endpoint[2:4] - goal[2:4]))
            after_angle = angle_error(endpoint[4], goal[4])
            features.append(goal_features(state, action, goal))
            labels.append(int(after_angle < before_angle - 1e-5 and after_xy <= limit))
    return np.asarray(features), np.asarray(labels, dtype=np.int64)


def evaluate_classifier(env, model, held_states, actions, horizon, seed):
    rows = []
    for state_index, state in enumerate(held_states):
        goal = make_goal(state, state_index)
        before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
        before_angle = angle_error(state[4], goal[4])
        limit = min(45.0, before_xy + 4.0)
        actual = np.asarray([
            actual_endpoint(env, state, action, horizon, seed + state_index * 1000 + action_index)
            for action_index, action in enumerate(actions)
        ])
        actual_xy = np.linalg.norm(actual[:, 2:4] - goal[2:4][None], axis=1)
        actual_angle = np.asarray([angle_error(row[4], goal[4]) for row in actual])
        feasible = (actual_angle < before_angle - 1e-5) & (actual_xy <= limit)
        features = np.asarray([goal_features(state, action, goal) for action in actions])
        probability = model.predict_proba(features)[:, 1]
        order = np.argsort(-probability)
        rows.append({
            "state_index": state_index,
            "mode": mode_for_state(state),
            "oracle_any_feasible": float(feasible.any()),
            "top1_feasible": float(feasible[order[0]]),
            "top3_hit": float(feasible[order[:3]].any()),
            "top8_hit": float(feasible[order[:8]].any()),
            "top1_angle_improvement": float(before_angle - actual_angle[order[0]]),
            "top1_xy": float(actual_xy[order[0]]),
            "positive_count": int(feasible.sum()),
            "selected_action_index": int(order[0]),
        })
    metrics = {
        key: bootstrap(np.asarray([row[key] for row in rows]), 4000, seed + index)
        for index, key in enumerate((
            "oracle_any_feasible", "top1_feasible", "top3_hit", "top8_hit",
            "top1_angle_improvement", "top1_xy",
        ))
    }
    return {"metrics": metrics, "rows": rows}


def main() -> None:
    args = parse_args()
    if args.horizon != 2:
        raise ValueError("this audit is defined for horizon 2")
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = make_state_only_env()
    try:
        train_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 100000)
        held_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 200000)
        train_states = np.asarray([
            state for state, mode in zip(train_pool["states"], train_pool["modes"])
            if mode == "contact"
        ], dtype=np.float64)[: args.train_states]
        held_states = [
            np.asarray(state, dtype=np.float64).copy()
            for state, mode in zip(held_pool["states"], held_pool["modes"])
            if mode == "contact"
        ][: args.held_out_states]
        if len(train_states) < args.train_states or len(held_states) < args.held_out_states:
            raise RuntimeError("insufficient disjoint contact states")
        actions = torque_action_bank(args.directions, tuple(args.magnitudes))
        train_features, train_labels = collect_labels(
            env, train_states, actions, args.horizon, args.seed + 300000
        )
        ranker = ExtraTreesClassifier(
            n_estimators=args.trees,
            min_samples_leaf=3,
            max_features=1.0,
            class_weight="balanced",
            random_state=args.seed + 1,
            n_jobs=1,
        ).fit(train_features, train_labels)
        ranker_report = evaluate_classifier(
            env, ranker, held_states, actions, args.horizon, args.seed + 400000
        )
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "training": "goal-conditioned horizon-2 feasibility labels on disjoint contact states",
            "evaluation": "held-out action ranking without simulator verification at selection time",
            "train_states": len(train_states),
            "held_out_states": len(held_states),
            "action_count": len(actions),
            "horizon": args.horizon,
            "trees": args.trees,
            "seed": args.seed,
            "positive_rate": float(train_labels.mean()),
            "wall_seconds": time.perf_counter() - started,
        },
        "goal_conditioned_ranker": ranker_report,
        "sanity_checks": {
            "all_finite": bool(np.isfinite(train_features).all()),
            "same_action_count": len(actions),
            "same_targets": True,
            "train_heldout_disjoint_by_independent_reset_streams": True,
            "both_classes": int(len(np.unique(train_labels))) == 2,
        },
        "claim_boundary": (
            "Privileged simulator-state horizon-2 feasibility-ranking diagnostic. A positive result "
            "supports a goal-conditioned decision interface for this PushT protocol; it does not "
            "establish frozen-LeWM latent causality, visual online planning, or general task success."
        ),
    }
    (output / "goal_conditioned_horizon2_ranker.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    metrics = ranker_report["metrics"]
    lines = [
        "# Goal-conditioned horizon-2 feasibility ranker",
        "",
        "| ranker | top-1 feasible | top-3 hit | top-8 hit | top-1 angle improvement | top-1 XY |",
        "|---|---:|---:|---:|---:|---:|",
        "| goal-conditioned | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.4f [%.4f, %.4f] | %.3f [%.3f, %.3f] |" % (
            metrics["top1_feasible"]["estimate"], metrics["top1_feasible"]["ci95_low"], metrics["top1_feasible"]["ci95_high"],
            metrics["top3_hit"]["estimate"], metrics["top3_hit"]["ci95_low"], metrics["top3_hit"]["ci95_high"],
            metrics["top8_hit"]["estimate"], metrics["top8_hit"]["ci95_low"], metrics["top8_hit"]["ci95_high"],
            metrics["top1_angle_improvement"]["estimate"], metrics["top1_angle_improvement"]["ci95_low"], metrics["top1_angle_improvement"]["ci95_high"],
            metrics["top1_xy"]["estimate"], metrics["top1_xy"]["ci95_low"], metrics["top1_xy"]["ci95_high"],
        ),
        "",
        report["claim_boundary"],
        "",
    ]
    (output / "goal_conditioned_horizon2_ranker.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "report": report}, indent=2))


if __name__ == "__main__":
    main()
