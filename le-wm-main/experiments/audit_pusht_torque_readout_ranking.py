"""Held-out ranking audit for world-frame versus contact-frame torque readouts.

The current learned torque arm predicts world-frame block ``dx, dy, dtheta``.
This audit tests whether that representation is the source of the ranking
failure by comparing it with a contact-frame model that predicts normal and
tangential block displacement plus angle change.  Both models use the same
balanced intervention data and are evaluated on disjoint genuine contact
states with a complete action fan.  Actual one-step simulator successors are
used only for evaluation, never as input to the learned selector.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_contact_history_velocity_diagnostic import (
    collect_trajectory_data,
    contact_geometry_features,
    feature_vector,
    mode_for_state,
    unit,
)
from experiments.run_pusht_learned_physical_dynamics import state_features
from experiments.run_pusht_object_centric_learned_planner import (
    angle_error,
    torque_action_bank,
)
from experiments.run_pusht_object_centric_learned_planner import (
    collect_contact_intervention_data,
)
from experiments.run_pusht_receding_block_pose_planner import make_state_only_env

gym.logger.min_level = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-transitions", type=int, default=768)
    parser.add_argument("--held-out-states", type=int, default=48)
    parser.add_argument("--pool-transitions", type=int, default=4096)
    parser.add_argument(
        "--ranking-train-states",
        type=int,
        default=96,
        help="disjoint genuine contact states used to supervise goal-conditioned feasibility",
    )
    parser.add_argument("--directions", type=int, default=24)
    parser.add_argument("--trees", type=int, default=96)
    parser.add_argument("--xy-margin", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def contact_frame_features(
    state: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    """Rotation-aware local contact features without absolute XY coordinates."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    previous_action = np.asarray(previous_action, dtype=np.float64)
    relative = state[2:4] - state[:2]
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]])
    block_angle = float(state[4])
    block_axis = np.asarray([np.cos(block_angle), np.sin(block_angle)])
    relative_angle = float(np.arctan2(relative[1], relative[0]) - block_angle)
    velocity = np.asarray(state[5:7], dtype=np.float64)
    return np.asarray([
        float(np.linalg.norm(relative)),
        float(np.dot(action, normal)),
        float(np.dot(action, tangent)),
        float(np.linalg.norm(action)),
        float(np.dot(previous_action, normal)),
        float(np.dot(previous_action, tangent)),
        float(np.dot(velocity, normal)),
        float(np.dot(velocity, tangent)),
        float(np.cos(relative_angle)),
        float(np.sin(relative_angle)),
        float(block_axis[0]),
        float(block_axis[1]),
    ], dtype=np.float64)


def goal_conditioned_features(
    state: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    """Add target-relative geometry to the local contact chart.

    The existing torque readout predicts an action effect without knowing the
    desired pose.  That is sufficient for effect estimation but not for
    ranking: the same tangential impulse can be useful for one target angle
    and harmful for the opposite one.  This feature map keeps the local
    contact representation and adds only current-state goal geometry.
    """
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    relative = state[2:4] - state[:2]
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]])
    goal_delta = goal[2:4] - state[2:4]
    angle_delta = float((goal[4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi)
    return np.concatenate([
        contact_frame_features(state, action, previous_action),
        np.asarray([
            float(np.dot(goal_delta, normal)),
            float(np.dot(goal_delta, tangent)),
            float(np.linalg.norm(goal_delta)),
            float(np.cos(angle_delta)),
            float(np.sin(angle_delta)),
            float(abs(angle_delta)),
        ], dtype=np.float64),
    ])


def fit_readout(
    data: dict[str, np.ndarray],
    variant: str,
    trees: int,
    seed: int,
) -> ExtraTreesRegressor:
    features = []
    targets = []
    for state, action, previous, next_state in zip(
        data["states"], data["actions"], data["previous_actions"], data["next_states"]
    ):
        if variant == "world_frame":
            feature = feature_vector(state, action, previous, "object_centric")
            target = np.asarray([
                next_state[2] - state[2],
                next_state[3] - state[3],
                (next_state[4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi,
            ])
        elif variant == "contact_frame":
            relative = state[2:4] - state[:2]
            normal = unit(relative)
            tangent = np.asarray([-normal[1], normal[0]])
            delta_xy = next_state[2:4] - state[2:4]
            feature = contact_frame_features(state, action, previous)
            target = np.asarray([
                np.dot(delta_xy, normal),
                np.dot(delta_xy, tangent),
                (next_state[4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi,
            ])
        else:
            raise ValueError(f"unknown readout variant: {variant}")
        features.append(feature)
        targets.append(target)
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(np.asarray(features), np.asarray(targets))
    return model


def fit_contact_frame_specialists(
    data: dict[str, np.ndarray],
    trees: int,
    seed: int,
) -> dict[str, ExtraTreesRegressor]:
    """Fit contact-frame readouts separately on current-state modes."""
    specialists = {"fallback": fit_readout(data, "contact_frame", trees, seed)}
    modes = np.asarray(data["modes"])
    for index, mode in enumerate(("free", "approach", "contact")):
        mask = modes == mode
        if int(mask.sum()) < 32:
            specialists[mode] = specialists["fallback"]
            continue
        subset = {key: value[mask] for key, value in data.items()}
        specialists[mode] = fit_readout(subset, "contact_frame", trees, seed + 101 + index)
    return specialists


def predict_effect(
    model: ExtraTreesRegressor,
    variant: str,
    state: np.ndarray,
    action: np.ndarray,
    previous: np.ndarray,
) -> np.ndarray:
    if variant == "world_frame":
        feature = feature_vector(state, action, previous, "object_centric")
        return np.asarray(model.predict(feature[None])[0], dtype=np.float64)
    feature = contact_frame_features(state, action, previous)
    local = np.asarray(model.predict(feature[None])[0], dtype=np.float64)
    relative = state[2:4] - state[:2]
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]])
    return np.asarray([
        local[0] * normal[0] + local[1] * tangent[0],
        local[0] * normal[1] + local[1] * tangent[1],
        local[2],
    ], dtype=np.float64)


def predict_mode_conditioned_effect(
    specialists: dict[str, ExtraTreesRegressor],
    state: np.ndarray,
    action: np.ndarray,
    previous: np.ndarray,
) -> np.ndarray:
    model = specialists.get(mode_for_state(state), specialists["fallback"])
    return predict_effect(model, "contact_frame", state, action, previous)


def make_goal(state: np.ndarray, distance: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    goal = np.asarray(state, dtype=np.float64).copy()
    goal[2:4] += distance * unit(rng.normal(size=2))
    goal[4] = float(state[4] + (0.8 if seed % 2 else -0.8))
    goal[5:7] = 0.0
    return goal


def probe_actual(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    action: np.ndarray,
    index: int,
) -> dict[str, Any]:
    env.reset(seed=index, options={"state": state.copy(), "goal_state": goal.copy()})
    observation, _, _, _, _ = env.step(action)
    after = np.asarray(observation["state"], dtype=np.float64)
    before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
    after_xy = float(np.linalg.norm(after[2:4] - goal[2:4]))
    before_angle = angle_error(state[4], goal[4])
    after_angle = angle_error(after[4], goal[4])
    return {
        "after_xy": after_xy,
        "after_angle": after_angle,
        "angle_improvement": before_angle - after_angle,
        "feasible": bool(
            after_angle < before_angle - 1e-5
            and after_xy <= min(45.0, before_xy + 4.0)
        ),
        "delta": [
            float(after[2] - state[2]),
            float(after[3] - state[3]),
            float((after[4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi),
        ],
    }


def collect_goal_conditioned_rank_data(
    env: gym.Env,
    states: np.ndarray,
    actions: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect labels for a learned action-feasibility ranker.

    The simulator successor is used only to create supervision.  At test
    time the ranker receives the current state, goal, and candidate action,
    but no successor or simulator probe.
    """
    features = []
    labels = []
    for state_index, state in enumerate(np.asarray(states, dtype=np.float64)):
        goal = make_goal(state, 27.0 if state_index % 2 else 40.0, seed + state_index)
        before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
        before_angle = angle_error(state[4], goal[4])
        limit = min(45.0, before_xy + 4.0)
        for action_index, action in enumerate(actions):
            env.reset(
                seed=seed + state_index * 1000 + action_index,
                options={"state": state.copy(), "goal_state": goal.copy()},
            )
            observation, _, _, _, _ = env.step(action)
            after = np.asarray(observation["state"], dtype=np.float64)
            after_xy = float(np.linalg.norm(after[2:4] - goal[2:4]))
            after_angle = angle_error(after[4], goal[4])
            features.append(goal_conditioned_features(
                state, action, np.zeros(2, dtype=np.float64), goal
            ))
            labels.append(float(after_angle < before_angle - 1e-5 and after_xy <= limit))
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def fit_feasibility_ranker(
    features: np.ndarray,
    labels: np.ndarray,
    trees: int,
    seed: int,
) -> ExtraTreesClassifier:
    if len(np.unique(labels)) < 2:
        raise ValueError("goal-conditioned ranker needs both feasible and infeasible labels")
    return ExtraTreesClassifier(
        n_estimators=trees,
        min_samples_leaf=3,
        max_features=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    ).fit(features, labels)


def bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(sample, 0.025)),
        "ci95_high": float(np.quantile(sample, 0.975)),
        "n": int(len(values)),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = make_state_only_env()
    try:
        train = collect_contact_intervention_data(env, args.train_transitions, args.seed)
        models = {
            variant: fit_readout(train, variant, args.trees, args.seed + index)
            for index, variant in enumerate(("world_frame", "contact_frame"))
        }
        mode_specialists = fit_contact_frame_specialists(
            train, args.trees, args.seed + 500
        )
        actions = torque_action_bank(args.directions)
        rank_pool = collect_trajectory_data(
            env, max(args.pool_transitions, args.ranking_train_states), args.seed + 200000
        )
        rank_train_states = np.asarray(
            rank_pool["states"][: args.ranking_train_states], dtype=np.float64
        )
        rank_features, rank_labels = collect_goal_conditioned_rank_data(
            env, rank_train_states, actions, args.seed + 300000
        )
        feasibility_ranker = fit_feasibility_ranker(
            rank_features, rank_labels, args.trees, args.seed + 700
        )
        pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 100000)
        held_out = []
        for state in pool["states"]:
            distance = float(np.linalg.norm(state[:2] - state[2:4]))
            if distance <= 65.0:
                held_out.append(np.asarray(state, dtype=np.float64).copy())
            if len(held_out) >= args.held_out_states:
                break
        if len(held_out) < args.held_out_states:
            raise RuntimeError("insufficient held-out contact states")
        records = []
        for state_index, state in enumerate(held_out):
            goal = make_goal(state, 27.0 if state_index % 2 else 40.0, args.seed + state_index)
            actual = [
                probe_actual(env, state, goal, action, args.seed + state_index * 1000 + action_index)
                for action_index, action in enumerate(actions)
            ]
            oracle_feasible = np.asarray([row["feasible"] for row in actual], dtype=bool)
            oracle_best = float(max(row["angle_improvement"] for row in actual))
            row = {
                "state_index": state_index,
                "mode": mode_for_state(state),
                "contact_distance": float(np.linalg.norm(state[:2] - state[2:4])),
                "before_xy": float(np.linalg.norm(state[2:4] - goal[2:4])),
                "before_angle": angle_error(state[4], goal[4]),
                "oracle_any_feasible": float(oracle_feasible.any()),
                "oracle_best_angle_improvement": oracle_best,
            }
            for variant in models:
                predicted = np.asarray([
                    predict_effect(models[variant], variant, state, action, np.zeros(2))
                    for action in actions
                ])
                predicted_xy = np.linalg.norm(
                    state[2:4][None] + predicted[:, :2] - goal[2:4][None], axis=1
                )
                predicted_angle = np.asarray([
                    angle_error(state[4] + effect[2], goal[4]) for effect in predicted
                ])
                limit = min(45.0, row["before_xy"] + args.xy_margin)
                predicted_feasible = (
                    (predicted_angle < row["before_angle"] - 1e-5)
                    & (predicted_xy <= limit)
                )
                predicted_score = predicted_angle + 0.02 * predicted_xy
                predicted_score[~predicted_feasible] = np.inf
                if np.isfinite(predicted_score).any():
                    ranking = np.argsort(predicted_score)
                else:
                    ranking = np.argsort(
                        predicted_xy + 100.0 * np.maximum(predicted_angle - row["before_angle"], 0.0)
                    )
                actual_feasible = oracle_feasible[ranking]
                top_actual = actual[int(ranking[0])]
                row[f"{variant}_top1_feasible"] = float(actual_feasible[0])
                row[f"{variant}_top3_hit"] = float(actual_feasible[:3].any())
                row[f"{variant}_top8_hit"] = float(actual_feasible[:8].any())
                row[f"{variant}_top1_angle_improvement"] = float(top_actual["angle_improvement"])
                row[f"{variant}_top1_xy"] = float(top_actual["after_xy"])
                true_effects = np.asarray([item["delta"] for item in actual])
                row[f"{variant}_effect_rmse"] = float(np.sqrt(np.mean((predicted - true_effects) ** 2)))
            specialist_predicted = np.asarray([
                predict_mode_conditioned_effect(
                    mode_specialists, state, action, np.zeros(2)
                )
                for action in actions
            ])
            specialist_xy = np.linalg.norm(
                state[2:4][None] + specialist_predicted[:, :2] - goal[2:4][None],
                axis=1,
            )
            specialist_angle = np.asarray([
                angle_error(state[4] + effect[2], goal[4])
                for effect in specialist_predicted
            ])
            specialist_feasible = (
                (specialist_angle < row["before_angle"] - 1e-5)
                & (specialist_xy <= min(45.0, row["before_xy"] + args.xy_margin))
            )
            specialist_score = specialist_angle + 0.02 * specialist_xy
            specialist_score[~specialist_feasible] = np.inf
            if np.isfinite(specialist_score).any():
                specialist_order = np.argsort(specialist_score)
            else:
                specialist_order = np.argsort(
                    specialist_xy + 100.0 * np.maximum(
                        specialist_angle - row["before_angle"], 0.0
                    )
                )
            specialist_actual = oracle_feasible[specialist_order]
            specialist_top = actual[int(specialist_order[0])]
            row["contact_frame_specialist_top1_feasible"] = float(specialist_actual[0])
            row["contact_frame_specialist_top3_hit"] = float(specialist_actual[:3].any())
            row["contact_frame_specialist_top8_hit"] = float(specialist_actual[:8].any())
            row["contact_frame_specialist_top1_angle_improvement"] = float(
                specialist_top["angle_improvement"]
            )
            row["contact_frame_specialist_top1_xy"] = float(specialist_top["after_xy"])
            row["contact_frame_specialist_effect_rmse"] = float(
                np.sqrt(np.mean((specialist_predicted - true_effects) ** 2))
            )
            rank_features_test = np.asarray([
                goal_conditioned_features(
                    state, action, np.zeros(2, dtype=np.float64), goal
                )
                for action in actions
            ])
            rank_probability = feasibility_ranker.predict_proba(rank_features_test)[:, 1]
            rank_order = np.argsort(-rank_probability)
            rank_actual = oracle_feasible[rank_order]
            rank_top = actual[int(rank_order[0])]
            row["goal_conditioned_ranker_top1_feasible"] = float(rank_actual[0])
            row["goal_conditioned_ranker_top3_hit"] = float(rank_actual[:3].any())
            row["goal_conditioned_ranker_top8_hit"] = float(rank_actual[:8].any())
            row["goal_conditioned_ranker_top1_angle_improvement"] = float(
                rank_top["angle_improvement"]
            )
            row["goal_conditioned_ranker_top1_xy"] = float(rank_top["after_xy"])
            records.append(row)
    finally:
        env.close()
    variants = ("world_frame", "contact_frame")
    summary = {}
    for variant_index, variant in enumerate(variants):
        summary[variant] = {
            metric: bootstrap(
                np.asarray([row[f"{variant}_{metric}"] for row in records]),
                args.bootstrap,
                args.seed + 500 + variant_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate((
                "top1_feasible", "top3_hit", "top8_hit",
                "top1_angle_improvement", "top1_xy", "effect_rmse",
            ))
            }
    specialist_metrics = (
        "top1_feasible", "top3_hit", "top8_hit",
        "top1_angle_improvement", "top1_xy", "effect_rmse",
    )
    summary["contact_frame_specialist"] = {
        metric: bootstrap(
            np.asarray([
                row[f"contact_frame_specialist_{metric}"] for row in records
            ]),
            args.bootstrap,
            args.seed + 1100 + metric_index,
        )
        for metric_index, metric in enumerate(specialist_metrics)
    }
    rank_metrics = (
        "top1_feasible", "top3_hit", "top8_hit",
        "top1_angle_improvement", "top1_xy",
    )
    summary["goal_conditioned_ranker"] = {
        metric: bootstrap(
            np.asarray([
                row[f"goal_conditioned_ranker_{metric}"] for row in records
            ]),
            args.bootstrap,
            args.seed + 900 + metric_index,
        )
        for metric_index, metric in enumerate(rank_metrics)
    }
    report = {
        "claim_boundary": (
            "Held-out privileged action-ranking audit. Actual simulator successors are used only "
            "to score rankings; this is not visual online-planning evidence."
        ),
        "protocol": {
            "train_transitions": args.train_transitions,
            "held_out_states": len(records),
            "actions_per_state": len(actions),
            "directions": args.directions,
            "trees": args.trees,
            "ranking_train_states": args.ranking_train_states,
            "ranking_supervision": "goal-conditioned one-step simulator labels on disjoint genuine states",
            "seed": args.seed,
            "wall_seconds": time.perf_counter() - started,
        },
        "summary": summary,
        "records": records,
        "sanity_checks": {
            "all_finite": bool(all(
                np.isfinite(float(value))
                for row in records
                for key, value in row.items()
                if key not in ("state_index", "mode")
            )),
            "same_action_count": len(actions),
            "held_out_states": len(records),
            "ranking_train_labels": int(len(rank_labels)),
            "ranking_positive_rate": float(np.mean(rank_labels)),
            "oracle_feasible_rate": float(np.mean([row["oracle_any_feasible"] for row in records])),
        },
    }
    (output / "torque_readout_ranking.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    lines = [
        "# Held-out torque readout ranking audit",
        "",
        "| variant | top-1 actual feasible | top-3 hit | top-8 hit | top-1 angle improvement | effect RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        item = summary[variant]
        lines.append(
            "| %s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.4f [%.4f, %.4f] | %.4f [%.4f, %.4f] |"
            % (
                variant,
                item["top1_feasible"]["estimate"], item["top1_feasible"]["ci95_low"], item["top1_feasible"]["ci95_high"],
                item["top3_hit"]["estimate"], item["top3_hit"]["ci95_low"], item["top3_hit"]["ci95_high"],
                item["top8_hit"]["estimate"], item["top8_hit"]["ci95_low"], item["top8_hit"]["ci95_high"],
                item["top1_angle_improvement"]["estimate"], item["top1_angle_improvement"]["ci95_low"], item["top1_angle_improvement"]["ci95_high"],
                item["effect_rmse"]["estimate"], item["effect_rmse"]["ci95_low"], item["effect_rmse"]["ci95_high"],
            )
        )
    item = summary["contact_frame_specialist"]
    lines.append(
        "| contact_frame_specialist | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.4f [%.4f, %.4f] | %.4f [%.4f, %.4f] |"
        % (
            item["top1_feasible"]["estimate"], item["top1_feasible"]["ci95_low"], item["top1_feasible"]["ci95_high"],
            item["top3_hit"]["estimate"], item["top3_hit"]["ci95_low"], item["top3_hit"]["ci95_high"],
            item["top8_hit"]["estimate"], item["top8_hit"]["ci95_low"], item["top8_hit"]["ci95_high"],
            item["top1_angle_improvement"]["estimate"], item["top1_angle_improvement"]["ci95_low"], item["top1_angle_improvement"]["ci95_high"],
            item["effect_rmse"]["estimate"], item["effect_rmse"]["ci95_low"], item["effect_rmse"]["ci95_high"],
        )
    )
    item = summary["goal_conditioned_ranker"]
    lines.append(
        "| goal_conditioned_ranker | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.4f [%.4f, %.4f] | n/a |"
        % (
            item["top1_feasible"]["estimate"], item["top1_feasible"]["ci95_low"], item["top1_feasible"]["ci95_high"],
            item["top3_hit"]["estimate"], item["top3_hit"]["ci95_low"], item["top3_hit"]["ci95_high"],
            item["top8_hit"]["estimate"], item["top8_hit"]["ci95_low"], item["top8_hit"]["ci95_high"],
            item["top1_angle_improvement"]["estimate"], item["top1_angle_improvement"]["ci95_low"], item["top1_angle_improvement"]["ci95_high"],
        )
    )
    lines += ["", report["claim_boundary"], ""]
    (output / "torque_readout_ranking.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "summary": summary, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
