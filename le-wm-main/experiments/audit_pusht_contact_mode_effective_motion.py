"""Held-out audit of contact-mode and effective-motion action ranking.

This experiment isolates the remaining learned-adapter failure before any new
closed-loop run.  A candidate action is useful only when it (i) improves the
current pose objective, (ii) produces non-trivial block motion, and (iii),
when already in contact, preserves contact.  The first condition is
goal-dependent; the latter two are candidate-specific state-transition
properties that are absent from an effect-only ranker.

The protocol is privileged simulator-state diagnosis.  The simulator
successor is used to create offline training labels and to score held-out
actions, but never as an input to a selector.  Train and held-out states use
independent reset streams.  This script is deliberately an offline gate: it
does not launch a new closed-loop claim unless the held-out ranking evidence
supports the interface.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.dummy import DummyClassifier

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
    fit_direct_contact_endpoint_model,
    fit_goal_conditioned_feasibility_classifier,
    goal_conditioned_ranker_features,
    make_state_only_env,
    torque_action_bank,
    torque_feature_vector,
    unit,
)


MODE_ORDER = ("free", "approach", "contact")
PHASE_ORDER = ("translation", "closure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-states-per-mode", type=int, default=24)
    parser.add_argument("--held-out-states-per-mode", type=int, default=16)
    parser.add_argument("--modes", nargs="+", choices=MODE_ORDER, default=list(MODE_ORDER))
    parser.add_argument("--pool-transitions", type=int, default=4096)
    parser.add_argument("--directions", type=int, default=16)
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[0.35, 0.55, 0.75])
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--goals-per-state", type=int, default=6)
    parser.add_argument(
        "--closure-only", action="store_true",
        help="use near-pose goals so every ranking group measures orientation closure",
    )
    parser.add_argument("--trees", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def short_angle(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"estimate": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(values.size),
    }


def goal_for_state(
    state: np.ndarray,
    state_index: int,
    rng: np.random.Generator,
    closure_only: bool = False,
) -> np.ndarray:
    """Generate a bounded target pose without exposing held-out successors."""
    state = np.asarray(state, dtype=np.float64)
    goal = state.copy()
    radius = float(rng.uniform(0.0, 30.0) if closure_only else rng.uniform(18.0, 180.0))
    direction = float(rng.uniform(-np.pi, np.pi))
    goal[2:4] = np.clip(
        state[2:4] + radius * np.asarray([np.cos(direction), np.sin(direction)]),
        20.0,
        492.0,
    )
    # Alternate local closure targets and larger translation targets so that
    # phase is not a hidden proxy for the train/held-out split.
    if not closure_only and state_index % 3 == 0:
        goal[2:4] = np.clip(
            state[2:4] + rng.normal(0.0, 28.0, size=2), 20.0, 492.0
        )
    goal[4] = short_angle(state[4] + float(rng.uniform(-0.60, 0.60) if closure_only else rng.uniform(-np.pi, np.pi)))
    return goal


def contact_mode_features(
    state: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    """Current-state candidate features for motion and contact prediction.

    The explicit PD target is included because PushT applies the action as a
    displacement from the current agent position.  No next-state quantity is
    present.  Absolute block coordinates are omitted; walls are outside this
    local contact-interface audit.
    """
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    previous_action = np.asarray(previous_action, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    relative = state[2:4] - state[:2]
    distance = float(np.linalg.norm(relative))
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    block_angle = float(state[4])
    rotation = np.asarray(
        [[np.cos(block_angle), np.sin(block_angle)],
         [-np.sin(block_angle), np.cos(block_angle)]],
        dtype=np.float64,
    )
    target_relative = state[:2] + 100.0 * action - state[2:4]
    goal_delta = goal[2:4] - state[2:4]
    goal_block = rotation @ goal_delta
    angle_delta = short_angle(goal[4] - state[4])
    velocity = np.asarray(state[5:7], dtype=np.float64)
    current_mode = mode_for_state(state)
    mode_one_hot = np.asarray([float(current_mode == mode) for mode in MODE_ORDER])
    return np.concatenate([
        np.asarray([
            distance,
            float(np.dot(action, normal)),
            float(np.dot(action, tangent)),
            float(np.linalg.norm(action)),
            float(np.dot(previous_action, normal)),
            float(np.dot(previous_action, tangent)),
            float(np.dot(action - previous_action, normal)),
            float(np.dot(action - previous_action, tangent)),
            float(np.dot(velocity, normal)),
            float(np.dot(velocity, tangent)),
            float(np.linalg.norm(velocity)),
            float(relative[0]),
            float(relative[1]),
            float(target_relative[0]),
            float(target_relative[1]),
            float(np.linalg.norm(target_relative)),
            float(np.dot(target_relative, normal)),
            float(np.dot(target_relative, tangent)),
            float(np.dot(goal_delta, normal)),
            float(np.dot(goal_delta, tangent)),
            float(np.linalg.norm(goal_delta)),
            float(goal_block[0]),
            float(goal_block[1]),
            float(np.cos(angle_delta)),
            float(np.sin(angle_delta)),
            abs(angle_delta),
            float(np.cos(np.arctan2(relative[1], relative[0]) - block_angle)),
            float(np.sin(np.arctan2(relative[1], relative[0]) - block_angle)),
        ], dtype=np.float64),
        mode_one_hot,
    ])


def actual_endpoint(env, state: np.ndarray, action: np.ndarray, horizon: int, seed: int) -> np.ndarray:
    env.reset(seed=seed, options={"state": state.copy(), "goal_state": state.copy()})
    endpoint = np.asarray(state, dtype=np.float64).copy()
    for _ in range(horizon):
        observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
        if terminated or truncated:
            break
    return endpoint


def row_from_transition(state: np.ndarray, action: np.ndarray, previous: np.ndarray, goal: np.ndarray, endpoint: np.ndarray) -> dict[str, Any]:
    before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
    after_xy = float(np.linalg.norm(endpoint[2:4] - goal[2:4]))
    before_angle = angle_error(state[4], goal[4])
    after_angle = angle_error(endpoint[4], goal[4])
    motion = float(np.linalg.norm(endpoint[2:4] - state[2:4]))
    phase = "translation" if before_xy >= 45.0 else "closure"
    xy_limit = min(45.0, before_xy + 4.0)
    pose_progress = {
        "xy": before_xy - after_xy,
        "angle": before_angle - after_angle,
    }
    useful_motion = motion >= 0.5
    preserves_contact = mode_for_state(endpoint) == "contact"
    if phase == "translation":
        desired_progress = pose_progress["xy"] > 0.0 and pose_progress["angle"] >= -0.12
    else:
        desired_progress = pose_progress["angle"] > 0.0 and after_xy <= xy_limit
    feasible = bool(desired_progress and useful_motion)
    if mode_for_state(state) == "contact":
        feasible = bool(feasible and preserves_contact)
    return {
        "state": np.asarray(state, dtype=np.float64),
        "action": np.asarray(action, dtype=np.float64),
        "previous_action": np.asarray(previous, dtype=np.float64),
        "goal": np.asarray(goal, dtype=np.float64),
        "endpoint": np.asarray(endpoint, dtype=np.float64),
        "mode": mode_for_state(state),
        "phase": phase,
        "xy_progress": float(pose_progress["xy"]),
        "angle_progress": float(pose_progress["angle"]),
        "block_motion": motion,
        "useful_motion": int(useful_motion),
        "preserves_contact": int(preserves_contact),
        "desired_progress": int(desired_progress),
        "feasible": int(feasible),
    }


def collect_rows(
    env,
    states: list[np.ndarray],
    actions: np.ndarray,
    goals_per_state: int,
    horizon: int,
    seed: int,
    training: bool,
    closure_only: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + (100 if training else 200))
    for state_index, state in enumerate(states):
        previous = np.zeros(2, dtype=np.float64)
        # The physical successor depends on (state, action, horizon), not on
        # the synthetic goal.  Cache it once per candidate and reuse it for
        # every goal so the training budget is not silently multiplied by
        # goals_per_state.
        endpoints = [
            actual_endpoint(
                env,
                state,
                action,
                horizon,
                seed + state_index * 100000 + action_index,
            )
            for action_index, action in enumerate(actions)
        ]
        for goal_index in range(goals_per_state):
            goal = goal_for_state(
                state,
                state_index * goals_per_state + goal_index,
                rng,
                closure_only=closure_only,
            )
            for action, endpoint in zip(actions, endpoints):
                row = row_from_transition(state, action, previous, goal, endpoint)
                row["features"] = contact_mode_features(state, action, previous, goal)
                rows.append(row)
    return rows


def select_balanced_states(
    pool: dict[str, np.ndarray], per_mode: int, seed: int, modes: tuple[str, ...]
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    pool_modes = np.asarray(pool["modes"])
    for mode in modes:
        indices = np.flatnonzero(pool_modes == mode)
        if len(indices) < per_mode:
            raise RuntimeError(f"insufficient {mode} states: need {per_mode}, got {len(indices)}")
        rng.shuffle(indices)
        selected.extend(np.asarray(pool["states"])[indices[:per_mode]].copy())
    return selected


class EffectiveMotionModel:
    """Multi-head learned candidate model with fixed auditable outputs."""

    def __init__(self, quality, motion, contact):
        self.quality = quality
        self.motion = motion
        self.contact = contact

    @staticmethod
    def _positive_probability(model, features: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
        classes = np.asarray(model.classes_)
        positive = np.flatnonzero(classes == 1)
        if len(positive) != 1:
            return np.full(len(features), float(classes[0] == 1))
        return probabilities[:, int(positive[0])]

    def predict(self, features: np.ndarray) -> dict[str, np.ndarray]:
        quality = np.asarray(self.quality.predict(features), dtype=np.float64)
        return {
            "xy_progress": quality[:, 0],
            "angle_progress": quality[:, 1],
            "motion": self._positive_probability(self.motion, features),
            "contact": self._positive_probability(self.contact, features),
        }


def fit_effective_motion_model(rows: list[dict[str, Any]], trees: int, seed: int) -> EffectiveMotionModel:
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    quality_targets = np.asarray([
        [row["xy_progress"], row["angle_progress"]] for row in rows
    ], dtype=np.float64)
    motion_labels = np.asarray([row["useful_motion"] for row in rows], dtype=np.int64)
    contact_labels = np.asarray([row["preserves_contact"] for row in rows], dtype=np.int64)
    quality = ExtraTreesRegressor(
        n_estimators=trees, min_samples_leaf=2, max_features=1.0,
        random_state=seed, n_jobs=1,
    ).fit(features, quality_targets)

    def fit_classifier(labels: np.ndarray, random_state: int):
        if len(np.unique(labels)) < 2:
            return DummyClassifier(strategy="constant", constant=int(labels[0])).fit(features, labels)
        return ExtraTreesClassifier(
            n_estimators=trees, min_samples_leaf=2, max_features=1.0,
            class_weight="balanced", random_state=random_state, n_jobs=1,
        ).fit(features, labels)

    return EffectiveMotionModel(
        quality,
        fit_classifier(motion_labels, seed + 11),
        fit_classifier(contact_labels, seed + 13),
    )


def predicted_feasible_rows(
    rows: list[dict[str, Any]],
    predictions: dict[str, np.ndarray],
    state: np.ndarray,
    *,
    motion_threshold: float = 0.5,
    contact_threshold: float = 0.5,
) -> np.ndarray:
    current_mode = mode_for_state(state)
    phase = rows[0]["phase"]
    if phase == "translation":
        desired = (predictions["xy_progress"] > 0.0) & (predictions["angle_progress"] >= -0.12)
    else:
        before_xy = float(np.linalg.norm(state[2:4] - rows[0]["goal"][2:4]))
        predicted_after_xy = before_xy - predictions["xy_progress"]
        desired = (predictions["angle_progress"] > 0.0) & (predicted_after_xy <= min(45.0, before_xy + 4.0))
    feasible = desired & (predictions["motion"] >= motion_threshold)
    if current_mode == "contact":
        feasible &= predictions["contact"] >= contact_threshold
    return feasible


def select_new(
    rows: list[dict[str, Any]],
    model: EffectiveMotionModel,
    *,
    motion_threshold: float = 0.5,
    contact_threshold: float = 0.5,
) -> tuple[int, dict[str, Any]]:
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    predictions = model.predict(features)
    feasible = predicted_feasible_rows(
        rows, predictions, rows[0]["state"],
        motion_threshold=motion_threshold,
        contact_threshold=contact_threshold,
    )
    phase = rows[0]["phase"]
    if phase == "translation":
        score = -predictions["xy_progress"] + 0.20 * np.maximum(-predictions["angle_progress"] - 0.12, 0.0)
    else:
        score = -predictions["angle_progress"] + 0.02 * np.maximum(-predictions["xy_progress"], 0.0)
    score = score + 0.50 * (1.0 - predictions["motion"])
    if mode_for_state(rows[0]["state"]) == "contact":
        score = score + 0.50 * (1.0 - predictions["contact"])
    if np.any(feasible):
        score = score.copy()
        score[~feasible] = np.inf
        selection_mode = "hard_motion_contact_gate"
    else:
        selection_mode = "soft_violation_fallback"
    selected = int(np.argmin(score))
    return selected, {
        "selection_mode": selection_mode,
        "predicted_motion_probability": float(predictions["motion"][selected]),
        "predicted_contact_probability": float(predictions["contact"][selected]),
        "predicted_xy_progress": float(predictions["xy_progress"][selected]),
        "predicted_angle_progress": float(predictions["angle_progress"][selected]),
        "predicted_feasible_count": int(feasible.sum()),
        "motion_threshold": float(motion_threshold),
        "contact_threshold": float(contact_threshold),
        "predicted_order": np.argsort(score, kind="mergesort").tolist(),
    }


def select_current_classifier(rows: list[dict[str, Any]], model) -> tuple[int, dict[str, Any]]:
    features = np.asarray([
        goal_conditioned_ranker_features(
            row["state"], row["action"], row["previous_action"], row["goal"], "contact_frame"
        ) for row in rows
    ], dtype=np.float64)
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    scores = probabilities[:, int(positive[0])] if len(positive) else np.zeros(len(rows))
    selected = int(np.argmax(scores))
    return selected, {
        "predicted_feasibility": float(scores[selected]),
        "predicted_feasible_count": int((scores >= 0.5).sum()),
        "predicted_order": np.argsort(-scores, kind="mergesort").tolist(),
    }


def local_to_world(state: np.ndarray, local: np.ndarray) -> np.ndarray:
    relative = np.asarray(state[2:4] - state[:2], dtype=np.float64)
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    return np.asarray([
        local[0] * normal[0] + local[1] * tangent[0],
        local[0] * normal[1] + local[1] * tangent[1],
        local[2],
    ])


def select_effect_only(rows: list[dict[str, Any]], model) -> tuple[int, dict[str, Any]]:
    state = rows[0]["state"]
    features = np.asarray([
        torque_feature_vector(state, row["action"], row["previous_action"], "contact_frame")
        for row in rows
    ], dtype=np.float64)
    local = np.asarray(model.predict(features), dtype=np.float64)
    effects = np.asarray([local_to_world(state, value) for value in local])
    goal = rows[0]["goal"]
    before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
    before_angle = angle_error(state[4], goal[4])
    predicted_xy = np.linalg.norm(state[2:4][None] + effects[:, :2] - goal[2:4][None], axis=1)
    predicted_angle = np.asarray([angle_error(state[4] + effect[2], goal[4]) for effect in effects])
    phase = rows[0]["phase"]
    if phase == "translation":
        score = predicted_xy + 0.10 * predicted_angle
    else:
        score = predicted_angle + 0.02 * predicted_xy
    selected = int(np.argmin(score))
    return selected, {
        "predicted_effect_finite": bool(np.isfinite(effects).all()),
        "predicted_order": np.argsort(score, kind="mergesort").tolist(),
    }


def candidate_utility_cost(row: dict[str, Any]) -> float:
    """Return the declared successor utility cost used for direct ranking.

    This is intentionally the same scalar objective used by ``metric_row``.
    The direct ranker learns this candidate-level quantity from training
    transitions instead of predicting a pose effect and applying a second,
    potentially inconsistent angle/contact transformation at deployment.
    """
    phase = str(row["phase"])
    progress = float(row["xy_progress"] if phase == "translation" else row["angle_progress"])
    return (
        -progress
        + 0.50 * (1.0 - float(row["useful_motion"]))
        + (0.50 * (1.0 - float(row["preserves_contact"]))
           if row["mode"] == "contact" else 0.0)
    )


def fit_direct_utility_ranker(rows: list[dict[str, Any]], trees: int, seed: int) -> ExtraTreesRegressor:
    """Fit a candidate-level utility model on disjoint training states."""
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    targets = np.asarray([candidate_utility_cost(row) for row in rows], dtype=np.float64)
    return ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed,
        n_jobs=1,
    ).fit(features, targets)


def fit_stratified_utility_rankers(
    rows: list[dict[str, Any]], trees: int, seed: int
) -> dict[str, ExtraTreesRegressor]:
    """Fit separate utility rankers for each current mode and planning phase."""
    fallback = fit_direct_utility_ranker(rows, trees, seed)
    models: dict[str, ExtraTreesRegressor] = {"fallback": fallback}
    for mode in MODE_ORDER:
        for phase in PHASE_ORDER:
            subset = [row for row in rows if row["mode"] == mode and row["phase"] == phase]
            key = f"{mode}/{phase}"
            models[key] = (
                fit_direct_utility_ranker(subset, trees, seed + 100 + len(models))
                if len(subset) >= 32 else fallback
            )
    return models


def select_direct_utility(rows: list[dict[str, Any]], model) -> tuple[int, dict[str, Any]]:
    """Rank the shared candidate fan directly by predicted task utility."""
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    predicted_cost = np.asarray(model.predict(features), dtype=np.float64)
    order = np.argsort(predicted_cost, kind="mergesort")
    return int(order[0]), {
        "predicted_utility_cost_finite": bool(np.isfinite(predicted_cost).all()),
        "predicted_utility_cost": float(predicted_cost[int(order[0])]),
        "predicted_order": order.tolist(),
    }


def select_stratified_utility(rows: list[dict[str, Any]], models) -> tuple[int, dict[str, Any]]:
    """Use the utility expert matching the current mode and translation phase."""
    key = f"{rows[0]['mode']}/{rows[0]['phase']}"
    model = models.get(key, models["fallback"])
    selected, info = select_direct_utility(rows, model)
    info["utility_route"] = key if key in models else "fallback"
    return selected, info


def metric_row(rows: list[dict[str, Any]], selected: int, info: dict[str, Any]) -> dict[str, Any]:
    selected_row = rows[selected]
    feasible = np.asarray([row["feasible"] for row in rows], dtype=np.int64)
    actual_score = np.asarray([
        (-row["xy_progress"] if rows[0]["phase"] == "translation" else -row["angle_progress"])
        + 0.50 * (1 - row["useful_motion"])
        + (0.50 * (1 - row["preserves_contact"]) if rows[0]["mode"] == "contact" else 0.0)
        for row in rows
    ])
    order = np.argsort(actual_score)
    predicted_order = np.asarray(info.get("predicted_order", [selected]), dtype=np.int64)
    if predicted_order.shape != (len(rows),):
        raise ValueError("selector must return one predicted order per candidate")
    oracle_available = bool(feasible.any())
    return {
        "mode": rows[0]["mode"],
        "phase": rows[0]["phase"],
        "selected_feasible": int(selected_row["feasible"]),
        "top3_hit": int(feasible[predicted_order[:3]].any()),
        "top8_hit": int(feasible[predicted_order[:8]].any()),
        "top3_hit_given_oracle": float(feasible[predicted_order[:3]].any()) if oracle_available else float("nan"),
        "top8_hit_given_oracle": float(feasible[predicted_order[:8]].any()) if oracle_available else float("nan"),
        "selected_useful_motion": int(selected_row["useful_motion"]),
        "selected_preserves_contact": int(selected_row["preserves_contact"]),
        "selected_xy_progress": float(selected_row["xy_progress"]),
        "selected_angle_progress": float(selected_row["angle_progress"]),
        "oracle_feasible": int(feasible.any()),
        "oracle_feasible_count": int(feasible.sum()),
        "selected_rank_by_true_score": int(np.flatnonzero(order == selected)[0] + 1),
        **info,
    }


def evaluate_selector(groups: list[list[dict[str, Any]]], selector_name: str, selector, bootstrap_draws: int, seed: int) -> dict[str, Any]:
    rows = []
    for group in groups:
        selected, info = selector(group)
        result = metric_row(group, selected, info)
        result["selector"] = selector_name
        rows.append(result)
    metric_names = (
        "selected_feasible", "top3_hit", "top8_hit", "top3_hit_given_oracle", "top8_hit_given_oracle",
        "selected_useful_motion", "selected_preserves_contact",
        "oracle_feasible", "selected_xy_progress", "selected_angle_progress",
        "selected_rank_by_true_score",
    )
    metrics = {
        name: bootstrap(np.asarray([row[name] for row in rows]), bootstrap_draws, seed + index)
        for index, name in enumerate(metric_names)
    }
    by_stratum: dict[str, Any] = {}
    for mode in MODE_ORDER:
        for phase in PHASE_ORDER:
            subset = [row for row in rows if row["mode"] == mode and row["phase"] == phase]
            if not subset:
                continue
            key = f"{mode}/{phase}"
            by_stratum[key] = {
                name: bootstrap(np.asarray([row[name] for row in subset]), bootstrap_draws, seed + 100 + index)
                for index, name in enumerate(("selected_feasible", "top3_hit", "top8_hit", "top3_hit_given_oracle", "top8_hit_given_oracle", "selected_useful_motion", "selected_preserves_contact", "selected_rank_by_true_score"))
            }
    return {"selector": selector_name, "metrics": metrics, "by_stratum": by_stratum, "rows": rows}


def main() -> None:
    args = parse_args()
    if args.horizon < 1 or args.goals_per_state < 2:
        raise ValueError("horizon must be positive and goals_per_state >= 2")
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = make_state_only_env()
    try:
        train_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 100000)
        held_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 200000)
        modes = tuple(args.modes)
        train_states = select_balanced_states(train_pool, args.train_states_per_mode, args.seed + 1, modes)
        held_states = select_balanced_states(held_pool, args.held_out_states_per_mode, args.seed + 2, modes)
        actions = torque_action_bank(args.directions, magnitudes=tuple(args.magnitudes))
        train_rows = collect_rows(env, train_states, actions, args.goals_per_state, args.horizon, args.seed + 300000, True, args.closure_only)
        held_rows = collect_rows(env, held_states, actions, args.goals_per_state, args.horizon, args.seed + 400000, False, args.closure_only)
        effective = fit_effective_motion_model(train_rows, args.trees, args.seed + 7)
        current = fit_goal_conditioned_feasibility_classifier(
            env,
            np.asarray(train_states),
            actions,
            horizon=args.horizon,
            trees=args.trees,
            seed=args.seed + 9,
            previous_actions=np.zeros((len(train_states), 2), dtype=np.float64),
            feature_interface="contact_frame",
            goals_per_state=args.goals_per_state,
        )
        effect_only = fit_direct_contact_endpoint_model(
            env,
            np.asarray(train_states),
            actions,
            horizon=args.horizon,
            trees=args.trees,
            seed=args.seed + 11,
            previous_actions=np.zeros((len(train_states), 2), dtype=np.float64),
            feature_interface="contact_frame",
        )
        direct_utility = fit_direct_utility_ranker(
            train_rows, args.trees, args.seed + 17
        )
        stratified_utility = fit_stratified_utility_rankers(
            train_rows, args.trees, args.seed + 19
        )
    finally:
        env.close()

    # Each held-out state/goal is a separate ranking group.  A group contains
    # exactly one complete candidate fan, so all selectors see identical cases.
    group_size = len(actions)
    groups = [held_rows[index:index + group_size] for index in range(0, len(held_rows), group_size)]
    if any(len(group) != group_size for group in groups):
        raise RuntimeError("held-out candidate groups are incomplete")
    reports = [
        evaluate_selector(groups, "effect_only", lambda group: select_effect_only(group, effect_only), args.bootstrap, args.seed + 20),
        evaluate_selector(groups, "existing_feasibility", lambda group: select_current_classifier(group, current), args.bootstrap, args.seed + 30),
        evaluate_selector(groups, "contact_mode_effective_motion", lambda group: select_new(group, effective), args.bootstrap, args.seed + 40),
        evaluate_selector(groups, "direct_utility_ranker", lambda group: select_direct_utility(group, direct_utility), args.bootstrap, args.seed + 50),
        evaluate_selector(groups, "stratified_utility_ranker", lambda group: select_stratified_utility(group, stratified_utility), args.bootstrap, args.seed + 60),
    ]
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "train_states": len(train_states),
            "held_out_states": len(held_states),
            "train_states_per_mode": args.train_states_per_mode,
            "held_out_states_per_mode": args.held_out_states_per_mode,
            "modes": list(modes),
            "action_count": len(actions),
            "directions": args.directions,
            "magnitudes": [float(value) for value in args.magnitudes],
            "horizon": args.horizon,
            "goals_per_state": args.goals_per_state,
            "closure_only": bool(args.closure_only),
            "trees": args.trees,
            "seed": args.seed,
            "successor_used_by_selector": False,
            "train_heldout_disjoint_by_independent_reset_streams": True,
            "useful_motion_threshold": 0.5,
            "contact_mode_thresholds": {"free_gt": 90.0, "approach_gt": 45.0, "contact_le": 45.0},
            "wall_seconds": time.perf_counter() - started,
        },
        "label_definition": {
            "desired_progress": "translation: positive XY progress and angle non-worsening; closure: positive angle progress within XY tube",
            "useful_motion": "block displacement >= 0.5 pixels over the fixed action horizon",
            "preserves_contact": "endpoint remains in the current contact mode when the source state is contact",
            "feasible": "desired_progress AND useful_motion AND contact preservation for contact sources",
        },
        "feature_interface": {
            "name": "explicit contact-mode and effective-motion features",
            "inputs": "current relative geometry, PD target relative to block, current velocity, previous action, goal-relative pose, current mode",
            "future_state_inputs": False,
        },
        "reports": reports,
        "sanity_checks": {
            "all_finite": bool(all(
                np.isfinite(value)
                for report_item in reports
                for metric in report_item["metrics"].values()
                for value in metric.values()
                if isinstance(value, (int, float))
            )),
            "same_candidate_fan": True,
            "same_held_out_groups": True,
            "train_rows": len(train_rows),
            "held_out_rows": len(held_rows),
            "held_out_groups": len(groups),
            "held_out_group_size": group_size,
        },
        "held_out_label_summary": {
            f"{mode}/{phase}": {
                "groups": int(sum(row[0]["mode"] == mode and row[0]["phase"] == phase for row in groups)),
                "oracle_available_rate": float(np.mean([
                    any(row["feasible"] for row in group)
                    for group in groups
                    if group[0]["mode"] == mode and group[0]["phase"] == phase
                ])) if any(group[0]["mode"] == mode and group[0]["phase"] == phase for group in groups) else float("nan"),
                "mean_feasible_candidates": float(np.mean([
                    sum(row["feasible"] for row in group)
                    for group in groups
                    if group[0]["mode"] == mode and group[0]["phase"] == phase
                ])) if any(group[0]["mode"] == mode and group[0]["phase"] == phase for group in groups) else float("nan"),
            }
            for mode in modes
            for phase in PHASE_ORDER
        },
        "claim_boundary": (
            "Privileged simulator-state held-out ranking audit. It tests whether explicit current-state "
            "motion and contact-preservation heads repair candidate selection; it does not establish frozen "
            "LeWM latent causality, visual online planning, universal world modeling, or closed-loop task success."
        ),
    }
    (output / "contact_mode_effective_motion_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# PushT contact-mode and effective-motion ranking audit",
        "",
        "| selector | feasible top-1 | top-3 hit | top-8 hit | top-8 given oracle | true-score rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in reports:
        metrics = item["metrics"]
        lines.append(
            "| %s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.2f [%.2f, %.2f] |" % (
                item["selector"],
                metrics["selected_feasible"]["estimate"], metrics["selected_feasible"]["ci95_low"], metrics["selected_feasible"]["ci95_high"],
                metrics["top3_hit"]["estimate"], metrics["top3_hit"]["ci95_low"], metrics["top3_hit"]["ci95_high"],
                metrics["top8_hit"]["estimate"], metrics["top8_hit"]["ci95_low"], metrics["top8_hit"]["ci95_high"],
                metrics["top8_hit_given_oracle"]["estimate"], metrics["top8_hit_given_oracle"]["ci95_low"], metrics["top8_hit_given_oracle"]["ci95_high"],
                metrics["selected_rank_by_true_score"]["estimate"], metrics["selected_rank_by_true_score"]["ci95_low"], metrics["selected_rank_by_true_score"]["ci95_high"],
            )
        )
    lines += ["", report["claim_boundary"], ""]
    (output / "contact_mode_effective_motion_audit.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "reports": reports, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
