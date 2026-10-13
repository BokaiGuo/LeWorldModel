"""Direct learned pose-decision ablation for PushT contact closure.

This experiment tests the remaining interface hypothesis rather than adding a
larger planner.  The selector learns a binary pose-directed decision event
directly from current-state features.  It never predicts an effect vector and
never simulates candidate successors at deployment.

The matched ablation crosses two factors:

* supervision horizon: one-step, two-step, or four-step repeated-action
  labels;
* contact handling: a hard joint label versus a soft contact-risk penalty.

All variants use the same held-out contact states, goals, action fan, and
closed-loop observations.  Successor simulation is used only for offline
training labels and for the post-hoc failure decomposition.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier

import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_pusht_contact_mode_effective_motion import (  # noqa: E402
    contact_mode_features,
    goal_for_state,
    select_balanced_states,
)
from experiments.run_pusht_contact_history_velocity_diagnostic import (  # noqa: E402
    collect_trajectory_data,
    mode_for_state,
)
from experiments.run_pusht_object_centric_learned_planner import (  # noqa: E402
    angle_error,
    goal_conditioned_ranker_features,
    make_state_only_env,
    torque_action_bank,
)


VARIANTS = (
    "h1_joint",
    "h1_soft_contact",
    "h1_soft_contact_feedback",
    "h1_turn_soft_contact",
    "h2_soft_contact",
    "h4_soft_contact",
    "multi_soft_contact",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-states", type=int, default=32)
    parser.add_argument("--held-out-episodes", type=int, default=32)
    parser.add_argument("--pool-transitions", type=int, default=4096)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[0.15, 0.35, 0.55, 0.75])
    parser.add_argument("--goals-per-state", type=int, default=6)
    parser.add_argument("--closure-steps", type=int, default=16)
    parser.add_argument("--trees", type=int, default=96)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261030)
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


def fit_binary(features: np.ndarray, labels: np.ndarray, trees: int, seed: int):
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return DummyClassifier(strategy="constant", constant=int(labels[0])).fit(features, labels)
    return ExtraTreesClassifier(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    ).fit(features, labels)


def positive_probability(model, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        return np.full(len(features), float(classes[0] == 1))
    return probabilities[:, int(positive[0])]


def action_endpoint(env, state: np.ndarray, action: np.ndarray, horizon: int, seed: int) -> np.ndarray:
    env.reset(seed=seed, options={"state": np.asarray(state).copy(), "goal_state": np.asarray(state).copy()})
    endpoint = np.asarray(state, dtype=np.float64).copy()
    for _ in range(horizon):
        observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
        if terminated or truncated:
            break
    return endpoint


def label_endpoint(
    state: np.ndarray,
    goal: np.ndarray,
    endpoint: np.ndarray,
    require_contact: bool,
    xy_tube_margin: float = 4.0,
    phase_switch_xy: float | None = None,
    translation_angle_tolerance: float = 0.12,
) -> dict[str, int]:
    before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
    after_xy = float(np.linalg.norm(endpoint[2:4] - goal[2:4]))
    before_angle = angle_error(state[4], goal[4])
    after_angle = angle_error(endpoint[4], goal[4])
    if phase_switch_xy is not None and before_xy > float(phase_switch_xy):
        # The terminal interface is conjunctive: first reduce object XY while
        # allowing a bounded angular drift, then close orientation inside the
        # XY tube.  Requiring angular improvement in this phase creates an
        # artificial no-oracle state for near-goal cases.
        pose = bool(
            after_xy < before_xy - 1e-5
            and after_angle <= before_angle + float(translation_angle_tolerance)
            and after_xy <= 45.0
        )
    else:
        pose = bool(
            after_angle < before_angle - 1e-5
            and after_xy <= min(45.0, before_xy + float(xy_tube_margin))
        )
    motion = bool(np.linalg.norm(endpoint[2:4] - state[2:4]) >= 0.5)
    contact = bool(mode_for_state(endpoint) == "contact")
    return {
        "pose": int(pose),
        "turn": int(after_angle < before_angle - 1e-5),
        "motion": int(motion),
        "contact": int(contact),
        "joint": int(pose and motion and (contact if require_contact else True)),
        "angle_regression": int(after_angle > before_angle + 0.02),
        "xy_regression": int(after_xy > before_xy + 1.0),
    }


def make_training_rows(
    env,
    states: list[np.ndarray],
    actions: np.ndarray,
    goals_per_state: int,
    horizons: tuple[int, ...],
    seed: int,
    xy_tube_margin: float = 4.0,
    phase_switch_xy: float | None = None,
    translation_angle_tolerance: float = 0.12,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 1001)
    max_horizon = max(horizons)
    for state_index, state in enumerate(states):
        endpoints: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
        for action_index, action in enumerate(actions):
            for horizon in horizons:
                endpoints[horizon].append(action_endpoint(
                    env, state, action, horizon,
                    seed + state_index * 100000 + action_index * 100 + horizon,
                ))
        for goal_index in range(goals_per_state):
            goal = goal_for_state(state, state_index * goals_per_state + goal_index, rng, closure_only=True)
            for action_index, action in enumerate(actions):
                labels = {
                    str(h): label_endpoint(
                        state,
                        goal,
                        endpoints[h][action_index],
                        require_contact=mode_for_state(state) == "contact",
                        xy_tube_margin=xy_tube_margin,
                        phase_switch_xy=phase_switch_xy,
                        translation_angle_tolerance=translation_angle_tolerance,
                    ) for h in horizons
                }
                rows.append({
                    "state": np.asarray(state, dtype=np.float64),
                    "action": np.asarray(action, dtype=np.float64),
                    "previous_action": np.zeros(2, dtype=np.float64),
                    "goal": np.asarray(goal, dtype=np.float64),
                    "features": contact_mode_features(state, action, np.zeros(2), goal),
                    "labels": labels,
                    "mode": mode_for_state(state),
                })
    return rows


def fit_models(rows: list[dict[str, Any]], horizons: tuple[int, ...], trees: int, seed: int) -> dict[str, Any]:
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    models: dict[str, Any] = {}
    for horizon in horizons:
        key = str(horizon)
        models[f"pose_{key}"] = fit_binary(features, np.asarray([row["labels"][key]["pose"] for row in rows]), trees, seed + horizon)
        models[f"turn_{key}"] = fit_binary(features, np.asarray([row["labels"][key]["turn"] for row in rows]), trees, seed + 10 + horizon)
        models[f"motion_{key}"] = fit_binary(features, np.asarray([row["labels"][key]["motion"] for row in rows]), trees, seed + 30 + horizon)
        models[f"joint_{key}"] = fit_binary(features, np.asarray([row["labels"][key]["joint"] for row in rows]), trees, seed + 20 + horizon)
        models[f"contact_{key}"] = fit_binary(features, np.asarray([row["labels"][key]["contact"] for row in rows]), trees, seed + 40 + horizon)
    return models


def candidate_rows(state: np.ndarray, goal: np.ndarray, previous: np.ndarray, actions: np.ndarray) -> list[dict[str, Any]]:
    return [{
        "state": np.asarray(state, dtype=np.float64),
        "action": np.asarray(action, dtype=np.float64),
        "previous_action": np.asarray(previous, dtype=np.float64),
        "goal": np.asarray(goal, dtype=np.float64),
        "features": contact_mode_features(state, action, previous, goal),
        "mode": mode_for_state(state),
    } for action in actions]


def score_variant(variant: str, models: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    p = {key: positive_probability(model, features) for key, model in models.items()}
    if variant == "h1_joint":
        score = -p["joint_1"]
    elif variant in ("h1_soft_contact", "h1_soft_contact_feedback"):
        score = -p["pose_1"] + 0.35 * (1.0 - p["contact_1"])
    elif variant == "h1_turn_soft_contact":
        score = -p["turn_1"] + 0.20 * (1.0 - p["motion_1"]) + 0.35 * (1.0 - p["contact_1"])
    elif variant == "h2_soft_contact":
        score = -p["pose_2"] + 0.35 * (1.0 - p["contact_1"])
    elif variant == "h4_soft_contact":
        score = -p["pose_4"] + 0.35 * (1.0 - p["contact_1"])
    elif variant == "multi_soft_contact":
        pose = 0.50 * p["pose_1"] + 0.30 * p["pose_2"] + 0.20 * p["pose_4"]
        score = -pose + 0.35 * (1.0 - p["contact_1"])
    else:
        raise ValueError(variant)
    order = np.argsort(score, kind="mergesort")
    selected = int(order[0])
    return selected, {
        "predicted_order": order.tolist(),
        "selected_score": float(score[selected]),
        "selected_pose_probability": float(p.get("pose_1", p["joint_1"])[selected]),
        "selected_turn_probability": float(p.get("turn_1", p["pose_1"])[selected]),
        "selected_contact_probability": float(p["contact_1"][selected]),
        "selected_joint_probability": float(p["joint_1"][selected]),
        "selection_variant": variant,
    }


def variant_horizon(variant: str) -> int:
    if variant == "h2_soft_contact":
        return 2
    if variant == "h4_soft_contact":
        return 4
    if variant == "multi_soft_contact":
        return 2
    return 1


def feedback_restrict_order(
    rows: list[dict[str, Any]],
    predicted_order: list[int],
    previous_action: np.ndarray,
    current: np.ndarray,
    goal: np.ndarray,
    previous_angle_error: float | None,
) -> tuple[int, bool]:
    """Use one-step observed angle regression as a conservative sign gate.

    This is deliberately a runtime observation gate, not a simulator successor
    query.  It activates only after a completed action made the observed
    scalar block-angle error worse, and then retains the learned ordering among
    candidates whose contact-frame tangential command opposes the preceding
    command.  If the gate has no support it fails open to the learned choice.
    """
    if previous_angle_error is None or not rows:
        return int(predicted_order[0]), False
    current_error = angle_error(float(current[4]), float(goal[4]))
    if current_error <= previous_angle_error + 0.02:
        return int(predicted_order[0]), False
    previous_features = contact_mode_features(
        current,
        np.asarray(previous_action, dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        goal,
    )
    previous_tangent = float(previous_features[2])
    if abs(previous_tangent) < 1.0e-6:
        return int(predicted_order[0]), False
    candidates = [
        int(index) for index in predicted_order
        if float(rows[int(index)]["features"][2]) * previous_tangent <= 0.0
    ]
    if not candidates:
        return int(predicted_order[0]), False
    return candidates[0], True


def true_candidate_diagnostics(
    env,
    state: np.ndarray,
    goal: np.ndarray,
    actions: np.ndarray,
    horizon: int,
    seed: int,
    xy_tube_margin: float = 4.0,
    phase_switch_xy: float | None = None,
    translation_angle_tolerance: float = 0.12,
) -> list[dict[str, Any]]:
    diagnostics = []
    for index, action in enumerate(actions):
        endpoint = action_endpoint(env, state, action, horizon, seed + index)
        diagnostics.append(
            label_endpoint(
                state,
                goal,
                endpoint,
                require_contact=mode_for_state(state) == "contact",
                xy_tube_margin=xy_tube_margin,
                phase_switch_xy=phase_switch_xy,
                translation_angle_tolerance=translation_angle_tolerance,
            )
        )
    return diagnostics


def run_episode(
    env,
    variant: str,
    models: dict[str, Any],
    state: np.ndarray,
    goal: np.ndarray,
    actions: np.ndarray,
    steps: int,
    seed: int,
    xy_tube_margin: float = 4.0,
    phase_switch_xy: float | None = None,
    translation_angle_tolerance: float = 0.12,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed, options={"state": np.asarray(state).copy(), "goal_state": np.asarray(goal).copy()})
    current = np.asarray(observation["state"], dtype=np.float64).copy()
    initial_xy = float(np.linalg.norm(current[2:4] - goal[2:4]))
    initial_angle = angle_error(current[4], goal[4])
    previous = np.zeros(2, dtype=np.float64)
    previous_angle_error: float | None = None
    trace = []
    execution_horizon = variant_horizon(variant)
    for step in range(steps):
        angle_before = angle_error(float(current[4]), float(goal[4]))
        rows = candidate_rows(current, goal, previous, actions)
        selected, info = score_variant(variant, models, rows)
        if variant == "h1_soft_contact_feedback":
            selected, feedback_used = feedback_restrict_order(
                rows,
                info["predicted_order"],
                previous,
                current,
                goal,
                previous_angle_error,
            )
            info["feedback_gate_used"] = int(feedback_used)
            info["feedback_selected"] = int(selected)
        candidate_truth = true_candidate_diagnostics(
            env,
            current,
            goal,
            actions,
            execution_horizon,
            seed + 10000 + step * 100,
            xy_tube_margin=xy_tube_margin,
            phase_switch_xy=phase_switch_xy,
            translation_angle_tolerance=translation_angle_tolerance,
        )
        chosen_truth = candidate_truth[selected]
        oracle_count = int(sum(item["joint"] for item in candidate_truth))
        terminated = False
        truncated = False
        observation = {"state": current.copy()}
        for _ in range(execution_horizon):
            observation, _, terminated, truncated, _ = env.step(actions[selected].astype(np.float32))
            if terminated or truncated:
                break
        next_state = np.asarray(observation["state"], dtype=np.float64).copy()
        current_mode = mode_for_state(current)
        next_mode = mode_for_state(next_state)
        trace.append({
            "step": step,
            "execution_horizon": execution_horizon,
            "mode": current_mode,
            "selected": selected,
            "oracle_joint_count": oracle_count,
            "oracle_available": int(oracle_count > 0),
            "selected_joint_one_step": int(chosen_truth["joint"]),
            "selected_pose_one_step": int(chosen_truth["pose"]),
            "selected_contact_one_step": int(chosen_truth["contact"]),
            "angle_regression": int(chosen_truth["angle_regression"]),
            "xy_regression": int(chosen_truth["xy_regression"]),
            "contact_lost": int(current_mode == "contact" and next_mode != "contact"),
            "failure_class": (
                "no_oracle_candidate" if oracle_count == 0 else
                "selected_feasible" if chosen_truth["joint"] else
                "selected_wrong"
            ),
            **info,
        })
        current = next_state
        previous = actions[selected].astype(np.float64)
        # Store the pre-action error so the next decision can determine
        # whether this completed action caused an observed regression.
        previous_angle_error = angle_before
        if terminated or truncated:
            break
    return {
        "variant": variant,
        "initial_xy": initial_xy,
        "initial_angle": initial_angle,
        "final_xy": float(np.linalg.norm(current[2:4] - goal[2:4])),
        "final_angle": angle_error(current[4], goal[4]),
        "strict_success": int(np.linalg.norm(current[2:4] - goal[2:4]) < 20.0 and angle_error(current[4], goal[4]) < np.pi / 9.0),
        "contact_fraction": float(np.mean([int(item["mode"] == "contact") for item in trace])) if trace else 0.0,
        "trace": trace,
        "final_state": current.tolist(),
    }


def summarize(reports: list[dict[str, Any]], variant: str, draws: int, seed: int) -> dict[str, Any]:
    subset = [row for row in reports if row["variant"] == variant]
    metric_names = ("strict_success", "final_xy", "final_angle", "contact_fraction")
    metrics = {name: bootstrap(np.asarray([row[name] for row in subset]), draws, seed + index) for index, name in enumerate(metric_names)}
    trace = [item for row in subset for item in row["trace"]]
    failure_counts = {name: int(sum(item["failure_class"] == name for item in trace)) for name in ("no_oracle_candidate", "selected_feasible", "selected_wrong")}
    failure_counts["angle_regression"] = int(sum(item["angle_regression"] for item in trace))
    failure_counts["contact_lost"] = int(sum(item["contact_lost"] for item in trace))
    failure_counts["oracle_available"] = int(sum(item["oracle_available"] for item in trace))
    failure_counts["steps"] = len(trace)
    return {"variant": variant, "metrics": metrics, "failure_counts": failure_counts, "rows": subset}


def main() -> None:
    args = parse_args()
    if args.train_states < 8 or args.held_out_episodes < 8 or args.goals_per_state < 2:
        raise ValueError("state, episode, and goal counts are too small")
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    actions = torque_action_bank(args.directions, magnitudes=tuple(args.magnitudes))
    horizons = (1, 2, 4)
    env = make_state_only_env()
    try:
        train_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 100000)
        held_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 200000)
        train_states = select_balanced_states(train_pool, args.train_states, args.seed + 1, ("contact",))
        held_states = select_balanced_states(held_pool, args.held_out_episodes, args.seed + 2, ("contact",))
        train_rows = make_training_rows(env, train_states, actions, args.goals_per_state, horizons, args.seed + 300000)
        models = fit_models(train_rows, horizons, args.trees, args.seed + 7)
        rng = np.random.default_rng(args.seed + 400000)
        cases = [
            {"case": index, "state": np.asarray(state, dtype=np.float64).copy(), "goal": goal_for_state(state, index, rng, closure_only=True)}
            for index, state in enumerate(held_states)
        ]
        reports = []
        for case in cases:
            for variant in VARIANTS:
                reports.append(run_episode(env, variant, models, case["state"], case["goal"], actions, args.closure_steps, args.seed + 500000 + case["case"]))
    finally:
        env.close()
    summaries = [summarize(reports, variant, args.bootstrap, args.seed + 600 + index * 100) for index, variant in enumerate(VARIANTS)]
    paired = {}
    reference = [row for row in reports if row["variant"] == "h1_joint"]
    for variant in VARIANTS[1:]:
        current = [row for row in reports if row["variant"] == variant]
        paired[variant] = {
            name: bootstrap(np.asarray([new[name] - old[name] for old, new in zip(reference, current)]), args.bootstrap, args.seed + 900 + index)
            for index, name in enumerate(("strict_success", "final_xy", "final_angle", "contact_fraction"))
        }
    report = {
    "protocol": {
            "environment": "swm/PushT-v1",
            "train_contact_states": len(train_states),
            "held_out_contact_episodes": len(held_states),
            "action_count": len(actions),
            "directions": args.directions,
            "magnitudes": [float(value) for value in args.magnitudes],
            "goals_per_training_state": args.goals_per_state,
            "closure_steps": args.closure_steps,
            "supervision_horizons": list(horizons),
            "trees": args.trees,
            "seed": args.seed,
            "runtime_candidate_successor_verification": False,
            "execution_horizon_is_variant_matched": True,
            "train_heldout_disjoint_by_independent_reset_streams": True,
            "wall_seconds": time.perf_counter() - started,
        },
        "variants": {
            "h1_joint": "direct one-step joint pose-motion-contact decision probability",
            "h1_soft_contact": "direct one-step pose decision with soft contact-risk penalty",
            "h1_turn_soft_contact": "direct one-step goal-directed rotation decision with soft motion and contact-risk penalties",
            "h2_soft_contact": "direct two-step pose decision with soft one-step contact-risk penalty",
            "h4_soft_contact": "direct four-step pose decision with soft one-step contact-risk penalty",
            "multi_soft_contact": "weighted direct pose probabilities from one-, two-, and four-step labels with soft contact-risk penalty",
        },
        "summaries": summaries,
        "paired_vs_h1_joint": paired,
        "sanity_checks": {
            "same_cases_and_goals": True,
            "same_action_fan": True,
            "runtime_candidate_successor_verification": False,
            "all_finite": bool(all(np.isfinite(value) for summary in summaries for metric in summary["metrics"].values() for value in metric.values() if isinstance(value, (int, float)))),
        },
        "claim_boundary": "Privileged simulator-state held-out contact-pose closure ablation. It tests the learned decision interface and failure mechanism; it is not visual LeWM planning, universal world modeling, or physical causality evidence.",
    }
    (output / "direct_pose_decision_ablation.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# Direct learned pose-decision ablation",
        "",
        "| variant | strict success | final XY | final angle | contact fraction | no oracle | selected wrong | angle regression | contact lost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        m = summary["metrics"]
        f = summary["failure_counts"]
        lines.append(
            "| %s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %d | %d | %d | %d |" % (
                summary["variant"],
                m["strict_success"]["estimate"], m["strict_success"]["ci95_low"], m["strict_success"]["ci95_high"],
                m["final_xy"]["estimate"], m["final_xy"]["ci95_low"], m["final_xy"]["ci95_high"],
                m["final_angle"]["estimate"], m["final_angle"]["ci95_low"], m["final_angle"]["ci95_high"],
                m["contact_fraction"]["estimate"], m["contact_fraction"]["ci95_low"], m["contact_fraction"]["ci95_high"],
                f["no_oracle_candidate"], f["selected_wrong"], f["angle_regression"], f["contact_lost"],
            )
        )
    lines += ["", report["claim_boundary"], ""]
    (output / "direct_pose_decision_ablation.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "summaries": summaries, "paired_vs_h1_joint": paired, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
