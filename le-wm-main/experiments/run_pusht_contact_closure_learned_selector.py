"""Closed-loop contact-pose closure without runtime successor verification.

This runner is the next test after the held-out candidate-ranking audit.  It
starts each policy from an independently held-out simulator contact state and
uses the same near-pose goal and nine-action fan for two selectors:

* ``existing_feasibility``: the previous goal-conditioned feasibility head;
* ``motion_contact``: the new pose-progress plus useful-motion/contact heads.

At deployment, neither selector simulates candidate successors.  The action
selected from the current observation is committed once, and the resulting
observation is used only as the next closed-loop state.  This is a privileged
simulator-state closure benchmark, not visual LeWM planning evidence.
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

from experiments.audit_pusht_contact_mode_effective_motion import (  # noqa: E402
    EffectiveMotionModel,
    MODE_ORDER,
    collect_rows,
    contact_mode_features,
    goal_for_state,
    fit_effective_motion_model,
    select_balanced_states,
    select_new,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-states", type=int, default=32)
    parser.add_argument("--held-out-episodes", type=int, default=32)
    parser.add_argument("--pool-transitions", type=int, default=2048)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument(
        "--magnitudes", type=float, nargs="+", default=None,
        help="candidate action magnitudes; defaults to the legacy single magnitude",
    )
    parser.add_argument("--magnitude", type=float, default=0.55, help=argparse.SUPPRESS)
    parser.add_argument("--goals-per-state", type=int, default=4)
    parser.add_argument("--closure-steps", type=int, default=16)
    parser.add_argument("--trees", type=int, default=64)
    parser.add_argument("--motion-threshold", type=float, default=0.5)
    parser.add_argument("--contact-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261012)
    return parser.parse_args()


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


def fit_existing_from_rows(rows: list[dict[str, Any]], trees: int, seed: int) -> ExtraTreesClassifier:
    features = np.asarray([
        goal_conditioned_ranker_features(
            row["state"], row["action"], row["previous_action"], row["goal"], "contact_frame"
        ) for row in rows
    ], dtype=np.float64)
    labels = np.asarray([row["feasible"] for row in rows], dtype=np.int64)
    if len(np.unique(labels)) < 2:
        raise RuntimeError("existing feasibility training rows contain only one class")
    return ExtraTreesClassifier(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    ).fit(features, labels)


def candidate_rows(state: np.ndarray, goal: np.ndarray, previous: np.ndarray, actions: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "state": np.asarray(state, dtype=np.float64),
            "goal": np.asarray(goal, dtype=np.float64),
            "action": np.asarray(action, dtype=np.float64),
            "previous_action": np.asarray(previous, dtype=np.float64),
            "mode": mode_for_state(state),
            "phase": "closure",
            "features": contact_mode_features(state, action, previous, goal),
        }
        for action in actions
    ]


def select_existing(model: ExtraTreesClassifier, rows: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
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
    return np.asarray(rows[selected]["action"], dtype=np.float32), {
        "selected": selected,
        "predicted_feasibility": float(scores[selected]),
        "predicted_order": np.argsort(-scores, kind="mergesort").tolist(),
    }


def select_motion_contact(
    model: EffectiveMotionModel,
    rows: list[dict[str, Any]],
    motion_threshold: float,
    contact_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected, info = select_new(
        rows, model,
        motion_threshold=motion_threshold,
        contact_threshold=contact_threshold,
    )
    return np.asarray(rows[selected]["action"], dtype=np.float32), {"selected": selected, **info}


def run_episode(
    env,
    selector_name: str,
    selector,
    state: np.ndarray,
    goal: np.ndarray,
    actions: np.ndarray,
    steps: int,
    reset_seed: int,
) -> dict[str, Any]:
    observation, _ = env.reset(
        seed=reset_seed,
        options={"state": np.asarray(state).copy(), "goal_state": np.asarray(goal).copy()},
    )
    current = np.asarray(observation["state"], dtype=np.float64).copy()
    initial_xy = float(np.linalg.norm(current[2:4] - goal[2:4]))
    initial_angle = angle_error(current[4], goal[4])
    previous = np.zeros(2, dtype=np.float64)
    contact_trace = []
    action_trace = []
    selection_trace = []
    for step in range(steps):
        rows = candidate_rows(current, goal, previous, actions)
        action, info = selector(rows)
        observation, _, terminated, truncated, _ = env.step(action)
        current = np.asarray(observation["state"], dtype=np.float64).copy()
        contact_trace.append(int(mode_for_state(current) == "contact"))
        action_trace.append(np.asarray(action, dtype=np.float64).tolist())
        selection_trace.append(info)
        previous = np.asarray(action, dtype=np.float64)
        if terminated or truncated:
            break
    final_xy = float(np.linalg.norm(current[2:4] - goal[2:4]))
    final_angle = angle_error(current[4], goal[4])
    return {
        "selector": selector_name,
        "initial_xy": initial_xy,
        "initial_angle": initial_angle,
        "final_xy": final_xy,
        "final_angle": final_angle,
        "xy_improvement": initial_xy - final_xy,
        "angle_improvement": initial_angle - final_angle,
        "strict_success": int(final_xy < 20.0 and final_angle < np.pi / 9.0),
        "contact_fraction": float(np.mean(contact_trace)) if contact_trace else 0.0,
        "contact_steps": int(sum(contact_trace)),
        "steps_executed": len(action_trace),
        "action_trace": action_trace,
        "selection_trace": selection_trace,
        "final_state": current.tolist(),
    }


def summarize(rows: list[dict[str, Any]], selector: str, bootstrap_draws: int, seed: int) -> dict[str, Any]:
    subset = [row for row in rows if row["selector"] == selector]
    metrics = {}
    for index, name in enumerate((
        "strict_success", "final_xy", "final_angle", "xy_improvement",
        "angle_improvement", "contact_fraction", "steps_executed",
    )):
        metrics[name] = bootstrap(np.asarray([row[name] for row in subset]), bootstrap_draws, seed + index)
    return {"selector": selector, "metrics": metrics, "rows": subset}


def main() -> None:
    args = parse_args()
    if args.train_states < 8 or args.held_out_episodes < 8 or args.closure_steps < 1:
        raise ValueError("state, episode, or horizon counts are too small")
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    magnitudes = tuple(args.magnitudes) if args.magnitudes is not None else (args.magnitude,)
    if not magnitudes or any(value <= 0.0 for value in magnitudes):
        raise ValueError("magnitudes must contain positive values")
    actions = torque_action_bank(args.directions, magnitudes=magnitudes)
    env = make_state_only_env()
    try:
        train_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 100000)
        held_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 200000)
        train_states = select_balanced_states(train_pool, args.train_states, args.seed + 1, ("contact",))
        held_states = select_balanced_states(held_pool, args.held_out_episodes, args.seed + 2, ("contact",))
        train_rows = collect_rows(
            env, train_states, actions, args.goals_per_state, 1,
            args.seed + 300000, True, closure_only=True,
        )
        motion_contact = fit_effective_motion_model(train_rows, args.trees, args.seed + 7)
        existing = fit_existing_from_rows(train_rows, args.trees, args.seed + 9)
        goal_rng = np.random.default_rng(args.seed + 400000)
        cases = [
            {
                "case": index,
                "state": np.asarray(state, dtype=np.float64).copy(),
                "goal": goal_for_state(state, index, goal_rng, closure_only=True),
            }
            for index, state in enumerate(held_states)
        ]
        reports = []
        for case in cases:
            for selector_name, selector in (
                ("existing_feasibility", lambda rows: select_existing(existing, rows)),
                ("motion_contact", lambda rows: select_motion_contact(
                    motion_contact, rows, args.motion_threshold, args.contact_threshold
                )),
            ):
                reports.append(run_episode(
                    env, selector_name, selector, case["state"], case["goal"],
                    actions, args.closure_steps, args.seed + 500000 + case["case"],
                ))
    finally:
        env.close()

    summaries = [
        summarize(reports, selector, args.bootstrap, args.seed + 600 + index * 100)
        for index, selector in enumerate(("existing_feasibility", "motion_contact"))
    ]
    existing_rows = [row for row in reports if row["selector"] == "existing_feasibility"]
    motion_rows = [row for row in reports if row["selector"] == "motion_contact"]
    paired = {}
    for index, name in enumerate(("strict_success", "final_xy", "final_angle", "xy_improvement", "angle_improvement", "contact_fraction")):
        paired[name] = bootstrap(
            np.asarray([new[name] - old[name] for old, new in zip(existing_rows, motion_rows)]),
            args.bootstrap,
            args.seed + 800 + index,
        )
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "train_contact_states": len(train_states),
            "held_out_contact_episodes": len(held_states),
            "action_count": len(actions),
            "directions": args.directions,
            "magnitudes": [float(value) for value in magnitudes],
            "motion_threshold": float(args.motion_threshold),
            "contact_threshold": float(args.contact_threshold),
            "goals_per_training_state": args.goals_per_state,
            "closure_steps": args.closure_steps,
            "trees": args.trees,
            "seed": args.seed,
            "training_successor_labels": True,
            "runtime_candidate_successor_verification": False,
            "train_heldout_disjoint_by_independent_reset_streams": True,
            "wall_seconds": time.perf_counter() - started,
        },
        "selectors": {
            "existing_feasibility": "same closure-only training rows, previous goal-conditioned feasibility head",
            "motion_contact": "same rows, pose-progress regressor plus useful-motion and contact-preservation heads",
        },
        "summaries": summaries,
        "paired_motion_contact_minus_existing": paired,
        "sanity_checks": {
            "same_cases_and_goals": True,
            "same_action_fan": True,
            "runtime_candidate_successor_verification": False,
            "all_finite": bool(all(np.isfinite(value) for value in paired.values() for value in value.values() if isinstance(value, (int, float)))),
        },
        "claim_boundary": (
            "Privileged simulator-state contact-pose closure benchmark without runtime candidate successor "
            "verification. A positive result is evidence of bounded learned closed-loop closure from held-out "
            "contact states, not visual LeWM planning, universal world modeling, or physical causality."
        ),
    }
    (output / "contact_closure_learned_selector.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# Learned contact-pose closure without runtime successor verification",
        "",
        "| selector | strict success | final XY | final angle | contact fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        metrics = item["metrics"]
        lines.append(
            "| %s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] |" % (
                item["selector"],
                metrics["strict_success"]["estimate"], metrics["strict_success"]["ci95_low"], metrics["strict_success"]["ci95_high"],
                metrics["final_xy"]["estimate"], metrics["final_xy"]["ci95_low"], metrics["final_xy"]["ci95_high"],
                metrics["final_angle"]["estimate"], metrics["final_angle"]["ci95_low"], metrics["final_angle"]["ci95_high"],
                metrics["contact_fraction"]["estimate"], metrics["contact_fraction"]["ci95_low"], metrics["contact_fraction"]["ci95_high"],
            )
        )
    lines += ["", report["claim_boundary"], ""]
    (output / "contact_closure_learned_selector.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "summaries": summaries, "paired": paired, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
