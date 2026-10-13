"""Dataset-aggregated learned pose adapter for PushT.

The preceding ablations show that the selector is trained on stable contact
states but deployed on its own induced state distribution.  This runner tests
that explanation with a strict three-way split:

* initial states train the first adapter;
* separate aggregation episodes provide visited state--goal pairs only for
  retraining;
* a fresh test stream is never used for retraining or model selection.

The adapter remains a direct decision model: it predicts pose-directed
decision events from current-state features, without effect-vector
reconstruction or runtime candidate successor simulation.
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
    contact_mode_features,
    goal_for_state,
    select_balanced_states,
)
from experiments.run_pusht_contact_history_velocity_diagnostic import (  # noqa: E402
    collect_trajectory_data,
    mode_for_state,
)
from experiments.run_pusht_direct_pose_decision_ablation import (  # noqa: E402
    VARIANTS,
    action_endpoint,
    angle_error,
    bootstrap,
    candidate_rows,
    fit_models,
    run_episode,
    score_variant,
)
from experiments.run_pusht_object_centric_learned_planner import (  # noqa: E402
    make_state_only_env,
    torque_action_bank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-states", type=int, default=32)
    parser.add_argument("--aggregate-episodes", type=int, default=16)
    parser.add_argument("--test-episodes", type=int, default=32)
    parser.add_argument("--pool-transitions", type=int, default=4096)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[0.10, 0.15, 0.25, 0.35, 0.55])
    parser.add_argument("--goals-per-state", type=int, default=6)
    parser.add_argument("--closure-steps", type=int, default=16)
    parser.add_argument("--trees", type=int, default=96)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--xy-tube-margin",
        type=float,
        default=4.0,
        help="per-decision XY tube margin used for offline pose labels; 4 px is the legacy protocol",
    )
    parser.add_argument(
        "--phase-switch-xy",
        type=float,
        default=None,
        help="if set, use translation-first labels above this XY error and orientation labels below it",
    )
    parser.add_argument("--translation-angle-tolerance", type=float, default=0.12)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261040)
    return parser.parse_args()


def make_goal_rows(
    env,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    actions: np.ndarray,
    horizons: tuple[int, ...],
    seed: int,
    xy_tube_margin: float,
    phase_switch_xy: float | None,
    translation_angle_tolerance: float,
) -> list[dict[str, Any]]:
    """Create labels for exact visited state--goal pairs, offline only."""
    rows: list[dict[str, Any]] = []
    for pair_index, (state, goal) in enumerate(pairs):
        state = np.asarray(state, dtype=np.float64).copy()
        goal = np.asarray(goal, dtype=np.float64).copy()
        endpoints = {h: [action_endpoint(env, state, action, h, seed + pair_index * 100000 + action_index * 100 + h)
                         for action_index, action in enumerate(actions)] for h in horizons}
        for action_index, action in enumerate(actions):
            labels = {}
            for horizon in horizons:
                endpoint = endpoints[horizon][action_index]
                before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
                after_xy = float(np.linalg.norm(endpoint[2:4] - goal[2:4]))
                before_angle = angle_error(state[4], goal[4])
                after_angle = angle_error(endpoint[4], goal[4])
                if phase_switch_xy is not None and before_xy > float(phase_switch_xy):
                    pose = (
                        after_xy < before_xy - 1e-5
                        and after_angle <= before_angle + float(translation_angle_tolerance)
                        and after_xy <= 45.0
                    )
                else:
                    pose = after_angle < before_angle - 1e-5 and after_xy <= min(45.0, before_xy + float(xy_tube_margin))
                motion = np.linalg.norm(endpoint[2:4] - state[2:4]) >= 0.5
                contact = mode_for_state(endpoint) == "contact"
                labels[str(horizon)] = {
                    "pose": int(pose),
                    "turn": int(after_angle < before_angle - 1e-5),
                    "motion": int(motion),
                    "contact": int(contact),
                    "joint": int(pose and motion and (contact if mode_for_state(state) == "contact" else True)),
                }
            rows.append({
                "state": state,
                "action": np.asarray(action, dtype=np.float64),
                "previous_action": np.zeros(2, dtype=np.float64),
                "goal": goal,
                "features": contact_mode_features(state, action, np.zeros(2), goal),
                "labels": labels,
                "mode": mode_for_state(state),
            })
    return rows


def aggregate_rollout(env, models: dict[str, Any], variant: str, state: np.ndarray, goal: np.ndarray, actions: np.ndarray, steps: int, seed: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    observation, _ = env.reset(seed=seed, options={"state": np.asarray(state).copy(), "goal_state": np.asarray(goal).copy()})
    current = np.asarray(observation["state"], dtype=np.float64).copy()
    previous = np.zeros(2, dtype=np.float64)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    trace = []
    for step in range(steps):
        pairs.append((current.copy(), np.asarray(goal).copy()))
        rows = candidate_rows(current, goal, previous, actions)
        selected, info = score_variant(variant, models, rows)
        action = actions[selected].astype(np.float32)
        observation, _, terminated, truncated, _ = env.step(action)
        current = np.asarray(observation["state"], dtype=np.float64).copy()
        previous = action.astype(np.float64)
        trace.append({"step": step, "mode": mode_for_state(current), "selected": selected, **info})
        if terminated or truncated:
            break
    return pairs, {"steps": len(trace), "trace": trace, "final_state": current.tolist()}


def evaluate(
    env,
    models: dict[str, Any],
    variants: tuple[str, ...],
    cases: list[dict[str, Any]],
    actions: np.ndarray,
    steps: int,
    seed: int,
    draws: int,
    xy_tube_margin: float,
    phase_switch_xy: float | None,
    translation_angle_tolerance: float,
) -> dict[str, Any]:
    reports = []
    for case in cases:
        for variant in variants:
            reports.append(
                run_episode(
                    env,
                    variant,
                    models,
                    case["state"],
                    case["goal"],
                    actions,
                    steps,
                    seed + case["case"],
                    xy_tube_margin=xy_tube_margin,
                    phase_switch_xy=phase_switch_xy,
                    translation_angle_tolerance=translation_angle_tolerance,
                )
            )
    summaries = []
    for index, variant in enumerate(variants):
        subset = [row for row in reports if row["variant"] == variant]
        summaries.append({
            "variant": variant,
            "metrics": {
                name: bootstrap(np.asarray([row[name] for row in subset]), draws, seed + index * 100 + metric_index)
                for metric_index, name in enumerate(("strict_success", "final_xy", "final_angle", "contact_fraction"))
            },
            "rows": subset,
        })
    return {"summaries": summaries, "reports": reports}


def main() -> None:
    args = parse_args()
    if args.rounds < 1 or args.train_states < 8 or args.aggregate_episodes < 4 or args.test_episodes < 8:
        raise ValueError("insufficient split sizes")
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    horizons = (1, 2, 4)
    actions = torque_action_bank(args.directions, magnitudes=tuple(args.magnitudes))
    env = make_state_only_env()
    aggregation_log = []
    try:
        train_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 100000)
        aggregate_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 200000)
        test_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 300000)
        train_states = select_balanced_states(train_pool, args.train_states, args.seed + 1, ("contact",))
        aggregate_states = select_balanced_states(aggregate_pool, args.aggregate_episodes, args.seed + 2, ("contact",))
        test_states = select_balanced_states(test_pool, args.test_episodes, args.seed + 3, ("contact",))
        rng = np.random.default_rng(args.seed + 400000)
        aggregate_cases = [{"case": index, "state": state.copy(), "goal": goal_for_state(state, index, rng, closure_only=True)} for index, state in enumerate(aggregate_states)]
        test_cases = [{"case": index, "state": state.copy(), "goal": goal_for_state(state, index, rng, closure_only=True)} for index, state in enumerate(test_states)]
        initial_pairs = [(state.copy(), goal_for_state(state, index, np.random.default_rng(args.seed + 500000 + index), closure_only=True)) for index, state in enumerate(train_states)]
        rows = make_goal_rows(
            env,
            initial_pairs,
            actions,
            horizons,
            args.seed + 600000,
            args.xy_tube_margin,
            args.phase_switch_xy,
            args.translation_angle_tolerance,
        )
        initial_models = fit_models(rows, horizons, args.trees, args.seed + 699000)
        round_models = []
        for round_index in range(args.rounds):
            models = fit_models(rows, horizons, args.trees, args.seed + 700000 + round_index)
            round_models.append(models)
            round_pairs = []
            traces = []
            for case in aggregate_cases:
                pairs, trace = aggregate_rollout(env, models, "h1_soft_contact", case["state"], case["goal"], actions, args.closure_steps, args.seed + 800000 + round_index * 10000 + case["case"])
                round_pairs.extend(pairs)
                traces.append(trace)
            aggregation_log.append({"round": round_index, "base_rows": len(rows), "visited_pairs": len(round_pairs), "modes": {mode: int(sum(pair[0] is not None and mode_for_state(pair[0]) == mode for pair in round_pairs)) for mode in ("free", "approach", "contact")}})
            rows.extend(
                make_goal_rows(
                    env,
                    round_pairs,
                    actions,
                    horizons,
                    args.seed + 900000 + round_index * 100000,
                    args.xy_tube_margin,
                    args.phase_switch_xy,
                    args.translation_angle_tolerance,
                )
            )
    finally:
        env.close()
    env = make_state_only_env()
    try:
        final_models = fit_models(rows, horizons, args.trees, args.seed + 999000)
        variants = (
            "h1_joint",
            "h1_soft_contact",
            "h1_soft_contact_feedback",
            "h1_turn_soft_contact",
            "h2_soft_contact",
            "h4_soft_contact",
            "multi_soft_contact",
        )
        base_evaluation = evaluate(
            env, initial_models, variants, test_cases, actions,
            args.closure_steps,
            args.seed + 1000000,
            args.bootstrap,
            args.xy_tube_margin,
            args.phase_switch_xy,
            args.translation_angle_tolerance,
        )
        final_evaluation = evaluate(
            env, final_models, variants, test_cases, actions,
            args.closure_steps,
            args.seed + 1000000,
            args.bootstrap,
            args.xy_tube_margin,
            args.phase_switch_xy,
            args.translation_angle_tolerance,
        )
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "initial_train_contact_states": len(train_states),
            "aggregation_episodes": len(aggregate_cases),
            "fresh_test_contact_episodes": len(test_cases),
            "action_count": len(actions),
            "directions": args.directions,
            "magnitudes": [float(value) for value in args.magnitudes],
            "closure_steps": args.closure_steps,
            "rounds": args.rounds,
            "xy_tube_margin": float(args.xy_tube_margin),
            "phase_switch_xy": None if args.phase_switch_xy is None else float(args.phase_switch_xy),
            "translation_angle_tolerance": float(args.translation_angle_tolerance),
            "supervision_horizons": list(horizons),
            "runtime_candidate_successor_verification": False,
            "three_way_state_stream_split": True,
            "wall_seconds": time.perf_counter() - started,
        },
        "aggregation": aggregation_log,
        "final_training_rows": len(rows),
        "evaluation": {
            "base": base_evaluation["summaries"],
            "dagger": final_evaluation["summaries"],
        },
        "paired_dagger_minus_base": {
            variant: {
                name: bootstrap(
                    np.asarray([
                        new[name] - old[name]
                        for old, new in zip(
                            [row for row in base_evaluation["reports"] if row["variant"] == variant],
                            [row for row in final_evaluation["reports"] if row["variant"] == variant],
                        )
                    ]),
                    args.bootstrap,
                    args.seed + 1200000 + variant_index * 100 + metric_index,
                )
                for metric_index, name in enumerate(("strict_success", "final_xy", "final_angle", "contact_fraction"))
            }
            for variant_index, variant in enumerate(variants)
        },
        "sanity_checks": {
            "same_test_cases_and_goals": True,
            "same_action_fan": True,
            "runtime_candidate_successor_verification": False,
            "all_finite": bool(all(np.isfinite(value) for section in (base_evaluation["summaries"], final_evaluation["summaries"]) for item in section for metric in item["metrics"].values() for value in metric.values() if isinstance(value, (int, float)))),
        },
        "claim_boundary": "Privileged simulator-state dataset-aggregated contact-pose closure. A positive result would support robustness to visited-state distribution shift, not visual LeWM planning, universal world modeling, or physical causality.",
    }
    (output / "dagger_pose_adapter.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# Dataset-aggregated learned pose adapter", "", "| variant | strict success | final XY | final angle | contact fraction |", "|---|---:|---:|---:|---:|"]
    for item in report["evaluation"]["dagger"]:
        metrics = item["metrics"]
        lines.append("| %s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] |" % (
            item["variant"],
            metrics["strict_success"]["estimate"], metrics["strict_success"]["ci95_low"], metrics["strict_success"]["ci95_high"],
            metrics["final_xy"]["estimate"], metrics["final_xy"]["ci95_low"], metrics["final_xy"]["ci95_high"],
            metrics["final_angle"]["estimate"], metrics["final_angle"]["ci95_low"], metrics["final_angle"]["ci95_high"],
            metrics["contact_fraction"]["estimate"], metrics["contact_fraction"]["ci95_low"], metrics["contact_fraction"]["ci95_high"],
        ))
    lines += ["", report["claim_boundary"], ""]
    (output / "dagger_pose_adapter.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "aggregation": aggregation_log, "evaluation": report["evaluation"], "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
