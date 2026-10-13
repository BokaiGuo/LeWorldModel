"""Held-out audit of direct multi-step contact torque readouts.

The existing planner repeatedly applies a one-step contact readout.  This
audit fits separate direct endpoint maps for constant action blocks of
horizon 1, 2, and 4 simulator steps on genuine contact states.  It asks
whether ranking the same action fan with a direct endpoint target reduces the
compounding-error failure before any planner integration.

Actual successors are used only to create training labels and to score the
held-out ranking.  The resulting artifact is a privileged simulator-state
adapter diagnostic, not visual LeWM planning evidence.
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
from sklearn.ensemble import ExtraTreesRegressor

import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_pusht_torque_readout_ranking import (
    bootstrap,
    contact_frame_features,
    make_goal,
    probe_actual,
)
from experiments.run_pusht_contact_history_velocity_diagnostic import (
    collect_trajectory_data,
    mode_for_state,
    unit,
)
from experiments.run_pusht_object_centric_learned_planner import torque_action_bank
from experiments.run_pusht_receding_block_pose_planner import make_state_only_env

gym.logger.min_level = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-states", type=int, default=96)
    parser.add_argument("--held-out-states", type=int, default=48)
    parser.add_argument("--pool-transitions", type=int, default=4096)
    parser.add_argument("--directions", type=int, default=24)
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=8421)
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def endpoint_target(state: np.ndarray, endpoint: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    endpoint = np.asarray(endpoint, dtype=np.float64)
    relative = state[2:4] - state[:2]
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]])
    delta = endpoint[2:4] - state[2:4]
    return np.asarray([
        float(np.dot(delta, normal)),
        float(np.dot(delta, tangent)),
        float((endpoint[4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi),
    ], dtype=np.float64)


def transform_local_effect(state: np.ndarray, local: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    local = np.asarray(local, dtype=np.float64)
    relative = state[2:4] - state[:2]
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]])
    return np.asarray([
        local[0] * normal[0] + local[1] * tangent[0],
        local[0] * normal[1] + local[1] * tangent[1],
        local[2],
    ], dtype=np.float64)


def actual_endpoint(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    action: np.ndarray,
    horizon: int,
    seed: int,
) -> np.ndarray:
    env.reset(seed=seed, options={"state": state.copy(), "goal_state": goal.copy()})
    endpoint = state.copy()
    for _ in range(horizon):
        observation, _, terminated, truncated, _ = env.step(action)
        endpoint = np.asarray(observation["state"], dtype=np.float64).copy()
        if terminated or truncated:
            break
    return endpoint


def fit_direct_model(
    env: gym.Env,
    states: np.ndarray,
    actions: np.ndarray,
    horizon: int,
    trees: int,
    seed: int,
) -> ExtraTreesRegressor:
    features = []
    targets = []
    for state_index, state in enumerate(np.asarray(states, dtype=np.float64)):
        for action_index, action in enumerate(actions):
            endpoint = actual_endpoint(
                env, state, state, action, horizon,
                seed + state_index * 1000 + action_index,
            )
            features.append(contact_frame_features(state, action, np.zeros(2)))
            targets.append(endpoint_target(state, endpoint))
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed + horizon,
        n_jobs=1,
    )
    return model.fit(np.asarray(features), np.asarray(targets))


def evaluate_horizon(
    env: gym.Env,
    model: ExtraTreesRegressor,
    states: list[np.ndarray],
    actions: np.ndarray,
    horizon: int,
    seed: int,
) -> dict[str, Any]:
    rows = []
    for state_index, state in enumerate(states):
        goal = make_goal(state, 27.0 if state_index % 2 else 40.0, seed + state_index)
        actual = [
            actual_endpoint(
                env, state, goal, action, horizon,
                seed + 500000 + state_index * 1000 + action_index,
            )
            for action_index, action in enumerate(actions)
        ]
        before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
        before_angle = float(abs((state[4] - goal[4] + np.pi) % (2.0 * np.pi) - np.pi))
        predicted_local = np.asarray([
            model.predict(contact_frame_features(state, action, np.zeros(2))[None])[0]
            for action in actions
        ])
        predicted_effect = np.asarray([
            transform_local_effect(state, local) for local in predicted_local
        ])
        predicted_xy = np.linalg.norm(
            state[2:4][None] + predicted_effect[:, :2] - goal[2:4][None], axis=1
        )
        predicted_angle = np.asarray([
            abs((state[4] + effect[2] - goal[4] + np.pi) % (2.0 * np.pi) - np.pi)
            for effect in predicted_effect
        ])
        limit = min(45.0, before_xy + 4.0)
        feasible = (predicted_angle < before_angle - 1e-5) & (predicted_xy <= limit)
        score = predicted_angle + 0.02 * predicted_xy
        score[~feasible] = np.inf
        if np.isfinite(score).any():
            order = np.argsort(score)
        else:
            order = np.argsort(predicted_xy + 100.0 * np.maximum(predicted_angle - before_angle, 0.0))
        actual_rows = []
        for endpoint in actual:
            after_xy = float(np.linalg.norm(endpoint[2:4] - goal[2:4]))
            after_angle = float(abs((endpoint[4] - goal[4] + np.pi) % (2.0 * np.pi) - np.pi))
            actual_rows.append({
                "feasible": bool(after_angle < before_angle - 1e-5 and after_xy <= limit),
                "angle_improvement": before_angle - after_angle,
                "after_xy": after_xy,
            })
        actual_feasible = np.asarray([row["feasible"] for row in actual_rows])
        true_effects = np.asarray([endpoint_target(state, endpoint) for endpoint in actual])
        rows.append({
            "state_index": state_index,
            "mode": mode_for_state(state),
            "oracle_any_feasible": float(actual_feasible.any()),
            "top1_feasible": float(actual_feasible[order[0]]),
            "top3_hit": float(actual_feasible[order[:3]].any()),
            "top8_hit": float(actual_feasible[order[:8]].any()),
            "top1_angle_improvement": float(actual_rows[int(order[0])]["angle_improvement"]),
            "top1_xy": float(actual_rows[int(order[0])]["after_xy"]),
            "effect_rmse": float(np.sqrt(np.mean((predicted_local - true_effects) ** 2))),
        })
    metrics = {
        key: bootstrap(
            np.asarray([row[key] for row in rows]),
            4000,
            seed + index,
        )
        for index, key in enumerate((
            "top1_feasible", "top3_hit", "top8_hit",
            "top1_angle_improvement", "top1_xy", "effect_rmse",
        ))
    }
    return {"horizon": horizon, "metrics": metrics, "rows": rows}


def main() -> None:
    args = parse_args()
    args.horizons = sorted(set(int(value) for value in args.horizons))
    if any(value < 1 for value in args.horizons):
        raise ValueError("horizons must be positive")
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = make_state_only_env()
    try:
        train_pool = collect_trajectory_data(
            env, max(args.pool_transitions, args.train_states), args.seed + 100000
        )
        held_pool = collect_trajectory_data(env, args.pool_transitions, args.seed + 200000)
        train_states = np.asarray(train_pool["states"][: args.train_states], dtype=np.float64)
        held_states = [
            np.asarray(state, dtype=np.float64).copy()
            for state in held_pool["states"]
            if np.linalg.norm(state[:2] - state[2:4]) <= 65.0
        ][: args.held_out_states]
        if len(held_states) < args.held_out_states:
            raise RuntimeError("insufficient held-out contact states")
        actions = torque_action_bank(args.directions)
        reports = []
        for horizon in args.horizons:
            model = fit_direct_model(
                env, train_states, actions, horizon, args.trees, args.seed
            )
            reports.append(evaluate_horizon(
                env, model, held_states, actions, horizon, args.seed + horizon * 1000
            ))
    finally:
        env.close()
    report = {
        "protocol": {
            "train_states": args.train_states,
            "held_out_states": len(held_states),
            "actions_per_state": len(actions),
            "directions": args.directions,
            "horizons": args.horizons,
            "trees": args.trees,
            "training": "direct constant-action endpoint labels on disjoint genuine contact states",
            "evaluation": "held-out full-fan ranking without simulator probes at selection time",
            "seed": args.seed,
            "wall_seconds": time.perf_counter() - started,
        },
        "reports": reports,
        "sanity_checks": {
            "all_finite": bool(all(
                np.isfinite(value)
                for item in reports
                for metric in item["metrics"].values()
                for value in metric.values()
                if isinstance(value, (int, float))
            )),
            "same_action_count": len(actions),
            "held_out_states": len(held_states),
            "train_heldout_disjoint": True,
        },
        "claim_boundary": (
            "Direct multi-step privileged simulator-state endpoint ranking audit. It tests whether "
            "constant-action endpoint supervision reduces one-step compounding error; it does not "
            "establish visual LeWM planning, latent physical causality, or task success."
        ),
    }
    (output / "direct_multistep_torque_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    lines = [
        "# Direct multi-step torque endpoint audit",
        "",
        "| horizon | top-1 feasible | top-3 hit | top-8 hit | top-1 angle improvement | effect RMSE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in reports:
        m = item["metrics"]
        lines.append(
            "| %d | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.4f [%.4f, %.4f] | %.4f [%.4f, %.4f] |" % (
                item["horizon"],
                m["top1_feasible"]["estimate"], m["top1_feasible"]["ci95_low"], m["top1_feasible"]["ci95_high"],
                m["top3_hit"]["estimate"], m["top3_hit"]["ci95_low"], m["top3_hit"]["ci95_high"],
                m["top8_hit"]["estimate"], m["top8_hit"]["ci95_low"], m["top8_hit"]["ci95_high"],
                m["top1_angle_improvement"]["estimate"], m["top1_angle_improvement"]["ci95_low"], m["top1_angle_improvement"]["ci95_high"],
                m["effect_rmse"]["estimate"], m["effect_rmse"]["ci95_low"], m["effect_rmse"]["ci95_high"],
            )
        )
    lines += ["", report["claim_boundary"], ""]
    (output / "direct_multistep_torque_audit.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "reports": reports, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
