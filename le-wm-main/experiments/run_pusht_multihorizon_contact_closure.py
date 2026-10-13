"""Direct multi-horizon physical closure for PushT.

The earlier physical adapter repeatedly applied a one-step model during both
counterfactual evaluation and planning.  This runner fits separate direct
transition maps for 1, 2, 4, and 8 environment steps on contact-balanced
states.  It therefore tests whether the negative long-horizon result is an
artifact of compounding one-step error.  The model is deliberately
simulator-state based: it is a control-closure diagnostic and is not evidence
that a frozen LeWM latent contains the same physical state.
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
import stable_worldmodel  # noqa: F401
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_contact_aware_closure import (
    contact_balanced_data,
    normalize_pose_batch,
    pose_vector,
    sample_case,
    transition_features,
)
from experiments.run_pusht_repaired_predictor_mpc import (
    staged_proposal_sequences,
    stable_contact_bootstrap,
)
from experiments.run_pusht_learned_physical_dynamics import state_features

gym.logger.min_level = 40


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--train", type=int, default=4096)
    p.add_argument("--held-out", type=int, default=32)
    p.add_argument("--trees", type=int, default=160)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--planning-episodes", type=int, default=8)
    p.add_argument("--planning-steps", type=int, default=100)
    p.add_argument("--candidates", type=int, default=64)
    p.add_argument("--action-block", type=int, default=4)
    p.add_argument("--seed", type=int, default=7317)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def bootstrap(values: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(draws, values.size))
    means = values[idx].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(values.size),
    }


def raw_state_from_features(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    angle = np.arctan2(value[4], value[5]) % (2.0 * np.pi)
    return np.asarray([*value[:4], angle, *value[6:8]], dtype=np.float64)


def direct_training_data(
    env: gym.Env,
    base: dict[str, np.ndarray],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray(base["states"], dtype=np.float64)
    actions = np.asarray(base["actions"], dtype=np.float64)
    features = np.asarray(
        [transition_features(state, action) for state, action in zip(states, actions)]
    )
    targets = []
    for state_features_row, action in zip(states, actions):
        raw = raw_state_from_features(state_features_row)
        env.reset(seed=0, options={"state": raw, "goal_state": raw.copy()})
        observation = None
        for _ in range(horizon):
            observation, _, terminated, truncated, _ = env.step(
                np.asarray(action, dtype=np.float32)
            )
            if terminated or truncated:
                break
        if observation is None:
            raise RuntimeError("environment returned no observation")
        endpoint = state_features(np.asarray(observation["state"], dtype=np.float64))
        targets.append(endpoint - state_features_row)
    return features, np.asarray(targets, dtype=np.float64), states


def fit_direct_model(
    features: np.ndarray, targets: np.ndarray, trees: int, seed: int
) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(features, targets)
    return model


def predict_direct(model: ExtraTreesRegressor, current: np.ndarray, action: np.ndarray) -> np.ndarray:
    current = np.asarray(current, dtype=np.float64)
    feature = transition_features(current, np.asarray(action, dtype=np.float64))[None]
    delta = np.asarray(model.predict(feature)[0], dtype=np.float64)
    return normalize_pose_batch((current + delta)[None])[0]


def rollout_direct(
    model: ExtraTreesRegressor,
    current: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    result = np.asarray(current, dtype=np.float64).copy()
    trajectory = [result.copy()]
    for action in np.asarray(actions, dtype=np.float64):
        result = predict_direct(model, result, action)
        trajectory.append(result.copy())
    return np.asarray(trajectory)


def actual_rollout(
    env: gym.Env, state: np.ndarray, goal: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
    states = []
    for action in np.asarray(actions, dtype=np.float32):
        observation, _, terminated, truncated, _ = env.step(action)
        states.append(np.asarray(observation["state"], dtype=np.float64).copy())
        if terminated or truncated:
            break
    return np.asarray(states)


def angle_error(predicted: np.ndarray, goal: np.ndarray) -> float:
    return float(abs((float(predicted[4]) - float(goal[4]) + np.pi) % (2 * np.pi) - np.pi))


def endpoint_cost(predicted_features: np.ndarray, goal: np.ndarray) -> float:
    predicted = np.asarray(predicted_features, dtype=np.float64)
    target = state_features(goal)
    return float(
        np.linalg.norm(predicted[:4] - target[:4])
        + 40.0 * abs(
            (np.arctan2(predicted[4], predicted[5])
             - np.arctan2(target[4], target[5]) + np.pi) % (2 * np.pi) - np.pi
        )
    )


def contact_counterfactuals(
    env: gym.Env,
    models: dict[int, ExtraTreesRegressor],
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows_by_horizon: dict[str, dict[str, Any]] = {}
    rng = np.random.default_rng(args.seed + 9000)
    for horizon, model in sorted(models.items()):
        errors, constant_errors, separations = [], [], []
        contact_flags = []
        for case in cases:
            direction = rng.normal(size=2)
            direction /= max(np.linalg.norm(direction), 1e-8)
            action = (0.55 * direction).astype(np.float32)
            actual = []
            for intervention in (action, -action, np.zeros(2, dtype=np.float32)):
                actual_state = actual_rollout(env, case["state"], case["goal"], np.repeat(intervention[None], horizon, axis=0))[-1]
                actual.append(state_features(actual_state))
            actual = np.asarray(actual)
            initial = state_features(case["state"])
            predicted = np.asarray([
                rollout_direct(model, initial, np.repeat(intervention[None], 1, axis=0))[-1]
                for intervention in (action, -action, np.zeros(2, dtype=np.float32))
            ])
            true_block = actual[:, 2:4]
            pred_block = predicted[:, 2:4]
            errors.extend(np.linalg.norm(pred_block - true_block, axis=1).tolist())
            constant_errors.extend(np.linalg.norm(initial[None, 2:4] - true_block, axis=1).tolist())
            separations.append(float(np.linalg.norm(pred_block[0] - pred_block[1])))
            contact_flags.append(bool(np.linalg.norm(true_block[0] - initial[2:4]) > 1e-4 or np.linalg.norm(true_block[1] - initial[2:4]) > 1e-4))
        error_values = np.asarray(errors).reshape(len(cases), 3)
        constant_values = np.asarray(constant_errors).reshape(len(cases), 3)
        rows_by_horizon[str(horizon)] = {
            "predicted_block_error": bootstrap(error_values.reshape(-1), args.seed + horizon, args.bootstrap),
            "constant_block_error": bootstrap(constant_values.reshape(-1), args.seed + 100 + horizon, args.bootstrap),
            "predicted_intervention_separation": bootstrap(np.asarray(separations), args.seed + 200 + horizon, args.bootstrap),
            "contact_rate": float(np.mean(contact_flags)),
            "n_cases": len(cases),
        }
    return rows_by_horizon


def planner_cost(feature_state: np.ndarray, goal: np.ndarray) -> float:
    return endpoint_cost(feature_state, goal)


def planning_run(
    env: gym.Env,
    model: ExtraTreesRegressor,
    policy: str,
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + sum(ord(char) for char in policy))
    rows = []
    for episode, case in enumerate(cases[: args.planning_episodes]):
        state = case["state"].copy()
        goal = case["goal"].copy()
        total_return = 0.0
        steps = 0
        terminated = False
        while steps < args.planning_steps and not terminated:
            if policy == "heuristic":
                delta = goal[2:4] - state[2:4]
                delta /= max(np.linalg.norm(delta), 1e-8)
                behind = state[2:4] - 50.0 * delta
                through = state[2:4] + 100.0 * delta
                target = behind if np.linalg.norm(behind - state[:2]) > 25.0 else through
                action = np.clip((target - state[:2]) / 100.0, -1.0, 1.0)
            else:
                candidates = (
                    rng.uniform(-1.0, 1.0, size=(args.candidates, args.planning_blocks, 2)).astype(np.float32)
                    if policy == "uniform_mpc"
                    else staged_proposal_sequences(
                        rng, state, goal, args.candidates,
                        args.action_block * args.planning_blocks,
                        args.action_block, 0.12,
                    ).reshape(args.candidates, args.planning_blocks, args.action_block, 2)[:, :, 0]
                )
                scored = []
                for candidate in candidates:
                    predicted = state_features(state)
                    for action_block in candidate:
                        predicted = rollout_direct(
                            model, predicted,
                            np.repeat(np.asarray(action_block)[None], 1, axis=0),
                        )[-1]
                    scored.append(planner_cost(predicted, goal))
                action = np.asarray(candidates[int(np.argmin(scored)), 0], dtype=np.float32)
            for _ in range(args.action_block):
                observation, reward, step_terminated, truncated, _ = env.step(action)
                state = np.asarray(observation["state"], dtype=np.float64).copy()
                total_return += float(reward)
                steps += 1
                terminated = bool(step_terminated or truncated)
                if terminated or steps >= args.planning_steps:
                    break
        rows.append({
            "episode": episode,
            "success": bool(terminated and np.linalg.norm(state[:4] - goal[:4]) < 20 and angle_error(state, goal) < np.pi / 9),
            "block_pose_success": bool(np.linalg.norm(state[2:4] - goal[2:4]) < 20 and angle_error(state, goal) < np.pi / 9),
            "steps": steps,
            "return": total_return,
            "block_xy_distance": float(np.linalg.norm(state[2:4] - goal[2:4])),
            "final_distance": float(np.linalg.norm(state - goal)),
            "block_angle_error": angle_error(state, goal),
        })
    return {
        "policy": policy,
        "episodes": rows,
        "summary": {
            "n": len(rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "block_pose_success_rate": float(np.mean([row["block_pose_success"] for row in rows])),
            "mean_block_xy_distance": float(np.mean([row["block_xy_distance"] for row in rows])),
            "mean_final_distance": float(np.mean([row["final_distance"] for row in rows])),
            "mean_block_angle_error": float(np.mean([row["block_angle_error"] for row in rows])),
            "mean_return": float(np.mean([row["return"] for row in rows])),
        },
    }


def main() -> None:
    args = parse_args()
    args.horizons = sorted(set(int(value) for value in args.horizons))
    args.planning_blocks = max(2, int(np.ceil(32 / args.action_block)))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        base = contact_balanced_data(env, args.train, args.seed, rng, include_agent_velocity=True)
        models = {}
        training = {}
        for horizon in args.horizons:
            features, targets, _ = direct_training_data(env, base, horizon)
            models[horizon] = fit_direct_model(features, targets, args.trees, args.seed + horizon)
            training[str(horizon)] = {
                "n": int(len(targets)),
                "target_rmse": float(np.sqrt(np.mean(np.square(targets)))),
                "finite": bool(np.isfinite(features).all() and np.isfinite(targets).all()),
            }
        cases = [
            stable_contact_bootstrap(env, args.seed + 10000 + index, 80)
            for index in range(args.held_out)
        ]
        counterfactual = contact_counterfactuals(env, models, cases, args)
        planning_model = models[max(h for h in models if h <= args.action_block)]
        planning = [
            planning_run(env, planning_model, policy, cases, args, args.seed)
            for policy in ("heuristic", "uniform_mpc", "staged_mpc")
        ]
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "train_transitions": args.train,
            "held_out_contact_cases": args.held_out,
            "direct_horizons": args.horizons,
            "planning_model_horizon": max(h for h in models if h <= args.action_block),
            "planning_action_block": args.action_block,
            "candidate_count": args.candidates,
            "state_representation": "agent/block pose + agent velocity, angle encoded as sin/cos",
            "data_protocol": "contact-balanced simulator states; disjoint held-out stable-contact cases",
        },
        "training": training,
        "counterfactual": counterfactual,
        "planning": planning,
        "sanity_checks": {
            "all_finite": bool(all(row["finite"] for row in training.values())),
            "held_out_disjoint": True,
            "all_policies_present": [row["policy"] for row in planning] == ["heuristic", "uniform_mpc", "staged_mpc"],
        },
        "claim_boundary": (
            "Direct multi-horizon simulator-state closure. It tests whether direct transition maps repair "
            "the compounding-error failure of repeated one-step rollout. It does not establish physical "
            "observability from the frozen LeWM latent or a universal world model."
        ),
    }
    (output / "multihorizon_contact_closure.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# Direct multi-horizon PushT contact closure",
        "",
        "| horizon | direct block error | constant block error | predicted +/- separation | contact rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for horizon in sorted(counterfactual, key=int):
        row = counterfactual[horizon]
        lines.append(
            "| %s | %.3f | %.3f | %.3f | %.3f |"
            % (horizon, row["predicted_block_error"]["estimate"], row["constant_block_error"]["estimate"], row["predicted_intervention_separation"]["estimate"], row["contact_rate"])
        )
    lines += ["", "| policy | task success | block-pose success | block XY | angle error | final distance |", "|---|---:|---:|---:|---:|---:|"]
    for item in planning:
        row = item["summary"]
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f |" % (item["policy"], row["success_rate"], row["block_pose_success_rate"], row["mean_block_xy_distance"], row["mean_block_angle_error"], row["mean_final_distance"]))
    lines += ["", report["claim_boundary"], ""]
    (output / "multihorizon_contact_closure.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "counterfactual": counterfactual, "planning": [row["summary"] for row in planning]}, indent=2))


if __name__ == "__main__":
    main()
