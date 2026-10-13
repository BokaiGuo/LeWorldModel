"""Deep diagnosis of the remaining contact-dynamics failure boundary.

The contact-stratified audit showed that mode-specialist regressors repair
free-space transitions but can make normal/tangential contact worse at long
horizons.  This runner tests three different explanations on genuine
trajectory states:

* ``state_action``: the full simulator state and current action;
* ``object_centric``: block pose plus relative contact geometry, excluding
  absolute agent pose and agent velocity from the input;
* ``history``: the full state/action features augmented with the previous
  action and action increment.

Every model predicts the same full physical-state delta and is evaluated on
the same held-out states with paired ``+a/-a/zero`` interventions.  The
protocol is a privileged simulator-state physical-adapter diagnostic.  It is
not a frozen-LeWM latent experiment or evidence of visual online planning.
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

from experiments.run_pusht_learned_physical_dynamics import state_features
from experiments.run_pusht_receding_block_pose_planner import make_state_only_env
from experiments.pusht_state import raw_angle_error, raw_block_angle

gym.logger.min_level = 40

MODE_NAMES = ("free", "approach", "contact")
MODEL_NAMES = ("state_action", "object_centric", "history")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-transitions", type=int, default=768)
    parser.add_argument("--held-out", type=int, default=24)
    parser.add_argument("--trees", type=int, default=96)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--action-magnitude", type=float, default=0.55)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261008)
    return parser.parse_args()


def unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value / max(float(np.linalg.norm(value)), 1e-8)


def mode_for_state(state: np.ndarray) -> str:
    distance = float(np.linalg.norm(np.asarray(state)[2:4] - np.asarray(state)[:2]))
    if distance > 90.0:
        return "free"
    if distance > 45.0:
        return "approach"
    return "contact"


def angle_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(raw_angle_error(raw_block_angle(first), raw_block_angle(second)))


def normalize_features(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).copy()
    norm = np.linalg.norm(value[..., 4:6], axis=-1, keepdims=True)
    value[..., 4:6] /= np.maximum(norm, 1e-8)
    value[..., :4] = np.clip(value[..., :4], 5.0, 507.0)
    value[..., 6:8] = np.clip(value[..., 6:8], -500.0, 500.0)
    return value


def contact_geometry_features(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Features invariant to global translation but sensitive to contact mode."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    relative = state[2:4] - state[:2]
    distance = float(np.linalg.norm(relative))
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]])
    # The raw environment state stores one scalar angle at index 4; index 5 is
    # agent velocity and must never enter the orientation chart.
    block_angle = raw_block_angle(state)
    rotation = np.asarray(
        [[np.cos(block_angle), np.sin(block_angle)],
         [-np.sin(block_angle), np.cos(block_angle)]]
    )
    return np.concatenate([
        state_features(state)[2:6],
        action,
        relative,
        [distance, float(np.dot(action, normal)), float(np.dot(action, tangent))],
        rotation @ relative,
        rotation @ action,
    ])


def feature_vector(
    state: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    model_name: str,
) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    previous_action = np.asarray(previous_action, dtype=np.float64)
    if model_name == "object_centric":
        return contact_geometry_features(state, action)
    base = np.concatenate([state_features(state), action])
    if model_name == "history":
        return np.concatenate([base, previous_action, action - previous_action])
    if model_name == "state_action":
        return base
    raise ValueError(f"unknown model: {model_name}")


def choose_behavior_action(
    state: np.ndarray,
    step: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate genuine approach/contact trajectories with changing actions."""
    relative = np.asarray(state[2:4] - state[:2], dtype=np.float64)
    distance = float(np.linalg.norm(relative))
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]])
    if distance > 75.0:
        action = normal
    elif step % 4 in (0, 1):
        action = 0.62 * normal + 0.38 * tangent
    elif step % 4 == 2:
        action = -0.25 * normal + 0.85 * tangent
    else:
        action = -0.45 * normal - 0.65 * tangent
    action = action + 0.08 * rng.normal(size=2)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def collect_trajectory_data(
    env: gym.Env,
    count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    previous_actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    modes: list[str] = []
    episode = 0
    while len(states) < count:
        observation, _ = env.reset(seed=seed + episode)
        current = np.asarray(observation["state"], dtype=np.float64).copy()
        previous = np.zeros(2, dtype=np.float32)
        rng = np.random.default_rng(seed + 1000 + episode)
        episode += 1
        for step in range(60):
            if len(states) >= count:
                break
            action = choose_behavior_action(current, step, rng)
            mode = mode_for_state(current)
            observation, _, terminated, truncated, _ = env.step(action)
            next_state = np.asarray(observation["state"], dtype=np.float64).copy()
            states.append(current.copy())
            actions.append(action.copy())
            previous_actions.append(previous.copy())
            next_states.append(next_state.copy())
            modes.append(mode)
            current = next_state
            previous = action
            if terminated or truncated:
                break
    return {
        "states": np.asarray(states, dtype=np.float64),
        "actions": np.asarray(actions, dtype=np.float64),
        "previous_actions": np.asarray(previous_actions, dtype=np.float64),
        "next_states": np.asarray(next_states, dtype=np.float64),
        "modes": np.asarray(modes),
    }


def fit_models(
    data: dict[str, np.ndarray],
    trees: int,
    seed: int,
) -> dict[str, ExtraTreesRegressor]:
    targets = np.asarray([
        state_features(next_state) - state_features(state)
        for state, next_state in zip(data["states"], data["next_states"])
    ])
    models: dict[str, ExtraTreesRegressor] = {}
    for index, model_name in enumerate(MODEL_NAMES):
        features = np.asarray([
            feature_vector(state, action, previous, model_name)
            for state, action, previous in zip(
                data["states"], data["actions"], data["previous_actions"]
            )
        ])
        model = ExtraTreesRegressor(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=seed + index,
            n_jobs=1,
        )
        model.fit(features, targets)
        models[model_name] = model
    return models


def predict_step(
    model: ExtraTreesRegressor,
    state: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    model_name: str,
) -> np.ndarray:
    features = feature_vector(state, action, previous_action, model_name)[None]
    delta = np.asarray(model.predict(features)[0], dtype=np.float64)
    predicted = state_features(state) + delta
    return normalize_features(predicted)


def rollout_model(
    model: ExtraTreesRegressor,
    state: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    horizon: int,
    model_name: str,
) -> np.ndarray:
    current = np.asarray(state, dtype=np.float64).copy()
    previous = np.asarray(previous_action, dtype=np.float64).copy()
    for _ in range(horizon):
        current = predict_step(model, current, action, previous, model_name)
        previous = np.asarray(action, dtype=np.float64).copy()
        current = np.asarray([
            current[0], current[1], current[2], current[3],
            np.arctan2(current[4], current[5]), current[6], current[7],
        ], dtype=np.float64)
    return state_features(current)


def actual_rollout(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    action: np.ndarray,
    horizon: int,
) -> np.ndarray:
    env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
    observation = None
    for _ in range(horizon):
        observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        if terminated or truncated:
            break
    if observation is None:
        raise RuntimeError("empty rollout")
    return np.asarray(observation["state"], dtype=np.float64).copy()


def bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(len(values)),
    }


def continuous_covariates(state: np.ndarray, action: np.ndarray) -> dict[str, float]:
    relative = np.asarray(state[2:4] - state[:2], dtype=np.float64)
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]])
    return {
        "contact_distance": float(np.linalg.norm(relative)),
        "action_normal": float(np.dot(action, normal)),
        "action_tangent": float(np.dot(action, tangent)),
        "agent_speed": float(np.linalg.norm(state[5:7])),
        "block_angle": raw_block_angle(state),
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def evaluate(
    env: gym.Env,
    models: dict[str, ExtraTreesRegressor],
    cases: list[dict[str, Any]],
    horizons: list[int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    covariate_rows: list[dict[str, Any]] = []
    for model_name, model in models.items():
        for horizon in horizons:
            rows: list[dict[str, Any]] = []
            for case_index, case in enumerate(cases):
                state = case["state"]
                action = case["action"]
                previous = case["previous_action"]
                interventions = [action, -action, np.zeros(2, dtype=np.float32)]
                actual = np.asarray([
                    state_features(actual_rollout(env, state, case["goal"], intervention, horizon))
                    for intervention in interventions
                ])
                predicted = np.asarray([
                    rollout_model(model, state, intervention, previous, horizon, model_name)
                    for intervention in interventions
                ])
                block_errors = np.linalg.norm(predicted[:, 2:4] - actual[:, 2:4], axis=1)
                angle_errors = np.asarray([
                    angle_error(predicted[index], actual[index])
                    for index in range(len(interventions))
                ])
                true_separation = float(np.linalg.norm(actual[0, 2:4] - actual[1, 2:4]))
                predicted_separation = float(np.linalg.norm(predicted[0, 2:4] - predicted[1, 2:4]))
                covariates = continuous_covariates(state, action)
                row = {
                    "case": case_index,
                    "mode": case["mode"],
                    "state": state,
                    "action": action,
                    "previous_action": previous,
                    "actual": actual,
                    "predicted": predicted,
                    "block_xy_error": block_errors,
                    "angle_error": angle_errors,
                    "true_plus_minus_separation": true_separation,
                    "predicted_plus_minus_separation": predicted_separation,
                    "separation_abs_error": abs(predicted_separation - true_separation),
                    **covariates,
                }
                rows.append(row)
                if model_name == "state_action":
                    actual_rotation = np.mean([
                        angle_error(actual[index], state_features(state))
                        for index in (0, 1)
                    ])
                    covariate_rows.append({
                        "case": case_index,
                        "mode": case["mode"],
                        "horizon": horizon,
                        "state_action_error": float(np.mean(block_errors)),
                        "actual_rotation_effect": float(actual_rotation),
                        **covariates,
                    })
            errors = np.asarray([np.mean(row["block_xy_error"]) for row in rows])
            angles = np.asarray([np.mean(row["angle_error"]) for row in rows])
            separations = np.asarray([row["separation_abs_error"] for row in rows])
            reports.append({
                "model": model_name,
                "horizon": int(horizon),
                "n_states": len(rows),
                "metrics": {
                    "block_xy_error": bootstrap(errors, args.bootstrap, args.seed + horizon),
                    "angle_error": bootstrap(angles, args.bootstrap, args.seed + 100 + horizon),
                    "separation_abs_error": bootstrap(separations, args.bootstrap, args.seed + 200 + horizon),
                },
                "rows": rows,
            })
    return reports, covariate_rows


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    args.horizons = sorted(set(int(item) for item in args.horizons))
    if any(item < 1 for item in args.horizons):
        raise ValueError("horizons must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = make_state_only_env()
    try:
        train = collect_trajectory_data(env, args.train_transitions, args.seed)
        models = fit_models(train, args.trees, args.seed)
        cases: list[dict[str, Any]] = []
        for index in range(args.held_out):
            observation, initial_info = env.reset(seed=args.seed + 500000 + index)
            state = np.asarray(observation["state"], dtype=np.float64).copy()
            goal = np.asarray(initial_info["goal_state"], dtype=np.float64).copy()
            previous = np.zeros(2, dtype=np.float32)
            rng = np.random.default_rng(args.seed + 600000 + index)
            # Advance with a genuine behavior prefix so held-out states carry
            # nonzero velocity and a nontrivial previous-action context.
            for step in range(6 + index % 7):
                action = choose_behavior_action(state, step, rng)
                observation, _, terminated, truncated, _ = env.step(action)
                state = np.asarray(observation["state"], dtype=np.float64).copy()
                previous = action.copy()
                if terminated or truncated:
                    break
            probe = choose_behavior_action(state, 10 + index, rng)
            # Reset again with the actual state after the prefix while
            # retaining the independent goal sampled at episode reset.  The
            # goal is not used by the adapter, but a valid independent goal
            # prevents accidental early termination during evaluation.
            reset_observation, reset_info = env.reset(
                seed=args.seed + 700000 + index,
                options={"state": state, "goal_state": goal},
            )
            state = np.asarray(reset_observation["state"], dtype=np.float64).copy()
            cases.append({
                "state": state,
                "goal": np.asarray(reset_info["goal_state"], dtype=np.float64).copy(),
                "action": (args.action_magnitude * unit(probe)).astype(np.float32),
                "previous_action": previous,
                "mode": mode_for_state(state),
            })
        reports, covariate_rows = evaluate(env, models, cases, args.horizons, args)
    finally:
        env.close()

    associations = []
    for horizon in args.horizons:
        subset = [row for row in covariate_rows if row["horizon"] == horizon]
        if not subset:
            continue
        associations.append({
            "horizon": horizon,
            "n": len(subset),
            "pearson_with_state_action_error": {
                key: pearson(
                    np.asarray([row[key] for row in subset]),
                    np.asarray([row["state_action_error"] for row in subset]),
                )
                for key in (
                    "contact_distance", "action_normal", "action_tangent",
                    "agent_speed", "block_angle", "actual_rotation_effect",
                )
            },
            "by_mode": {
                mode: {
                    "n": sum(row["mode"] == mode for row in subset),
                    "mean_error": float(np.mean([
                        row["state_action_error"] for row in subset if row["mode"] == mode
                    ])) if any(row["mode"] == mode for row in subset) else float("nan"),
                }
                for mode in MODE_NAMES
            },
        })
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "training_protocol": "genuine trajectory states with changing normal/tangential actions",
            "held_out_protocol": "same held-out state receives +a/-a/zero; previous action is retained",
            "models": list(MODEL_NAMES),
            "horizons": args.horizons,
            "train_transitions": args.train_transitions,
            "held_out": args.held_out,
            "trees": args.trees,
            "action_magnitude": args.action_magnitude,
            "bootstrap_draws": args.bootstrap,
            "seed": args.seed,
            "wall_seconds": time.perf_counter() - started,
        },
        "training": {
            "n": int(len(train["states"])),
            "mode_counts": {mode: int(np.sum(train["modes"] == mode)) for mode in MODE_NAMES},
            "finite": bool(
                np.isfinite(train["states"]).all()
                and np.isfinite(train["actions"]).all()
                and np.isfinite(train["previous_actions"]).all()
                and np.isfinite(train["next_states"]).all()
            ),
        },
        "reports": reports,
        "continuous_associations": associations,
        "sanity_checks": {
            "held_out_count": len(cases),
            "all_finite": bool(all(
                np.isfinite(item["metrics"]["block_xy_error"]["estimate"])
                for item in reports
            )),
            "paired_interventions": True,
            "models": list(MODEL_NAMES),
        },
        "claim_boundary": (
            "Privileged simulator-state physical-adapter diagnosis. The comparison isolates object-centric geometry, "
            "velocity-bearing state, and previous-action history as predictive inputs; it does not establish frozen-LeWM "
            "latent causality, visual online planning, or general PushT task performance."
        ),
    }
    (output / "contact_history_velocity_diagnostic.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True)
    )
    lines = [
        "# PushT contact history and velocity diagnostic",
        "",
        "Errors are paired held-out means over +a/-a/zero interventions.",
        "",
        "| model | horizon | block XY error | angle error | separation error |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in reports:
        metrics = item["metrics"]
        lines.append(
            "| %s | %d | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] |" % (
                item["model"], item["horizon"],
                metrics["block_xy_error"]["estimate"], metrics["block_xy_error"]["ci95_low"], metrics["block_xy_error"]["ci95_high"],
                metrics["angle_error"]["estimate"], metrics["angle_error"]["ci95_low"], metrics["angle_error"]["ci95_high"],
                metrics["separation_abs_error"]["estimate"], metrics["separation_abs_error"]["ci95_low"], metrics["separation_abs_error"]["ci95_high"],
            )
        )
    lines += ["", "## Continuous associations", ""]
    for item in associations:
        lines.append("h=%d: %s" % (item["horizon"], ", ".join(
            "%s=%.3f" % (key, value)
            for key, value in item["pearson_with_state_action_error"].items()
        )))
    lines += ["", report["claim_boundary"], ""]
    (output / "contact_history_velocity_diagnostic.md").write_text("\n".join(lines))
    print(json.dumps({
        "output": str(output),
        "training": report["training"],
        "sanity_checks": report["sanity_checks"],
    }, indent=2))


if __name__ == "__main__":
    main()
