"""Contact-stratified paired action-effect and physical-adapter audit.

The previous closure experiments mixed free-space, approach, normal-contact,
and tangential-contact transitions in aggregate.  This experiment constructs
the four states explicitly, applies equal-norm ``+a/-a/zero`` interventions
from the *same* state, and evaluates held-out autoregressive physical models
separately by contact mode.  It is a privileged simulator-state diagnostic;
the learned models here are physical adapters, not frozen-LeWM predictors.
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

gym.logger.min_level = 40


STRATA = ("free", "approach", "normal_contact", "tangential_contact")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--train-per-stratum", type=int, default=256)
    p.add_argument("--held-out-per-stratum", type=int, default=16)
    p.add_argument("--trees", type=int, default=96)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 8])
    p.add_argument("--action-magnitude", type=float, default=0.55)
    p.add_argument("--bootstrap", type=int, default=4000)
    p.add_argument("--seed", type=int, default=20261006)
    return p.parse_args()


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def rotate(vector: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


def normalize_features(features: np.ndarray) -> np.ndarray:
    result = np.asarray(features, dtype=np.float64).copy()
    norm = np.linalg.norm(result[..., 4:6], axis=-1, keepdims=True)
    result[..., 4:6] /= np.maximum(norm, 1e-8)
    result[..., :4] = np.clip(result[..., :4], 5.0, 507.0)
    result[..., 6:] = np.clip(result[..., 6:], -500.0, 500.0)
    return result


def geometry_features(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    relative = state[2:4] - state[:2]
    distance = float(np.linalg.norm(relative))
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]])
    return np.concatenate([
        state,
        action,
        relative,
        [distance, float(np.dot(action, normal)), float(np.dot(action, tangent))],
    ])


def build_state(
    env: gym.Env,
    seed: int,
    stratum: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    observation, info = env.reset(seed=seed)
    state = np.asarray(observation["state"], dtype=np.float64).copy()
    goal = np.asarray(info["goal_state"], dtype=np.float64).copy()
    block = state[2:4].copy()
    direction = unit(rng.normal(size=2))
    if stratum == "free":
        distance = rng.uniform(105.0, 155.0)
    elif stratum == "approach":
        distance = rng.uniform(55.0, 78.0)
    else:
        distance = rng.uniform(16.0, 28.0)
    state[:2] = np.clip(block - distance * direction, 8.0, 504.0)
    state[-2:] = 0.0
    reset_observation, _ = env.reset(seed=seed, options={"state": state, "goal_state": goal})
    return np.asarray(reset_observation["state"], dtype=np.float64).copy(), goal


def intervention_action(state: np.ndarray, stratum: str, magnitude: float, rng: np.random.Generator) -> np.ndarray:
    relative = unit(np.asarray(state[2:4]) - np.asarray(state[:2]))
    if stratum == "tangential_contact":
        direction = rotate(relative, np.pi / 2.0)
    elif stratum in ("approach", "normal_contact"):
        direction = relative
    else:
        direction = unit(rng.normal(size=2))
    return (magnitude * direction).astype(np.float32)


def one_rollout(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    action: np.ndarray,
    horizon: int,
) -> np.ndarray:
    env.reset(seed=0, options={"state": np.asarray(state).copy(), "goal_state": np.asarray(goal).copy()})
    observation = None
    for _ in range(horizon):
        observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        if terminated or truncated:
            break
    if observation is None:
        raise RuntimeError("empty rollout")
    return np.asarray(observation["state"], dtype=np.float64).copy()


def collect_training_data(
    env: gym.Env,
    per_stratum: int,
    seed: int,
    magnitude: float,
) -> dict[str, np.ndarray]:
    states, actions, next_states = [], [], []
    for stratum_index, stratum in enumerate(STRATA):
        for index in range(per_stratum):
            state, goal = build_state(env, seed + 100000 * stratum_index + index, stratum, np.random.default_rng(seed + index + 17 * stratum_index))
            rng = np.random.default_rng(seed + 700000 + 31 * index + stratum_index)
            base = intervention_action(state, stratum, magnitude, rng)
            # Use all three signs in training so the model cannot use sign as a
            # shortcut for a single contact mode.
            action = (base if index % 3 == 0 else -base if index % 3 == 1 else np.zeros(2)).astype(np.float32)
            next_state = one_rollout(env, state, goal, action, 1)
            states.append(state_features(state))
            actions.append(action.astype(np.float64))
            next_states.append(state_features(next_state))
    return {
        "states": np.asarray(states, dtype=np.float64),
        "actions": np.asarray(actions, dtype=np.float64),
        "next_states": np.asarray(next_states, dtype=np.float64),
    }


def fit_model(data: dict[str, np.ndarray], trees: int, seed: int, use_geometry: bool) -> ExtraTreesRegressor:
    state_rows = data["states"]
    if use_geometry:
        x = np.asarray([
            geometry_features(state, action)
            for state, action in zip(state_rows, data["actions"])
        ])
    else:
        x = np.concatenate([state_rows, data["actions"]], axis=1)
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed + int(use_geometry),
        n_jobs=1,
    )
    model.fit(x, data["next_states"] - data["states"])
    return model


def predict_next(model: ExtraTreesRegressor, state: np.ndarray, action: np.ndarray, use_geometry: bool) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if use_geometry:
        features = geometry_features(state, action)[None]
    else:
        features = np.concatenate([state, action])[None]
    delta = model.predict(features)[0]
    return normalize_features(state + delta)


def rollout_model(model: ExtraTreesRegressor, state: np.ndarray, action: np.ndarray, horizon: int, use_geometry: bool) -> np.ndarray:
    current = state_features(state)
    for _ in range(horizon):
        current = predict_next(model, current, action, use_geometry)
    return current


def raw_angle(features: np.ndarray) -> float:
    return float(np.arctan2(features[4], features[5]))


def angle_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(abs((raw_angle(predicted) - raw_angle(actual) + np.pi) % (2.0 * np.pi) - np.pi))


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


def eval_stratum(
    env: gym.Env,
    model: ExtraTreesRegressor,
    cases: list[dict[str, Any]],
    horizons: list[int],
    use_geometry: bool,
    args: argparse.Namespace,
    model_name: str,
) -> list[dict[str, Any]]:
    reports = []
    for horizon in horizons:
        rows = []
        for case_index, case in enumerate(cases):
            state, goal, action = case["state"], case["goal"], case["action"]
            actions = [action, -action, np.zeros(2, dtype=np.float32)]
            actual = np.asarray([state_features(one_rollout(env, state, goal, act, horizon)) for act in actions])
            predicted = np.asarray([rollout_model(model, state, act, horizon, use_geometry) for act in actions])
            constant = np.repeat(state_features(state)[None], 3, axis=0)
            errors = np.asarray([np.linalg.norm(predicted[i, 2:4] - actual[i, 2:4]) for i in range(3)])
            constant_errors = np.asarray([np.linalg.norm(constant[i, 2:4] - actual[i, 2:4]) for i in range(3)])
            angle_errors = np.asarray([angle_error(predicted[i], actual[i]) for i in range(3)])
            true_sep = float(np.linalg.norm(actual[0, 2:4] - actual[1, 2:4]))
            predicted_sep = float(np.linalg.norm(predicted[0, 2:4] - predicted[1, 2:4]))
            rows.append({
                "case": case_index,
                "stratum": case["stratum"],
                "action": action,
                "actual": actual,
                "predicted": predicted,
                "block_xy_error": errors,
                "constant_block_xy_error": constant_errors,
                "angle_error": angle_errors,
                "true_plus_minus_separation": true_sep,
                "predicted_plus_minus_separation": predicted_sep,
                "separation_abs_error": abs(predicted_sep - true_sep),
            })
        error_values = np.concatenate([row["block_xy_error"] for row in rows])
        constant_values = np.concatenate([row["constant_block_xy_error"] for row in rows])
        angle_values = np.concatenate([row["angle_error"] for row in rows])
        sep_values = np.asarray([row["separation_abs_error"] for row in rows], dtype=np.float64)
        reports.append({
            "model": model_name,
            "horizon": horizon,
            "stratum": cases[0]["stratum"],
            "n_states": len(rows),
            "metrics": {
                "block_xy_error": bootstrap(error_values, args.bootstrap, args.seed + horizon + len(model_name)),
                "constant_block_xy_error": bootstrap(constant_values, args.bootstrap, args.seed + 101 + horizon),
                "angle_error": bootstrap(angle_values, args.bootstrap, args.seed + 202 + horizon),
                "plus_minus_separation_abs_error": bootstrap(sep_values, args.bootstrap, args.seed + 303 + horizon),
                "true_plus_minus_separation": bootstrap(np.asarray([row["true_plus_minus_separation"] for row in rows]), args.bootstrap, args.seed + 404 + horizon),
                "predicted_plus_minus_separation": bootstrap(np.asarray([row["predicted_plus_minus_separation"] for row in rows]), args.bootstrap, args.seed + 505 + horizon),
            },
            "rows": rows,
        })
    return reports


def main() -> None:
    args = parse_args()
    args.horizons = sorted(set(int(horizon) for horizon in args.horizons))
    if any(horizon < 1 for horizon in args.horizons):
        raise ValueError("horizons must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        train = collect_training_data(env, args.train_per_stratum, args.seed, args.action_magnitude)
        models = {
            "state_action": fit_model(train, args.trees, args.seed, False),
            "contact_geometry": fit_model(train, args.trees, args.seed, True),
        }
        specialist_models = {}
        train_strata = np.asarray([
            STRATA[index // args.train_per_stratum]
            for index in range(len(train["states"]))
        ])
        for stratum_index, stratum in enumerate(STRATA):
            mask = train_strata == stratum
            specialist_models[stratum] = fit_model(
                {key: value[mask] for key, value in train.items()},
                args.trees,
                args.seed + 100 + stratum_index,
                True,
            )
        held_out = {stratum: [] for stratum in STRATA}
        for stratum_index, stratum in enumerate(STRATA):
            for index in range(args.held_out_per_stratum):
                state, goal = build_state(env, args.seed + 500000 + 10000 * stratum_index + index, stratum, np.random.default_rng(args.seed + 600000 + index + stratum_index))
                action = intervention_action(state, stratum, args.action_magnitude, np.random.default_rng(args.seed + 700000 + index))
                held_out[stratum].append({"state": state, "goal": goal, "action": action, "stratum": stratum})
        reports = []
        for stratum in STRATA:
            for model_name, model in models.items():
                reports.extend(eval_stratum(env, model, held_out[stratum], args.horizons, model_name == "contact_geometry", args, model_name))
            reports.extend(eval_stratum(
                env,
                specialist_models[stratum],
                held_out[stratum],
                args.horizons,
                True,
                args,
                "specialist_geometry",
            ))
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "strata": list(STRATA),
            "state_construction": "same simulator state within each +a/-a/zero triplet; explicit agent-block distance and normal/tangential action mode",
            "train_per_stratum": args.train_per_stratum,
            "held_out_per_stratum": args.held_out_per_stratum,
            "horizons": args.horizons,
            "action_magnitude": args.action_magnitude,
            "bootstrap_draws": args.bootstrap,
            "seed": args.seed,
            "wall_seconds": time.perf_counter() - started,
        },
        "training": {
            "n": int(len(train["states"])),
            "stratum_counts": {stratum: args.train_per_stratum for stratum in STRATA},
            "finite": bool(np.isfinite(train["states"]).all() and np.isfinite(train["actions"]).all() and np.isfinite(train["next_states"]).all()),
        },
        "reports": reports,
        "sanity_checks": {
            "held_out_stratum_counts": {stratum: len(held_out[stratum]) for stratum in STRATA},
            "all_finite": bool(all(np.isfinite(item["metrics"]["block_xy_error"]["estimate"]) for item in reports)),
            "paired_triplets": True,
            "models": ["state_action", "contact_geometry", "specialist_geometry"],
        },
        "claim_boundary": (
            "Privileged simulator-state physical-adapter diagnostic. The stratified paired protocol identifies where a "
            "state/action model loses action-effect information, but it does not establish frozen-LeWM latent causality, "
            "visual online planning, or general PushT task performance."
        ),
    }
    (output / "contact_stratified_paired_intervention.json").write_text(json.dumps(jsonable(report), indent=2, sort_keys=True))
    lines = [
        "# PushT contact-stratified paired intervention",
        "",
        "Each held-out state receives equal-norm +a/-a/zero interventions. Intervals are paired bootstrap CIs.",
        "",
        "| model | stratum | horizon | block XY error | constant error | +a/-a separation error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in reports:
        metrics = item["metrics"]
        lines.append(
            "| %s | %s | %d | %.3f [%.3f, %.3f] | %.3f | %.3f |" % (
                item["model"], item["stratum"], item["horizon"],
                metrics["block_xy_error"]["estimate"], metrics["block_xy_error"]["ci95_low"], metrics["block_xy_error"]["ci95_high"],
                metrics["constant_block_xy_error"]["estimate"], metrics["plus_minus_separation_abs_error"]["estimate"],
            )
        )
    lines += ["", report["claim_boundary"], ""]
    (output / "contact_stratified_paired_intervention.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "training": report["training"], "sanity_checks": report["sanity_checks"]}, indent=2))


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


if __name__ == "__main__":
    main()
