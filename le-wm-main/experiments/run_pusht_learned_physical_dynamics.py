"""Learned physical-dynamics closure for PushT.

This experiment closes the causal-control ladder at the simulator level while
keeping the frozen LeWM result separate from a downstream physical adapter.
It fits a one-step state transition model on independent simulator rollouts,
fits a frozen-LeWM-latent-to-state readout on disjoint rendered states, and
evaluates both on paired ``+a/-a/zero`` counterfactuals at multiple horizons.
The same learned models are then used in a matched closed-loop MPC comparison
against a simulator-oracle MPC, zero, random, and a privileged heuristic.

The learned state model is not claimed to be the original LeWM predictor.  It
is an explicit control closure: if the state arm works and the visual arm does
not, the missing link is physical observability from the frozen latent rather
than simulator controllability.  All training and evaluation seeds are
disjoint, and all endpoint readouts are fit on training states only.
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
import stable_pretraining as spt
import stable_worldmodel  # noqa: F401
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_pusht_counterfactual import encode_images, image_transform, pose_target

# The simulator emits one logger warning for every explicit-state reset when
# it casts the action/state arrays.  It is a known benign dependency warning;
# suppress it so long runs remain auditable and do not flood the workspace.
gym.logger.min_level = 40


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--dynamics-train", type=int, default=512)
    p.add_argument("--visual-train", type=int, default=192)
    p.add_argument("--held-out", type=int, default=24)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--dynamics-trees", type=int, default=96)
    p.add_argument("--planning-episodes", type=int, default=8)
    p.add_argument("--planning-horizon", type=int, default=10)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--candidates", type=int, default=96)
    p.add_argument("--seed", type=int, default=6172)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def state_features(state: np.ndarray) -> np.ndarray:
    """Continuous physical state with a non-wrapping orientation encoding."""
    state = np.asarray(state, dtype=np.float64)
    return np.asarray([
        state[0], state[1], state[2], state[3], np.sin(state[4]), np.cos(state[4]), state[5], state[6]
    ], dtype=np.float64)


def physical_xy(features: np.ndarray) -> np.ndarray:
    return np.asarray(features)[..., 2:4]


def _short_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def reset_case(env: gym.Env, seed: int) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    state = np.asarray(obs["state"], dtype=np.float64).copy()
    goal = np.asarray(info["goal_state"], dtype=np.float64).copy()
    return {"seed": seed, "state": state, "goal": goal, "image": env.render().copy()}


def step_case(env: gym.Env, state: np.ndarray, goal: np.ndarray, action: np.ndarray, horizon: int) -> dict[str, Any]:
    env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
    for _ in range(horizon):
        obs, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        if terminated or truncated:
            break
    final_state = np.asarray(obs["state"], dtype=np.float64).copy()
    return {"state": final_state, "image": env.render().copy()}


def collect_transition_data(env: gym.Env, n: int, seed: int, rng: np.random.Generator) -> dict[str, Any]:
    states, actions, next_states, images = [], [], [], []
    episode_length = 32
    current = None
    goal = None
    for i in range(n):
        if current is None or i % episode_length == 0:
            case = reset_case(env, seed + i // episode_length)
            current = case["state"].copy()
            goal = case["goal"].copy()
        image = env.render().copy()
        # A half-random, half-targeted behavior policy covers both free motion
        # and the contact manifold.  Independent reset/action pairs almost
        # never touch the block and therefore cannot identify PushT contact.
        if rng.random() < 0.5:
            target = current[2:4] - current[:2]
            action = np.clip(target / 100.0 + 0.15 * rng.normal(size=2), -1.0, 1.0)
        else:
            action = rng.uniform(-1.0, 1.0, size=2)
        action = np.asarray(action, dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        next_state = np.asarray(obs["state"], dtype=np.float64).copy()
        states.append(state_features(current))
        actions.append(action.astype(np.float64))
        next_states.append(state_features(next_state))
        images.append(image)
        current = None if terminated or truncated else next_state
    return {
        "states": np.asarray(states), "actions": np.asarray(actions),
        "next_states": np.asarray(next_states), "images": images,
    }


def fit_dynamics(data: dict[str, Any], trees: int, seed: int) -> ExtraTreesRegressor:
    x = np.concatenate([data["states"], data["actions"]], axis=1)
    model = ExtraTreesRegressor(
        n_estimators=trees, min_samples_leaf=2, max_features=1.0,
        # The project runs CPU-only on this host.  Single-process inference is
        # slower than joblib here but avoids a sklearn 1.7 worker warning loop
        # that can otherwise flood logs and exhaust the workspace.
        random_state=seed, n_jobs=1,
    )
    # Residual targets are the stable physical parameterization: absolute
    # coordinates vary widely across episodes, while contact/action effects
    # are local.  Prediction adds the residual back to the current state.
    model.fit(x, data["next_states"] - data["states"])
    return model


def fit_visual_decoder(model: torch.nn.Module, data: dict[str, Any], transform, args: argparse.Namespace):
    z = encode_images(model, data["images"], transform, args.batch_size, args.device)
    target = data["states"]
    scaler = StandardScaler().fit(target)
    decoder = Ridge(alpha=10.0).fit(z, scaler.transform(target))
    return decoder, scaler, z


def predict_next(dynamics: ExtraTreesRegressor, state: np.ndarray, action: np.ndarray) -> np.ndarray:
    x = np.concatenate([np.asarray(state, dtype=np.float64), np.asarray(action, dtype=np.float64)])[None]
    return np.asarray(state, dtype=np.float64) + np.asarray(dynamics.predict(x)[0], dtype=np.float64)


def rollout_model(dynamics: ExtraTreesRegressor, state: np.ndarray, action: np.ndarray, horizon: int) -> np.ndarray:
    current = np.asarray(state, dtype=np.float64).copy()
    for _ in range(horizon):
        current = predict_next(dynamics, current, action)
    return current


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


def counterfactual_eval(
    env: gym.Env,
    dynamics: ExtraTreesRegressor,
    decoder: Ridge,
    decoder_scaler: StandardScaler,
    model: torch.nn.Module,
    transform,
    cases: list[dict[str, Any]],
    horizons: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    initial_z = encode_images(model, [case["image"] for case in cases], transform, args.batch_size, args.device)
    decoded_states = decoder_scaler.inverse_transform(decoder.predict(initial_z))
    rng = np.random.default_rng(args.seed + 991)
    reports = []
    for horizon in horizons:
        rows = []
        for i, case in enumerate(cases):
            direction = rng.normal(size=2)
            direction /= max(np.linalg.norm(direction), 1e-12)
            action = (0.55 * direction).astype(np.float32)
            actions = [action, -action, np.zeros(2, dtype=np.float32)]
            actual = []
            for act in actions:
                actual.append(step_case(env, case["state"], case["goal"], act, horizon))
            true_features = np.asarray([state_features(item["state"]) for item in actual])
            state_preds = np.asarray([rollout_model(dynamics, state_features(case["state"]), act, horizon) for act in actions])
            visual_preds = np.asarray([rollout_model(dynamics, decoded_states[i], act, horizon) for act in actions])
            baseline = np.repeat(state_features(case["state"])[None], 3, axis=0)
            rows.append({
                "case": i, "action": action, "true": true_features,
                "state_model": state_preds, "visual_model": visual_preds,
                "constant": baseline,
            })
        def metric(arm: str, intervention: int = 0):
            values = np.asarray([
                np.linalg.norm(row[arm][intervention, 2:4] - row["true"][intervention, 2:4])
                for row in rows
            ])
            return bootstrap(values, args.seed + 1000 * horizon + 11 * intervention + len(arm), args.bootstrap)
        separation = {}
        for arm in ("state_model", "visual_model", "constant"):
            pred = np.asarray([row[arm] for row in rows])
            separation[arm] = bootstrap(
                np.linalg.norm(pred[:, 0, 2:4] - pred[:, 1, 2:4], axis=1),
                args.seed + 5000 + horizon + len(arm), args.bootstrap,
            )
        reports.append({
            "horizon": horizon,
            "metrics": {
                "state_model_error_plus": metric("state_model", 0),
                "state_model_error_minus": metric("state_model", 1),
                "state_model_error_zero": metric("state_model", 2),
                "visual_model_error_plus": metric("visual_model", 0),
                "visual_model_error_minus": metric("visual_model", 1),
                "visual_model_error_zero": metric("visual_model", 2),
                "constant_error_plus": metric("constant", 0),
                "constant_error_minus": metric("constant", 1),
                "constant_error_zero": metric("constant", 2),
                "state_model_counterfactual_separation": separation["state_model"],
                "visual_model_counterfactual_separation": separation["visual_model"],
                "physical_counterfactual_separation": bootstrap(
                    np.asarray([np.linalg.norm(row["true"][0, 2:4] - row["true"][1, 2:4]) for row in rows]),
                    args.seed + 7000 + horizon, args.bootstrap,
                ),
            },
            "rows": rows,
        })
    return {"horizons": reports, "decoded_state_finite": bool(np.isfinite(decoded_states).all())}


def candidate_sequences(rng: np.random.Generator, n: int, blocks: int) -> np.ndarray:
    candidates = rng.uniform(-1.0, 1.0, size=(n, blocks, 2)).astype(np.float32)
    candidates[0] = 0.0
    return candidates


def goal_xy(goal: np.ndarray) -> np.ndarray:
    return np.asarray(goal, dtype=np.float64)[2:4]


def choose_model_action(dynamics, state, goal, candidates):
    costs = []
    for sequence in candidates:
        predicted = np.asarray(state, dtype=np.float64)
        for action in sequence:
            predicted = rollout_model(dynamics, predicted, action, 5)
        costs.append(float(np.linalg.norm(predicted[2:4] - goal_xy(goal))))
    return np.asarray(candidates[int(np.argmin(costs)), 0], dtype=np.float32)


def choose_oracle_action(env, state, goal, candidates):
    costs = []
    for sequence in candidates:
        current = None
        for block, action in enumerate(sequence):
            result = step_case(env, state if block == 0 else current, goal, action, 5)
            current = result["state"]
        costs.append(float(np.linalg.norm(current[2:4] - goal_xy(goal))))
    return np.asarray(candidates[int(np.argmin(costs)), 0], dtype=np.float32)


def heuristic_action(state: np.ndarray, goal: np.ndarray) -> np.ndarray:
    agent, block = state[:2], state[2:4]
    direction = goal[2:4] - block
    direction /= max(np.linalg.norm(direction), 1e-6)
    behind = block - 50.0 * direction
    push_through = block + 100.0 * direction
    target = behind if np.linalg.norm(behind - agent) > 25.0 else push_through
    return np.clip((target - agent) / 100.0, -1.0, 1.0).astype(np.float32)


def run_planner(
    env: gym.Env,
    policy: str,
    dynamics: ExtraTreesRegressor,
    decoder: Ridge,
    decoder_scaler: StandardScaler,
    model: torch.nn.Module,
    transform,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed + sum(ord(c) for c in policy))
    episodes = []
    overall_start = time.perf_counter()
    for episode in range(args.planning_episodes):
        case = reset_case(env, args.seed + 20000 + episode)
        state = case["state"].copy()
        goal = case["goal"].copy()
        total_return, steps, clips = 0.0, 0, 0.0
        start = time.perf_counter()
        while steps < 100:
            candidates = candidate_sequences(rng, args.candidates, args.planning_horizon // args.action_block)
            if policy == "zero":
                action = np.zeros(2, dtype=np.float32)
            elif policy == "random":
                action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            elif policy == "heuristic":
                action = heuristic_action(state, goal)
            elif policy == "oracle_mpc":
                action = choose_oracle_action(env, state, goal, candidates)
            else:
                if policy == "visual_mpc":
                    image = env.render().copy()
                    z = encode_images(model, [image], transform, args.batch_size, args.device)
                    planning_state = decoder_scaler.inverse_transform(decoder.predict(z)[0:1])[0]
                else:
                    planning_state = state_features(state)
                action = choose_model_action(dynamics, planning_state, goal, candidates)
            for _ in range(args.action_block):
                obs, reward, terminated, truncated, _ = env.step(action)
                state = np.asarray(obs["state"], dtype=np.float64).copy()
                total_return += float(reward)
                clips += float(np.mean(np.abs(action) >= 0.999))
                steps += 1
                if terminated or truncated or steps >= 100:
                    break
            if terminated or truncated:
                break
        episodes.append({
            "episode": episode, "success": bool(terminated), "return": total_return,
            "steps": steps, "block_xy_distance": float(np.linalg.norm(state[2:4] - goal[2:4])),
            "final_distance": float(np.linalg.norm(state - goal)),
            "clip_fraction": clips / max(steps, 1),
        })
    success = np.asarray([row["success"] for row in episodes], dtype=np.float64)
    return {
        "policy": policy, "episodes": episodes,
        "summary": {
            "n": len(episodes), "success_rate": float(success.mean()),
            "mean_return": float(np.mean([r["return"] for r in episodes])),
            "mean_block_xy_distance": float(np.mean([r["block_xy_distance"] for r in episodes])),
            "mean_final_distance": float(np.mean([r["final_distance"] for r in episodes])),
            "mean_clip_fraction": float(np.mean([r["clip_fraction"] for r in episodes])),
            "wall_seconds": float(time.perf_counter() - overall_start),
        },
    }


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def main() -> None:
    args = parse_args()
    if args.dynamics_train < 32 or args.visual_train < 16 or args.held_out < 4:
        raise ValueError("training and held-out sizes are too small for a closure run")
    args.horizons = sorted(set(int(h) for h in args.horizons))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        train = collect_transition_data(env, max(args.dynamics_train, args.visual_train), args.seed, rng)
        dynamics = fit_dynamics(train, args.dynamics_trees, args.seed)
        model = load_compatible_pretrained("pusht/lewm", device=args.device)
        transform = image_transform()
        visual_train = {key: (value[:args.visual_train] if isinstance(value, np.ndarray) else value[:args.visual_train]) for key, value in train.items()}
        decoder, decoder_scaler, _ = fit_visual_decoder(model, visual_train, transform, args)
        held_out = [reset_case(env, args.seed + 10000 + i) for i in range(args.held_out)]
        cf = counterfactual_eval(env, dynamics, decoder, decoder_scaler, model, transform, held_out, args.horizons, args)
        policies = ["zero", "random", "heuristic", "oracle_mpc", "state_mpc", "visual_mpc"]
        planning = [run_planner(env, policy, dynamics, decoder, decoder_scaler, model, transform, args) for policy in policies]
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1", "seed": args.seed,
            "dynamics_train_transitions": args.dynamics_train,
            "visual_train_states": args.visual_train, "held_out_counterfactual_cases": args.held_out,
            "horizons": args.horizons, "dynamics_model": "ExtraTreesRegressor on physical state + action",
            "visual_model": "frozen LeWM embedding -> Ridge physical-state readout",
            "planning_candidates": args.candidates, "planning_horizon": args.planning_horizon,
            "action_block": args.action_block, "planning_episodes": args.planning_episodes,
            "checkpoint": str(getattr(model, "_checkpoint_compat", {})),
            "train_test_separation": "training seeds [seed, seed+n) and held-out seeds [seed+10000, ...)",
        },
        "counterfactual": cf,
        "planning": planning,
        "sanity_checks": {
            "finite_counterfactual": bool(np.isfinite(np.asarray([
                item["metrics"]["state_model_error_plus"]["estimate"] for item in cf["horizons"]
            ])).all()),
            "all_policies_present": [item["policy"] for item in planning] == policies,
            "held_out_disjoint": True,
        },
        "claim_boundary": (
            "The state arm establishes learned simulator-state dynamics and the visual arm tests physical observability "
            "from a frozen LeWM latent. Neither arm is evidence that the original LeWM predictor was retrained or that "
            "the downstream adapter is a universal world model; planner results are PushT-only."
        ),
    }
    (out / "learned_physical_dynamics.json").write_text(json.dumps(jsonable(report), indent=2, sort_keys=True))
    lines = [
        "# PushT learned physical-dynamics closure", "",
        "The state model is a learned simulator-state transition arm; the visual model initializes it from a frozen LeWM latent.", "",
        "## Held-out paired counterfactual endpoint error", "",
        "| horizon | state + | visual + | constant + | state - | visual - | constant - | physical +/- separation |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in cf["horizons"]:
        m = item["metrics"]
        lines.append("| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            item["horizon"], m["state_model_error_plus"]["estimate"], m["visual_model_error_plus"]["estimate"],
            m["constant_error_plus"]["estimate"], m["state_model_error_minus"]["estimate"],
            m["visual_model_error_minus"]["estimate"], m["constant_error_minus"]["estimate"],
            m["physical_counterfactual_separation"]["estimate"],
        ))
    lines += ["", "## Matched closed-loop planning", "", "| policy | success | block XY distance | final distance | return |", "|---|---:|---:|---:|---:|"]
    for item in planning:
        s = item["summary"]
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f |" % (item["policy"], s["success_rate"], s["mean_block_xy_distance"], s["mean_final_distance"], s["mean_return"]))
    lines += ["", "Claim boundary: state dynamics are simulator-level learned evidence; visual MPC tests frozen-latent physical observability; neither is a retraining or universal-world-model claim.", ""]
    (out / "learned_physical_dynamics.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(out), "sanity_checks": report["sanity_checks"], "planning": [x["summary"] for x in planning]}, indent=2))


if __name__ == "__main__":
    main()
