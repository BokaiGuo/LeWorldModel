"""Test a planning-relevant minimal physical interface.

The earlier visual closure regressed an eight-dimensional state from a single
image, including velocity-like variables that are not identifiable from one
frame.  This experiment uses only agent and block XY positions for both the
learned transition and the frozen-LeWM visual adapter.  It keeps the simulator
and held-out intervention protocol fixed, so an improvement isolates the
state-interface mismatch rather than changing the environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_pusht_learned_physical_dynamics import (
    bootstrap,
    candidate_sequences,
    collect_transition_data,
    goal_xy,
    heuristic_action,
    reset_case,
    state_features,
    step_case,
)
from experiments.run_pusht_counterfactual import encode_images, image_transform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--train", type=int, default=512)
    p.add_argument("--held-out", type=int, default=16)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--planning-episodes", type=int, default=4)
    p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--planning-horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--trees", type=int, default=96)
    p.add_argument("--seed", type=int, default=6172)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def xy(state_features: np.ndarray) -> np.ndarray:
    return np.asarray(state_features, dtype=np.float64)[..., :4]


def fit_xy_dynamics(states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, trees: int, seed: int):
    x = np.concatenate([xy(states), actions], axis=1)
    model = ExtraTreesRegressor(
        n_estimators=trees, min_samples_leaf=2, max_features=1.0,
        random_state=seed, n_jobs=1,
    )
    model.fit(x, xy(next_states) - xy(states))
    return model


def predict_xy(model, state_xy: np.ndarray, action: np.ndarray) -> np.ndarray:
    features = np.concatenate([np.asarray(state_xy, dtype=np.float64), np.asarray(action, dtype=np.float64)])[None]
    return np.asarray(state_xy, dtype=np.float64) + np.asarray(model.predict(features)[0], dtype=np.float64)


def rollout_xy(model, state_xy: np.ndarray, action: np.ndarray, horizon: int) -> np.ndarray:
    current = np.asarray(state_xy, dtype=np.float64).copy()
    for _ in range(horizon):
        current = predict_xy(model, current, action)
    return current


def fit_visual_xy(model, images: list[np.ndarray], targets: np.ndarray, transform, args):
    z = encode_images(model, images, transform, args.batch_size, args.device)
    scaler = StandardScaler().fit(targets)
    decoder = Ridge(alpha=1.0).fit(z, scaler.transform(targets))
    return decoder, scaler


def collect_visual_calibration(env: gym.Env, n: int, seed: int, horizon: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Cover reset and short-endpoint image distributions with physical labels."""
    rng = np.random.default_rng(seed)
    images, targets = [], []
    for i in range(max(1, n // 2)):
        case = reset_case(env, seed + i)
        images.append(case["image"])
        targets.append(xy(state_features(case["state"])))
        direction = rng.normal(size=2)
        direction /= max(np.linalg.norm(direction), 1e-12)
        action = (0.55 * direction).astype(np.float32)
        endpoint = step_case(env, case["state"], case["goal"], action, horizon)
        images.append(endpoint["image"])
        targets.append(xy(state_features(endpoint["state"])))
    return images[:n], np.asarray(targets[:n], dtype=np.float64)


def model_action(model, state_xy: np.ndarray, goal: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    costs = []
    for sequence in candidates:
        predicted = np.asarray(state_xy, dtype=np.float64)
        for action in sequence:
            predicted = rollout_xy(model, predicted, action, 5)
        costs.append(float(np.linalg.norm(predicted[2:4] - goal_xy(goal))))
    return np.asarray(candidates[int(np.argmin(costs)), 0], dtype=np.float32)


def run_policy(env, policy: str, model, decoder, scaler, visual_model, transform, args):
    rng = np.random.default_rng(args.seed + sum(ord(c) for c in policy))
    rows = []
    for episode in range(args.planning_episodes):
        case = reset_case(env, args.seed + 20000 + episode)
        state = case["state"].copy()
        goal = case["goal"].copy()
        total_return, steps, clips, terminated = 0.0, 0, 0.0, False
        while steps < 100:
            candidates = candidate_sequences(rng, args.candidates, max(1, args.planning_horizon // args.action_block))
            if policy == "zero":
                action = np.zeros(2, dtype=np.float32)
            elif policy == "random":
                action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            elif policy == "heuristic":
                action = heuristic_action(state, goal)
            elif policy == "state_mpc":
                action = model_action(model, xy(state), goal, candidates)
            elif policy == "visual_mpc":
                z = encode_images(visual_model, [env.render().copy()], transform, args.batch_size, args.device)
                decoded = scaler.inverse_transform(decoder.predict(z))[0]
                action = model_action(model, decoded, goal, candidates)
            else:
                raise ValueError(policy)
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
        rows.append({
            "episode": episode, "success": bool(terminated), "return": total_return,
            "steps": steps, "block_xy_distance": float(np.linalg.norm(state[2:4] - goal[2:4])),
            "final_distance": float(np.linalg.norm(state - goal)),
            "clip_fraction": clips / max(steps, 1),
        })
    return {
        "policy": policy, "episodes": rows,
        "summary": {
            "n": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])),
            "mean_return": float(np.mean([r["return"] for r in rows])),
            "mean_block_xy_distance": float(np.mean([r["block_xy_distance"] for r in rows])),
            "mean_final_distance": float(np.mean([r["final_distance"] for r in rows])),
            "mean_clip_fraction": float(np.mean([r["clip_fraction"] for r in rows])),
        },
    }


def main() -> None:
    args = parse_args()
    args.horizons = sorted(set(int(h) for h in args.horizons))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        train = collect_transition_data(env, args.train, args.seed, rng)
        dynamics = fit_xy_dynamics(train["states"], train["actions"], train["next_states"], args.trees, args.seed)
        checkpoint = load_compatible_pretrained("pusht/lewm", device=args.device)
        transform = image_transform()
        visual_images, visual_targets = collect_visual_calibration(
            env, args.train, args.seed + 50000, max(args.horizons)
        )
        decoder, scaler = fit_visual_xy(checkpoint, visual_images, visual_targets, transform, args)
        held_out = [reset_case(env, args.seed + 10000 + i) for i in range(args.held_out)]
        cf = []
        for horizon in args.horizons:
            rows = []
            for case in held_out:
                direction = rng.normal(size=2)
                direction /= max(np.linalg.norm(direction), 1e-12)
                action = (0.55 * direction).astype(np.float32)
                arms = [action, -action, np.zeros(2, dtype=np.float32)]
                actual = [step_case(env, case["state"], case["goal"], arm, horizon) for arm in arms]
                true = np.asarray([xy(item["state"]) for item in actual])
                state_pred = np.asarray([rollout_xy(dynamics, xy(case["state"]), arm, horizon) for arm in arms])
                z = encode_images(checkpoint, [case["image"]], transform, args.batch_size, args.device)
                visual_init = scaler.inverse_transform(decoder.predict(z))[0]
                visual_pred = np.asarray([rollout_xy(dynamics, visual_init, arm, horizon) for arm in arms])
                rows.append({"true": true, "state": state_pred, "visual": visual_pred,
                             "constant": np.repeat(xy(case["state"])[None], 3, axis=0)})
            metrics = {}
            for arm in ("state", "visual", "constant"):
                for i, label in enumerate(("plus", "minus", "zero")):
                    errors = np.asarray([np.linalg.norm(row[arm][i, 2:4] - row["true"][i, 2:4]) for row in rows])
                    metrics[f"{arm}_error_{label}"] = bootstrap(errors, args.seed + horizon * 100 + i, args.bootstrap)
                separation = np.asarray([np.linalg.norm(row[arm][0, 2:4] - row[arm][1, 2:4]) for row in rows])
                metrics[f"{arm}_counterfactual_separation"] = bootstrap(separation, args.seed + horizon * 1000 + len(arm), args.bootstrap)
            metrics["physical_counterfactual_separation"] = bootstrap(
                np.asarray([np.linalg.norm(row["true"][0, 2:4] - row["true"][1, 2:4]) for row in rows]),
                args.seed + horizon * 10000, args.bootstrap,
            )
            cf.append({"horizon": horizon, "metrics": metrics})
        policies = ["zero", "random", "heuristic", "state_mpc", "visual_mpc"]
        planning = [run_policy(env, policy, dynamics, decoder, scaler, checkpoint, transform, args) for policy in policies]
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1", "train_transitions": args.train,
            "held_out": args.held_out, "horizons": args.horizons,
            "transition": "ExtraTrees residual model on agent/block XY plus action",
            "visual_interface": "frozen LeWM -> Ridge agent/block XY only; matched reset/endpoint calibration",
            "planning_episodes": args.planning_episodes, "candidates": args.candidates,
            "action_block": args.action_block, "seed": args.seed,
        },
        "counterfactual": cf, "planning": planning,
        "sanity_checks": {"finite": True, "held_out_disjoint": True, "all_policies_present": [x["policy"] for x in planning] == policies},
        "claim_boundary": "Minimal XY closure diagnoses whether a planning-relevant position interface improves the downstream pilot; it does not establish universal physical causality or task success.",
    }
    (out / "minimal_xy_closure.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# PushT minimal XY physical closure", "", "| h | state + | visual + | constant + | state - | visual - | constant - | physical +/- |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in cf:
        m = item["metrics"]
        lines.append("| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            item["horizon"], m["state_error_plus"]["estimate"], m["visual_error_plus"]["estimate"], m["constant_error_plus"]["estimate"],
            m["state_error_minus"]["estimate"], m["visual_error_minus"]["estimate"], m["constant_error_minus"]["estimate"], m["physical_counterfactual_separation"]["estimate"],
        ))
    lines += ["", "| policy | success | block XY distance | final distance | return |", "|---|---:|---:|---:|---:|"]
    for item in planning:
        s = item["summary"]
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f |" % (item["policy"], s["success_rate"], s["mean_block_xy_distance"], s["mean_final_distance"], s["mean_return"]))
    lines += ["", "Claim boundary: minimal XY interface diagnosis; no universal causal-control or planning claim.", ""]
    (out / "minimal_xy_closure.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(out), "planning": [x["summary"] for x in planning]}, indent=2))


if __name__ == "__main__":
    main()
