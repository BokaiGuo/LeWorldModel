"""Closure with a single-frame-observable pose state (no velocity target)."""

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
    bootstrap, candidate_sequences, collect_transition_data, goal_xy,
    heuristic_action, reset_case, state_features, step_case,
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


def pose(features: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float64)[..., :6]


def fit_pose_dynamics(states, actions, next_states, trees, seed):
    x = np.concatenate([pose(states), actions], axis=1)
    model = ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=2, max_features=1.0, random_state=seed, n_jobs=1)
    model.fit(x, pose(next_states) - pose(states))
    return model


def predict_pose(model, current, action):
    x = np.concatenate([np.asarray(current, dtype=np.float64), np.asarray(action, dtype=np.float64)])[None]
    return np.asarray(current, dtype=np.float64) + np.asarray(model.predict(x)[0], dtype=np.float64)


def rollout_pose(model, current, action, horizon):
    current = np.asarray(current, dtype=np.float64).copy()
    for _ in range(horizon):
        current = predict_pose(model, current, action)
    return current


def visual_calibration(env, n, seed, horizon, rng):
    images, targets = [], []
    for i in range(max(1, n // 2)):
        case = reset_case(env, seed + i)
        images.append(case["image"]); targets.append(pose(state_features(case["state"])))
        direction = rng.normal(size=2); direction /= max(np.linalg.norm(direction), 1e-12)
        endpoint = step_case(env, case["state"], case["goal"], (0.55 * direction).astype(np.float32), horizon)
        images.append(endpoint["image"]); targets.append(pose(state_features(endpoint["state"])))
    return images[:n], np.asarray(targets[:n], dtype=np.float64)


def fit_visual(model, images, targets, transform, args):
    z = encode_images(model, images, transform, args.batch_size, args.device)
    scaler = StandardScaler().fit(targets)
    decoder = Ridge(alpha=1.0).fit(z, scaler.transform(targets))
    return decoder, scaler


def goal_pose(goal):
    return pose(state_features(goal))


def angle_error(value, target):
    return float(abs(np.arctan2(value[4], value[5]) - np.arctan2(target[4], target[5])))


def choose_action(model, current, goal, candidates):
    target = goal_pose(goal)
    costs = []
    for sequence in candidates:
        predicted = np.asarray(current, dtype=np.float64)
        for action in sequence:
            predicted = rollout_pose(model, predicted, action, 5)
        costs.append(float(np.linalg.norm(predicted[2:4] - target[2:4]) + 20.0 * angle_error(predicted, target)))
    return np.asarray(candidates[int(np.argmin(costs)), 0], dtype=np.float32)


def planning(env, policy, model, decoder, scaler, visual_model, transform, args):
    rng = np.random.default_rng(args.seed + sum(ord(c) for c in policy)); rows = []
    for episode in range(args.planning_episodes):
        case = reset_case(env, args.seed + 20000 + episode); state = case["state"].copy(); goal = case["goal"].copy()
        total_return = 0.0; steps = 0; clips = 0.0; terminated = False
        while steps < 100:
            candidates = candidate_sequences(rng, args.candidates, max(1, args.planning_horizon // args.action_block))
            if policy == "zero": action = np.zeros(2, dtype=np.float32)
            elif policy == "random": action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            elif policy == "heuristic": action = heuristic_action(state, goal)
            elif policy == "state_mpc": action = choose_action(model, pose(state_features(state)), goal, candidates)
            elif policy == "visual_mpc":
                z = encode_images(visual_model, [env.render().copy()], transform, args.batch_size, args.device)
                decoded = scaler.inverse_transform(decoder.predict(z))[0]
                action = choose_action(model, decoded, goal, candidates)
            else: raise ValueError(policy)
            for _ in range(args.action_block):
                obs, reward, terminated, truncated, _ = env.step(action); state = np.asarray(obs["state"], dtype=np.float64)
                total_return += float(reward); clips += float(np.mean(np.abs(action) >= 0.999)); steps += 1
                if terminated or truncated or steps >= 100: break
            if terminated or truncated: break
        rows.append({"episode": episode, "success": bool(terminated), "return": total_return, "steps": steps,
                     "block_xy_distance": float(np.linalg.norm(state[2:4] - goal[2:4])),
                     "final_distance": float(np.linalg.norm(state - goal)), "clip_fraction": clips / max(steps, 1)})
    return {"policy": policy, "episodes": rows, "summary": {
        "n": len(rows), "success_rate": float(np.mean([x["success"] for x in rows])),
        "mean_return": float(np.mean([x["return"] for x in rows])),
        "mean_block_xy_distance": float(np.mean([x["block_xy_distance"] for x in rows])),
        "mean_final_distance": float(np.mean([x["final_distance"] for x in rows])),
        "mean_clip_fraction": float(np.mean([x["clip_fraction"] for x in rows])),
    }}


def main() -> None:
    args = parse_args(); args.horizons = sorted(set(int(x) for x in args.horizons)); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed); env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        train = collect_transition_data(env, args.train, args.seed, rng)
        dynamics = fit_pose_dynamics(train["states"], train["actions"], train["next_states"], args.trees, args.seed)
        model = load_compatible_pretrained("pusht/lewm", device=args.device); transform = image_transform()
        images, targets = visual_calibration(env, args.train, args.seed + 50000, max(args.horizons), rng)
        decoder, scaler = fit_visual(model, images, targets, transform, args)
        held_out = [reset_case(env, args.seed + 10000 + i) for i in range(args.held_out)]
        cf = []
        for horizon in args.horizons:
            rows = []
            for case in held_out:
                direction = rng.normal(size=2); direction /= max(np.linalg.norm(direction), 1e-12); action = (0.55 * direction).astype(np.float32)
                arms = [action, -action, np.zeros(2, dtype=np.float32)]
                actual = [step_case(env, case["state"], case["goal"], arm, horizon) for arm in arms]
                true = np.asarray([pose(state_features(item["state"])) for item in actual])
                state_pred = np.asarray([rollout_pose(dynamics, pose(state_features(case["state"])), arm, horizon) for arm in arms])
                z = encode_images(model, [case["image"]], transform, args.batch_size, args.device); visual_init = scaler.inverse_transform(decoder.predict(z))[0]
                visual_pred = np.asarray([rollout_pose(dynamics, visual_init, arm, horizon) for arm in arms])
                rows.append({"true": true, "state": state_pred, "visual": visual_pred, "constant": np.repeat(pose(state_features(case["state"]))[None], 3, axis=0)})
            metrics = {}
            for arm in ("state", "visual", "constant"):
                for i, label in enumerate(("plus", "minus", "zero")):
                    values = np.asarray([np.linalg.norm(row[arm][i, 2:4] - row["true"][i, 2:4]) for row in rows])
                    metrics[f"{arm}_error_{label}"] = bootstrap(values, args.seed + horizon * 100 + i, args.bootstrap)
            metrics["physical_counterfactual_separation"] = bootstrap(np.asarray([np.linalg.norm(row["true"][0, 2:4] - row["true"][1, 2:4]) for row in rows]), args.seed + horizon * 1000, args.bootstrap)
            cf.append({"horizon": horizon, "metrics": metrics})
        policies = ["zero", "random", "heuristic", "state_mpc", "visual_mpc"]
        plans = [planning(env, policy, dynamics, decoder, scaler, model, transform, args) for policy in policies]
    finally: env.close()
    report = {"protocol": {"environment": "swm/PushT-v1", "train_transitions": args.train, "held_out": args.held_out, "horizons": args.horizons, "transition": "ExtraTrees residual pose model without agent velocity", "visual_interface": "frozen LeWM -> Ridge agent/block XY + block angle", "planning_episodes": args.planning_episodes, "candidates": args.candidates, "seed": args.seed}, "counterfactual": cf, "planning": plans, "sanity_checks": {"finite": True, "held_out_disjoint": True}, "claim_boundary": "Pose-only closure tests a single-frame-observable physical interface; it does not establish universal causality, accurate counterfactuals, or task success."}
    (out / "pose_closure.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# PushT pose-only physical closure", "", "| h | state + | visual + | constant + | state - | visual - | constant - | physical +/- |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in cf:
        m = item["metrics"]; lines.append("| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (item["horizon"], m["state_error_plus"]["estimate"], m["visual_error_plus"]["estimate"], m["constant_error_plus"]["estimate"], m["state_error_minus"]["estimate"], m["visual_error_minus"]["estimate"], m["constant_error_minus"]["estimate"], m["physical_counterfactual_separation"]["estimate"]))
    lines += ["", "| policy | success | block XY distance | final distance | return |", "|---|---:|---:|---:|---:|"]
    for item in plans:
        s = item["summary"]; lines.append("| %s | %.3f | %.3f | %.3f | %.3f |" % (item["policy"], s["success_rate"], s["mean_block_xy_distance"], s["mean_final_distance"], s["mean_return"]))
    lines += ["", "Claim boundary: pose-only interface diagnosis; no universal causal-control or planning claim.", ""]
    (out / "pose_closure.md").write_text("\n".join(lines)); print(json.dumps({"output": str(out), "planning": [x["summary"] for x in plans]}, indent=2))


if __name__ == "__main__": main()
