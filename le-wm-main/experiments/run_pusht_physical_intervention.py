"""Matched physical action interventions in the PushT simulator.

This is deliberately independent of the learned model and of the training
dataset.  Each pair starts from the same explicit simulator state and goal,
then receives equal-norm opposite actions (plus a zero-action control).  The
result is a simulator-level intervention estimate, not an observational
action/latent correlation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401  # registers swm/PushT-v1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--pairs", type=int, default=32)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--action-magnitude", type=float, default=0.55)
    return p.parse_args()


def _pose_features(state: np.ndarray) -> np.ndarray:
    """Return physically interpretable block pose features."""
    return np.asarray([state[2], state[3], np.sin(state[4]), np.cos(state[4])], dtype=np.float64)


def _short_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _run(env: gym.Env, state: np.ndarray, goal: np.ndarray, action: np.ndarray, horizon: int, seed: int) -> dict:
    obs, info = env.reset(seed=seed, options={"state": state.copy(), "goal_state": goal.copy()})
    del obs
    initial = np.asarray(info["block_pose"], dtype=np.float64).copy()
    contacts = 0
    rewards = []
    for _ in range(horizon):
        _, reward, terminated, truncated, step_info = env.step(action.astype(np.float32, copy=False))
        rewards.append(float(reward))
        contacts += int(step_info.get("n_contacts", 0) > 0)
        if terminated or truncated:
            break
    final = np.asarray(step_info["block_pose"], dtype=np.float64).copy()
    goal_pose = np.asarray(step_info["goal_pose"], dtype=np.float64)
    block_delta = final - initial
    block_delta[2] = _short_angle(block_delta[2])
    goal_delta = final - goal_pose
    goal_delta[2] = _short_angle(goal_delta[2])
    return {
        "initial_block_pose": initial,
        "final_block_pose": final,
        "block_delta": block_delta,
        "goal_distance": float(np.linalg.norm(goal_delta)),
        "block_xy_displacement": float(np.linalg.norm(block_delta[:2])),
        "block_pose_displacement": float(np.linalg.norm(block_delta)),
        "contacts": int(contacts),
        "reward_sum": float(np.sum(rewards)),
        "steps": int(len(rewards)),
    }


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap input must be a non-empty vector")
    if draws <= 0:
        raise ValueError("bootstrap count must be positive")
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n_pairs": int(values.size),
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    if args.pairs < 2 or args.horizon < 1:
        raise ValueError("pairs must be >=2 and horizon must be >=1")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    plus_env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    minus_env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    zero_env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    rows = []
    try:
        for pair_id in range(args.pairs):
            seed = args.seed + pair_id
            probe = gym.make("swm/PushT-v1", render_mode="rgb_array")
            try:
                initial_obs, initial_info = probe.reset(seed=seed)
                # The simulator exposes the actual initial state through its observation.
                state = np.asarray(initial_obs["state"], dtype=np.float64).copy()
                goal = np.asarray(initial_info["goal_state"], dtype=np.float64).copy()
            finally:
                probe.close()
            direction = rng.normal(size=2)
            direction /= max(np.linalg.norm(direction), 1e-12)
            action = args.action_magnitude * direction
            plus = _run(plus_env, state, goal, action, args.horizon, seed)
            minus = _run(minus_env, state, goal, -action, args.horizon, seed)
            zero = _run(zero_env, state, goal, np.zeros(2), args.horizon, seed)
            contrast = {
                "block_xy_effect": float(np.linalg.norm(np.asarray(plus["final_block_pose"][:2]) - np.asarray(minus["final_block_pose"][:2]))),
                "block_pose_effect": float(np.linalg.norm(np.asarray(plus["final_block_pose"]) - np.asarray(minus["final_block_pose"]))),
                "goal_distance_effect": float(minus["goal_distance"] - plus["goal_distance"]),
                "reward_effect": float(plus["reward_sum"] - minus["reward_sum"]),
                "contact_effect": float(plus["contacts"] - minus["contacts"]),
            }
            rows.append({"pair_id": pair_id, "seed": seed, "action": action, "plus": plus, "minus": minus, "zero": zero, "contrast": contrast})
    finally:
        plus_env.close()
        minus_env.close()
        zero_env.close()

    def vector(name: str) -> np.ndarray:
        return np.asarray([row["contrast"][name] for row in rows], dtype=np.float64)

    bootstrap_rng = np.random.default_rng(args.seed + 991)
    metrics = {
        "block_xy_effect": _bootstrap_mean(vector("block_xy_effect"), bootstrap_rng, args.bootstrap),
        "block_pose_effect": _bootstrap_mean(vector("block_pose_effect"), bootstrap_rng, args.bootstrap),
        "goal_distance_effect_plus_minus": _bootstrap_mean(vector("goal_distance_effect"), bootstrap_rng, args.bootstrap),
        "reward_effect_plus_minus": _bootstrap_mean(vector("reward_effect"), bootstrap_rng, args.bootstrap),
        "contact_effect_plus_minus": _bootstrap_mean(vector("contact_effect"), bootstrap_rng, args.bootstrap),
        "plus_contact_rate": float(np.mean([row["plus"]["contacts"] > 0 for row in rows])),
        "minus_contact_rate": float(np.mean([row["minus"]["contacts"] > 0 for row in rows])),
        "zero_contact_rate": float(np.mean([row["zero"]["contacts"] > 0 for row in rows])),
        "plus_mean_xy_displacement": float(np.mean([row["plus"]["block_xy_displacement"] for row in rows])),
        "minus_mean_xy_displacement": float(np.mean([row["minus"]["block_xy_displacement"] for row in rows])),
        "zero_mean_xy_displacement": float(np.mean([row["zero"]["block_xy_displacement"] for row in rows])),
    }
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "design": "same explicit initial state and goal; equal-norm opposite action intervention plus zero-action control",
            "pairs": args.pairs,
            "horizon": args.horizon,
            "action_magnitude": args.action_magnitude,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "estimand": "paired simulator intervention contrast; not an observational or learned-model causal claim",
        },
        "metrics": metrics,
        "sanity_checks": {
            "finite_all": bool(np.isfinite(np.asarray(vector("block_xy_effect"))).all()),
            "paired_rows": len(rows) == args.pairs,
            "opposite_action_norms": bool(all(np.isclose(np.linalg.norm(row["action"]), args.action_magnitude) for row in rows)),
            "zero_control_present": True,
        },
        "rows": rows,
    }
    (out / "physical_intervention.json").write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    md = [
        "# PushT matched physical intervention",
        "",
        "This report uses paired simulator interventions from identical explicit states and goals.",
        "It establishes a simulator-level action effect only; it does not establish learned-model causality or online planning.",
        "",
        "| estimand | estimate | 95% paired-bootstrap CI |",
        "|---|---:|---:|",
    ]
    for key, label in [("block_xy_effect", "block XY effect"), ("block_pose_effect", "block pose effect"), ("goal_distance_effect_plus_minus", "goal-distance effect (+ minus -)"), ("reward_effect_plus_minus", "reward effect (+ minus -)")]:
        item = metrics[key]
        md.append(f"| {label} | {item['estimate']:.6f} | [{item['ci95_low']:.6f}, {item['ci95_high']:.6f}] |")
    md += ["", f"Pairs: {args.pairs}; horizon: {args.horizon}; action magnitude: {args.action_magnitude}; bootstrap draws: {args.bootstrap}.", ""]
    (out / "physical_intervention.md").write_text("\n".join(md))
    print(json.dumps({"output": str(out), "metrics": metrics, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
