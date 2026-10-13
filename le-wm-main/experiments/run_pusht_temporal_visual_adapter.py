"""Test whether two-frame LeWM features repair the single-frame state interface."""

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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_pusht_counterfactual import encode_images, image_transform
from experiments.run_pusht_learned_physical_dynamics import reset_case, state_features, step_case


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--calibration", type=int, default=512)
    p.add_argument("--pairs", type=int, default=24)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--seed", type=int, default=9172)
    p.add_argument("--action-magnitude", type=float, default=0.55)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def bootstrap(values: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(draws, values.size))
    means = values[idx].mean(axis=1)
    return {
        "estimate": float(values.mean()), "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)), "n": int(values.size),
    }


def calibration_data(env: gym.Env, n: int, seed: int, horizon: int, magnitude: float):
    rng = np.random.default_rng(seed)
    previous, current, targets = [], [], []
    for i in range(n):
        case = reset_case(env, seed + i)
        direction = rng.normal(size=2)
        direction /= max(np.linalg.norm(direction), 1e-12)
        endpoint = step_case(
            env, case["state"], case["goal"],
            (magnitude * direction).astype(np.float32), horizon,
        )
        previous.append(case["image"])
        current.append(endpoint["image"])
        targets.append(state_features(endpoint["state"]))
    return previous, current, np.asarray(targets, dtype=np.float64)


def fit_decoder(z_prev: np.ndarray, z_current: np.ndarray, targets: np.ndarray, temporal: bool):
    x = np.concatenate([z_current, z_current - z_prev], axis=1) if temporal else z_current
    scaler = StandardScaler().fit(targets)
    decoder = Ridge(alpha=10.0).fit(x, scaler.transform(targets))
    return decoder, scaler


def decode(decoder, scaler, z_prev: np.ndarray, z_current: np.ndarray, temporal: bool) -> np.ndarray:
    x = np.concatenate([z_current, z_current - z_prev], axis=1) if temporal else z_current
    return scaler.inverse_transform(decoder.predict(x))


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = load_compatible_pretrained("pusht/lewm", device=args.device)
    transform = image_transform()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        cal_prev, cal_current, cal_targets = calibration_data(
            env, args.calibration, args.seed, args.horizon, args.action_magnitude
        )
        held_out = [reset_case(env, args.seed + 10000 + i) for i in range(args.pairs)]
        previous, current, true = [], [], []
        rng = np.random.default_rng(args.seed + 99)
        for case in held_out:
            direction = rng.normal(size=2)
            direction /= max(np.linalg.norm(direction), 1e-12)
            action = (args.action_magnitude * direction).astype(np.float32)
            for arm in (action, -action, np.zeros(2, dtype=np.float32)):
                endpoint = step_case(env, case["state"], case["goal"], arm, args.horizon)
                previous.append(case["image"])
                current.append(endpoint["image"])
                true.append(state_features(endpoint["state"]))
    finally:
        env.close()

    z_cal_prev = encode_images(model, cal_prev, transform, args.batch_size, args.device)
    z_cal_current = encode_images(model, cal_current, transform, args.batch_size, args.device)
    z_prev = encode_images(model, previous, transform, args.batch_size, args.device)
    z_current = encode_images(model, current, transform, args.batch_size, args.device)
    true = np.asarray(true, dtype=np.float64)
    results = []
    for name, temporal in (("static_single_frame", False), ("temporal_two_frame", True)):
        decoder, scaler = fit_decoder(z_cal_prev, z_cal_current, cal_targets, temporal)
        predicted = decode(decoder, scaler, z_prev, z_current, temporal)
        block_error = np.linalg.norm(predicted[:, 2:4] - true[:, 2:4], axis=1)
        agent_error = np.linalg.norm(predicted[:, :2] - true[:, :2], axis=1)
        velocity_error = np.linalg.norm(predicted[:, 6:8] - true[:, 6:8], axis=1)
        full_error = np.linalg.norm(predicted - true, axis=1)
        results.append({
            "decoder": name,
            "block_xy_error": bootstrap(block_error, args.seed + 11, args.bootstrap),
            "agent_xy_error": bootstrap(agent_error, args.seed + 12, args.bootstrap),
            "velocity_error": bootstrap(velocity_error, args.seed + 13, args.bootstrap),
            "full_state_feature_error": bootstrap(full_error, args.seed + 14, args.bootstrap),
            "finite": bool(np.isfinite(predicted).all()),
        })
    report = {
        "protocol": {
            "environment": "swm/PushT-v1", "checkpoint": str(getattr(model, "_checkpoint_compat", {})),
            "calibration_pairs": args.calibration, "held_out_pairs": args.pairs,
            "horizon": args.horizon, "action_magnitude": args.action_magnitude,
            "feature": "z_current for static; [z_current, z_current-z_previous] for temporal",
            "target": "agent XY, block XY, sin/cos block angle, agent velocity",
            "calibration_test_disjoint": True,
        },
        "results": results,
        "sanity_checks": {"finite_all": all(row["finite"] for row in results), "n_test_rows": int(true.shape[0])},
        "claim_boundary": "Two-frame decoder tests physical-state observability from frozen features; it does not establish learned causality, accurate counterfactual prediction, or planning gain.",
    }
    (out / "temporal_visual_adapter.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# PushT temporal visual adapter", "", "| decoder | block XY error | agent XY error | velocity error | full state error |", "|---|---:|---:|---:|---:|"]
    for row in results:
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f |" % (
            row["decoder"], row["block_xy_error"]["estimate"], row["agent_xy_error"]["estimate"],
            row["velocity_error"]["estimate"], row["full_state_feature_error"]["estimate"],
        ))
    lines += ["", "Claim boundary: temporal observability diagnostic; no causal-control or planning claim.", ""]
    (out / "temporal_visual_adapter.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(out), "results": results, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
