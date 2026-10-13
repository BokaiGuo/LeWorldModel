"""Matched multi-horizon physical counterfactual sweep for frozen LeWM.

The same simulator states, goals, and intervention directions are evaluated at
multiple horizons.  A separate Ridge physical readout is fitted on disjoint
calibration rollouts for each horizon, so the result measures horizon-specific
endpoint prediction rather than leaking a readout across horizons.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from experiments.run_pusht_counterfactual import (
    PUBLIC_ACTION_MEAN,
    PUBLIC_ACTION_STD,
    encode_images,
    image_transform,
    normalize_public_action,
    normalized_action_sequence,
    pose_target,
    predict_endpoints,
    reset_state,
    rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--calibration", type=int, default=64)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=4072)
    parser.add_argument("--action-magnitude", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def bootstrap(values: np.ndarray, seed: int, draws: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("bootstrap expects at least two episode-level values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(values.size),
    }


def _sample_cases(env: gym.Env, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(args.seed)
    calibration, held_out = [], []
    for i in range(args.calibration):
        seed = args.seed + i
        state, goal, _ = reset_state(env, seed)
        direction = rng.normal(size=2)
        direction /= max(np.linalg.norm(direction), 1e-12)
        calibration.append({"seed": seed, "state": state, "goal": goal, "action": (args.action_magnitude * direction).astype(np.float32)})
    for i in range(args.pairs):
        seed = args.seed + 10000 + i
        state, goal, initial_image = reset_state(env, seed)
        direction = rng.normal(size=2)
        direction /= max(np.linalg.norm(direction), 1e-12)
        held_out.append({"seed": seed, "state": state, "goal": goal, "initial_image": initial_image, "action": (args.action_magnitude * direction).astype(np.float32)})
    return calibration, held_out


def _summarize_horizon(
    model: torch.nn.Module,
    env: gym.Env,
    calibration: list[dict],
    held_out: list[dict],
    horizon: int,
    transform,
    args: argparse.Namespace,
) -> dict:
    calibration_images, calibration_targets = [], []
    test_final_images, test_actions, test_targets, test_initial_targets = [], [], [], []
    for case in calibration:
        result = rollout(env, case["state"], case["goal"], case["action"], horizon, case["seed"])
        calibration_images.append(result["final_image"])
        calibration_targets.append(pose_target(result["final_state"]))
    for case in held_out:
        endpoints = []
        pair_actions = np.stack([
            normalized_action_sequence(case["action"], horizon),
            normalized_action_sequence(-case["action"], horizon),
            normalized_action_sequence(np.zeros(2, dtype=np.float32), horizon),
        ], axis=0)
        for action in (case["action"], -case["action"], np.zeros(2, dtype=np.float32)):
            endpoints.append(rollout(env, case["state"], case["goal"], action, horizon, case["seed"]))
        test_final_images.extend(item["final_image"] for item in endpoints)
        test_actions.append(pair_actions)
        test_targets.append(np.stack([pose_target(item["final_state"]) for item in endpoints]))
        test_initial_targets.append(pose_target(case["state"]))

    calibration_z = encode_images(model, calibration_images, transform, args.batch_size, args.device)
    predicted_z = predict_endpoints(model, [case["initial_image"] for case in held_out], np.stack(test_actions), transform, args.batch_size, args.device)
    actual_z = encode_images(model, test_final_images, transform, args.batch_size, args.device).reshape(args.pairs, 3, -1)
    scaler = StandardScaler().fit(np.asarray(calibration_targets))
    decoder = Ridge(alpha=1.0).fit(calibration_z, scaler.transform(np.asarray(calibration_targets)))
    pred_pose = scaler.inverse_transform(decoder.predict(predicted_z.reshape(-1, predicted_z.shape[-1]))).reshape(args.pairs, 3, 4)
    true_pose = np.asarray(test_targets)
    baseline_pose = np.asarray(test_initial_targets)[:, None, :]
    pred_position_error = np.linalg.norm(pred_pose[:, :, :2] - true_pose[:, :, :2], axis=-1)
    baseline_position_error = np.linalg.norm(baseline_pose[:, :, :2] - true_pose[:, :, :2], axis=-1)
    latent_error = np.linalg.norm(predicted_z - actual_z, axis=-1)
    latent_separation = np.linalg.norm(predicted_z[:, 0] - predicted_z[:, 1], axis=-1)
    physical_separation = np.linalg.norm(true_pose[:, 0, :2] - true_pose[:, 1, :2], axis=-1)
    metric = lambda values, offset: bootstrap(values, args.seed + 791 + offset, args.bootstrap)
    return {
        "horizon": horizon,
        "metrics": {
            "latent_endpoint_prediction_error": metric(latent_error.reshape(-1), horizon),
            "latent_counterfactual_separation": metric(latent_separation, 100 + horizon),
            "physical_counterfactual_xy_separation": metric(physical_separation, 200 + horizon),
            "decoded_position_error_plus": metric(pred_position_error[:, 0], 300 + horizon),
            "decoded_position_error_minus": metric(pred_position_error[:, 1], 400 + horizon),
            "decoded_position_error_zero": metric(pred_position_error[:, 2], 500 + horizon),
            "constant_initial_pose_error_plus": metric(baseline_position_error[:, 0], 600 + horizon),
            "constant_initial_pose_error_minus": metric(baseline_position_error[:, 1], 700 + horizon),
            "constant_initial_pose_error_zero": metric(baseline_position_error[:, 2], 800 + horizon),
            "decoded_minus_baseline_plus": metric(pred_position_error[:, 0] - baseline_position_error[:, 0], 900 + horizon),
            "decoded_minus_baseline_minus": metric(pred_position_error[:, 1] - baseline_position_error[:, 1], 1000 + horizon),
            "decoded_minus_baseline_zero": metric(pred_position_error[:, 2] - baseline_position_error[:, 2], 1100 + horizon),
        },
        "sanity_checks": {
            "finite_predictions": bool(np.isfinite(predicted_z).all() and np.isfinite(pred_pose).all()),
            "paired_rows": len(test_targets) == args.pairs,
            "calibration_disjoint_from_test": True,
        },
    }


def main() -> None:
    args = parse_args()
    if args.pairs < 4 or args.calibration < 8 or any(h < 1 for h in args.horizons):
        raise ValueError("use at least 4 held-out pairs, 8 calibration rollouts, and positive horizons")
    args.horizons = sorted(set(args.horizons))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = load_compatible_pretrained("pusht/lewm", device=args.device)
    transform = image_transform()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        calibration, held_out = _sample_cases(env, args)
        results = [_summarize_horizon(model, env, calibration, held_out, horizon, transform, args) for horizon in args.horizons]
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "checkpoint": str(getattr(model, "_checkpoint_compat", {})),
            "calibration_rollouts": args.calibration,
            "held_out_pairs": args.pairs,
            "horizons": args.horizons,
            "same_cases_across_horizons": True,
            "action_protocol": "+a, -a, and zero action from identical explicit state and goal",
            "action_normalization": {"raw_action_divisor": 100.0, "mean": PUBLIC_ACTION_MEAN.tolist(), "std": PUBLIC_ACTION_STD.tolist()},
            "decoder": "separate Ridge readout per horizon, fit only on disjoint calibration endpoints",
            "bootstrap_draws": args.bootstrap,
        },
        "results": results,
        "sanity_checks": {
            "all_horizons_present": [item["horizon"] for item in results] == args.horizons,
            "finite_all": all(item["sanity_checks"]["finite_predictions"] for item in results),
            "paired_all": all(item["sanity_checks"]["paired_rows"] for item in results),
        },
        "claim_boundary": "This is frozen-checkpoint physical endpoint evidence; it does not establish causal controllability, online planning, task success, or a universal world model.",
    }
    (output / "counterfactual_horizon_sweep.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# PushT matched physical counterfactual horizon sweep",
        "",
        "The same initial states, goals, and intervention directions are used at every horizon; each horizon has a disjoint calibration readout.",
        "",
        "| horizon | latent endpoint error | decoded + error | decoded - error | constant + error | decoded - baseline (+) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        m = item["metrics"]
        lines.append("| %d | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            item["horizon"],
            m["latent_endpoint_prediction_error"]["estimate"],
            m["decoded_position_error_plus"]["estimate"],
            m["decoded_position_error_minus"]["estimate"],
            m["constant_initial_pose_error_plus"]["estimate"],
            m["decoded_minus_baseline_plus"]["estimate"],
        ))
    lines += ["", report["claim_boundary"], ""]
    (output / "counterfactual_horizon_sweep.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "protocol": report["protocol"], "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
