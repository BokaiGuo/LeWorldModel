"""Diagnose whether frozen-LeWM physical observability is a readout failure.

This is a decoder-only ablation: the LeWM checkpoint and held-out simulator
interventions are fixed, while the physical readout varies in target scope,
model class, and calibration-set size.  A failure of a linear full-pose
readout is not promoted to representation-level information absence unless
an XY-only nonlinear readout also fails on the same held-out images.
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
from experiments.run_pusht_counterfactual_observability_decomposition import (
    color_mask_features,
    pixel_features,
)
from experiments.run_pusht_counterfactual import (
    encode_images,
    image_transform,
    normalized_action_sequence,
    pose_target,
    predict_endpoints,
    reset_state,
    rollout,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--calibration", type=int, default=512)
    p.add_argument("--calibration-sizes", nargs="+", type=int, default=[64, 128, 256, 512])
    p.add_argument("--pairs", type=int, default=24)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--seed", type=int, default=8272)
    p.add_argument("--action-magnitude", type=float, default=0.55)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--trees", type=int, default=160)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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


def _sample_cases(env: gym.Env, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(args.seed)
    calibration, held_out = [], []
    for i in range(args.calibration):
        seed = args.seed + i
        state, goal, _ = reset_state(env, seed)
        direction = rng.normal(size=2)
        direction /= max(np.linalg.norm(direction), 1e-12)
        calibration.append({
            "seed": seed, "state": state, "goal": goal,
            "action": (args.action_magnitude * direction).astype(np.float32),
        })
    for i in range(args.pairs):
        seed = args.seed + 10000 + i
        state, goal, initial_image = reset_state(env, seed)
        direction = rng.normal(size=2)
        direction /= max(np.linalg.norm(direction), 1e-12)
        held_out.append({
            "seed": seed, "state": state, "goal": goal,
            "initial_image": initial_image,
            "action": (args.action_magnitude * direction).astype(np.float32),
        })
    return calibration, held_out


def _targets(cases: list[dict], env: gym.Env, horizon: int) -> tuple[list[np.ndarray], np.ndarray]:
    images, targets = [], []
    for case in cases:
        result = rollout(env, case["state"], case["goal"], case["action"], horizon, case["seed"])
        images.append(result["final_image"])
        targets.append(pose_target(result["final_state"]))
    return images, np.asarray(targets, dtype=np.float64)


def fit_predictor(name: str, x: np.ndarray, y: np.ndarray, args: argparse.Namespace):
    if name == "ridge_xy":
        scaler = StandardScaler().fit(y[:, :2])
        model = Ridge(alpha=1.0).fit(x, scaler.transform(y[:, :2]))
        return lambda values: scaler.inverse_transform(model.predict(values))
    if name == "ridge_full_pose":
        scaler = StandardScaler().fit(y)
        model = Ridge(alpha=1.0).fit(x, scaler.transform(y))
        return lambda values: scaler.inverse_transform(model.predict(values))[:, :2]
    if name == "extra_trees_xy":
        model = ExtraTreesRegressor(
            n_estimators=args.trees, min_samples_leaf=3, max_features=1.0,
            random_state=args.seed, n_jobs=1,
        ).fit(x, y[:, :2])
        return lambda values: model.predict(values)
    if name == "extra_trees_full_pose":
        model = ExtraTreesRegressor(
            n_estimators=args.trees, min_samples_leaf=3, max_features=1.0,
            random_state=args.seed, n_jobs=1,
        ).fit(x, y)
        return lambda values: model.predict(values)[:, :2]
    raise ValueError(name)


def evaluate_arm(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    observed_x: np.ndarray,
    predicted_x: np.ndarray,
    true_xy: np.ndarray,
    initial_x: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    fit = fit_predictor(name, train_x, train_y, args)
    observed = fit(observed_x)
    predicted = fit(predicted_x)
    initial = fit(initial_x)
    errors = {
        "observed": np.linalg.norm(observed - true_xy, axis=1),
        "predicted": np.linalg.norm(predicted - true_xy, axis=1),
        "initial": np.linalg.norm(initial - true_xy, axis=1),
    }
    return {
        "decoder": name,
        "errors": {key: bootstrap(value, args.seed + 100 + i, args.bootstrap)
                    for i, (key, value) in enumerate(errors.items())},
        "finite": bool(np.isfinite(observed).all() and np.isfinite(predicted).all() and np.isfinite(initial).all()),
    }


def main() -> None:
    args = parse_args()
    if args.calibration < max(args.calibration_sizes) or args.pairs < 8 or args.horizon < 1:
        raise ValueError("calibration must cover all requested sizes and pairs must be >= 8")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    args.calibration_sizes = sorted(set(args.calibration_sizes))
    model = load_compatible_pretrained("pusht/lewm", device=args.device)
    transform = image_transform()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        calibration, held_out = _sample_cases(env, args)
        calibration_images, calibration_y = _targets(env=env, cases=calibration, horizon=args.horizon)
        initial_images = [case["initial_image"] for case in held_out]
        final_images, sequences, true_xy = [], [], []
        for case in held_out:
            actions = [case["action"], -case["action"], np.zeros(2, dtype=np.float32)]
            sequences.append(np.stack([
                normalized_action_sequence(action, args.horizon) for action in actions
            ]))
            endpoints = [rollout(env, case["state"], case["goal"], action, args.horizon, case["seed"])
                         for action in actions]
            final_images.extend(item["final_image"] for item in endpoints)
            true_xy.append(np.stack([pose_target(item["final_state"])[:2] for item in endpoints]))
        true_xy = np.asarray(true_xy).reshape(-1, 2)
    finally:
        env.close()

    calibration_z = encode_images(model, calibration_images, transform, args.batch_size, args.device)
    initial_z = encode_images(model, initial_images, transform, args.batch_size, args.device)
    observed_z = encode_images(model, final_images, transform, args.batch_size, args.device)
    predicted_z = predict_endpoints(model, initial_images, np.asarray(sequences), transform, args.batch_size, args.device)
    predicted_z = predicted_z.reshape(-1, predicted_z.shape[-1])
    initial_z = np.repeat(initial_z, 3, axis=0)

    controls = {
        "latent": (observed_z, predicted_z, initial_z),
        "raw_pixels": (
            pixel_features(final_images), pixel_features(final_images), pixel_features(initial_images).repeat(3, axis=0)
        ),
        "block_mask": (
            color_mask_features(final_images), color_mask_features(final_images), color_mask_features(initial_images).repeat(3, axis=0)
        ),
    }
    # Image controls do not have a distinct predicted arm; using the observed
    # image for that column is intentional and makes them capacity references.
    results = []
    for size in args.calibration_sizes:
        for control_name, (observed_x, predicted_x, initial_x) in controls.items():
            train_x = calibration_z[:size] if control_name == "latent" else (
                pixel_features(calibration_images)[:size] if control_name == "raw_pixels"
                else color_mask_features(calibration_images)[:size]
            )
            train_y = calibration_y[:size]
            if control_name == "latent":
                decoder_names = ["ridge_xy", "ridge_full_pose", "extra_trees_xy", "extra_trees_full_pose"]
            elif control_name == "raw_pixels":
                decoder_names = ["ridge_xy"]
            else:
                decoder_names = ["ridge_xy"]
            for decoder_name in decoder_names:
                result = evaluate_arm(decoder_name, train_x, train_y, observed_x, predicted_x, true_xy, initial_x, args)
                result.update({"calibration_size": size, "control": control_name})
                results.append(result)

    report = {
        "protocol": {
            "environment": "swm/PushT-v1", "checkpoint": str(getattr(model, "_checkpoint_compat", {})),
            "horizon": args.horizon, "calibration_rollouts": args.calibration, "held_out_pairs": args.pairs,
            "calibration_sizes": args.calibration_sizes, "target": "block XY only for XY decoders; full pose is an ablation",
            "test_split": "same held-out +a/-a/zero images and predicted latent endpoints for every decoder",
            "image_controls": "raw 32x32 pixels and grey-blue movable-block mask; no green goal mask",
        },
        "results": results,
        "sanity_checks": {
            "calibration_disjoint": True,
            "all_finite": all(item["finite"] for item in results),
            "all_sizes_present": sorted(set(item["calibration_size"] for item in results)) == args.calibration_sizes,
            "held_out_rows": len(true_xy) == args.pairs * 3,
        },
        "claim_boundary": "Decoder ablations diagnose frozen-latent physical observability; they do not establish causal action effects, accurate physical counterfactual prediction, online planning gain, or task success.",
    }
    (output / "observability_decoder_ablation.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# PushT observability decoder ablation", "",
        "All decoders use the same disjoint calibration/test split and the same held-out intervention images.", "",
        "| size | control | decoder | observed error | predicted error | initial error |", "|---:|---|---|---:|---:|---:|",
    ]
    for item in results:
        e = item["errors"]
        lines.append("| %d | %s | %s | %.3f | %.3f | %.3f |" % (
            item["calibration_size"], item["control"], item["decoder"],
            e["observed"]["estimate"], e["predicted"]["estimate"], e["initial"]["estimate"],
        ))
    lines += ["", "Claim boundary: decoder-only physical-observability diagnosis; no causal-control or planning claim.", ""]
    (output / "observability_decoder_ablation.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
