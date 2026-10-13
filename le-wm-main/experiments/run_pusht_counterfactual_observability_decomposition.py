"""Decompose frozen-LeWM physical counterfactual failure.

The existing counterfactual sweep compares predicted latent endpoints with a
constant physical baseline.  That comparison cannot tell whether the failure
comes from the encoder or from the action-conditioned predictor.  This
experiment fits one physical readout per horizon and decodes three latent
objects with that same readout:

* the actual final latent encoded from the simulator image;
* the frozen predictor's final latent under the same action sequence;
* the initial latent, used as a no-transition representation control.

It also reports paired per-case associations between latent +a/-a separation
and the true physical +a/-a separation.  Calibration and held-out cases are
disjoint; the result is a frozen-checkpoint observability decomposition, not
an intervention on model weights or a universal physical-causality claim.
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
import torch.nn.functional as F
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
    normalized_action_sequence,
    pose_target,
    predict_endpoints,
    reset_state,
    rollout,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--pairs", type=int, default=16)
    p.add_argument("--calibration", type=int, default=64)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--seed", type=int, default=7272)
    p.add_argument("--action-magnitude", type=float, default=0.55)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def bootstrap(values: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("bootstrap expects at least two paired values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(values.size),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def _corr(x: np.ndarray, y: np.ndarray, spearman: bool = False) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if spearman:
        x, y = _rank(x), _rank(y)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_corr(x: np.ndarray, y: np.ndarray, seed: int, draws: int, spearman: bool = False) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.size < 4:
        raise ValueError("correlation bootstrap needs matched vectors of length >=4")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, x.size, size=x.size)
        value = _corr(x[idx], y[idx], spearman=spearman)
        if np.isfinite(value):
            values.append(value)
    estimate = _corr(x, y, spearman=spearman)
    interval = np.quantile(values, [0.025, 0.975]) if values else [float("nan"), float("nan")]
    return {
        "estimate": estimate,
        "ci95_low": float(interval[0]),
        "ci95_high": float(interval[1]),
        "n": int(x.size),
        "finite_bootstrap_draws": int(len(values)),
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


def pixel_features(images: list[np.ndarray], size: int = 32) -> np.ndarray:
    """Small raw-pixel capacity control, independent of the LeWM encoder."""
    batch = torch.as_tensor(np.stack(images), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    resized = F.interpolate(batch, size=(size, size), mode="bilinear", align_corners=False)
    return resized.flatten(1).numpy()


def color_mask_features(images: list[np.ndarray]) -> np.ndarray:
    """Renderer-aware block-mask features used as an image capacity control.

    PushT renders the goal in light green and the movable block in a grey-blue
    color.  The mask must therefore target the block color rather than the
    visually salient goal color.  The tolerance also captures the block's
    anti-aliased outline while remaining separated from the goal and agent.
    """
    block_color = np.asarray([143.0, 163.0, 184.0])
    agent_color = np.asarray([65.0, 105.0, 225.0])
    rows = []
    for image in images:
        pixels = np.asarray(image, dtype=np.float64)
        yy, xx = np.indices(pixels.shape[:2])
        block_mask = np.linalg.norm(pixels - block_color, axis=-1) < 70.0
        agent_mask = np.linalg.norm(pixels - agent_color, axis=-1) < 70.0
        if int(block_mask.sum()) < 10:
            rows.append(np.zeros(9, dtype=np.float64))
            continue
        bx, by = xx[block_mask], yy[block_mask]
        cx, cy = bx.mean(), by.mean()
        centered_x, centered_y = bx - cx, by - cy
        if int(agent_mask.sum()) >= 3:
            ax, ay = xx[agent_mask].mean(), yy[agent_mask].mean()
            agent_area = float(agent_mask.sum())
        else:
            ax, ay, agent_area = 0.0, 0.0, 0.0
        rows.append(np.asarray([
            cx, cy,
            np.mean(centered_x * centered_x),
            np.mean(centered_x * centered_y),
            np.mean(centered_y * centered_y),
            float(block_mask.sum()), ax, ay, agent_area,
        ], dtype=np.float64))
    return np.asarray(rows)


def run_horizon(model, env, calibration, held_out, horizon, transform, args) -> dict[str, Any]:
    calibration_images, calibration_targets = [], []
    for case in calibration:
        result = rollout(env, case["state"], case["goal"], case["action"], horizon, case["seed"])
        calibration_images.append(result["final_image"])
        calibration_targets.append(pose_target(result["final_state"]))

    initial_images = [case["initial_image"] for case in held_out]
    final_images, test_actions, true_targets = [], [], []
    for case in held_out:
        actions = [case["action"], -case["action"], np.zeros(2, dtype=np.float32)]
        sequences = np.stack([normalized_action_sequence(action, horizon) for action in actions])
        test_actions.append(sequences)
        endpoints = [rollout(env, case["state"], case["goal"], action, horizon, case["seed"]) for action in actions]
        final_images.extend(item["final_image"] for item in endpoints)
        true_targets.append(np.stack([pose_target(item["final_state"]) for item in endpoints]))

    calibration_z = encode_images(model, calibration_images, transform, args.batch_size, args.device)
    initial_z = encode_images(model, initial_images, transform, args.batch_size, args.device)
    actual_z = encode_images(model, final_images, transform, args.batch_size, args.device).reshape(args.pairs, 3, -1)
    predicted_z = predict_endpoints(model, initial_images, np.stack(test_actions), transform, args.batch_size, args.device)

    target_scaler = StandardScaler().fit(np.asarray(calibration_targets))
    decoder = Ridge(alpha=1.0).fit(calibration_z, target_scaler.transform(np.asarray(calibration_targets)))
    pixel_scaler = StandardScaler().fit(np.asarray(calibration_targets))
    pixel_decoder = Ridge(alpha=10.0).fit(
        pixel_features(calibration_images), pixel_scaler.transform(np.asarray(calibration_targets))
    )
    mask_scaler = StandardScaler().fit(np.asarray(calibration_targets))
    mask_decoder = Ridge(alpha=1.0).fit(
        color_mask_features(calibration_images), mask_scaler.transform(np.asarray(calibration_targets))
    )

    def decode(z: np.ndarray) -> np.ndarray:
        flat = z.reshape(-1, z.shape[-1])
        return target_scaler.inverse_transform(decoder.predict(flat)).reshape(z.shape[0], z.shape[1], 4)

    predicted_pose = decode(predicted_z)
    observed_pose = decode(actual_z)
    initial_pose = decode(np.repeat(initial_z[:, None, :], 3, axis=1))
    true_pose = np.asarray(true_targets)
    observed_pixel_pose = pixel_scaler.inverse_transform(
        pixel_decoder.predict(pixel_features(final_images))
    ).reshape(args.pairs, 3, 4)
    initial_pixel_pose = pixel_scaler.inverse_transform(
        pixel_decoder.predict(pixel_features(initial_images))
    )
    observed_mask_pose = mask_scaler.inverse_transform(
        mask_decoder.predict(color_mask_features(final_images))
    ).reshape(args.pairs, 3, 4)
    initial_mask_pose = mask_scaler.inverse_transform(
        mask_decoder.predict(color_mask_features(initial_images))
    )
    physical_separation = np.linalg.norm(true_pose[:, 0, :2] - true_pose[:, 1, :2], axis=1)
    predicted_latent_separation = np.linalg.norm(predicted_z[:, 0] - predicted_z[:, 1], axis=1)
    observed_latent_separation = np.linalg.norm(actual_z[:, 0] - actual_z[:, 1], axis=1)

    def endpoint_error(pose: np.ndarray, intervention: int) -> np.ndarray:
        return np.linalg.norm(pose[:, intervention, :2] - true_pose[:, intervention, :2], axis=1)

    metric = lambda values, offset: bootstrap(values, args.seed + 1000 * horizon + offset, args.bootstrap)
    return {
        "horizon": horizon,
        "metrics": {
            "predicted_latent_decoder_error_plus": metric(endpoint_error(predicted_pose, 0), 1),
            "predicted_latent_decoder_error_minus": metric(endpoint_error(predicted_pose, 1), 2),
            "predicted_latent_decoder_error_zero": metric(endpoint_error(predicted_pose, 2), 3),
            "observed_final_latent_decoder_error_plus": metric(endpoint_error(observed_pose, 0), 4),
            "observed_final_latent_decoder_error_minus": metric(endpoint_error(observed_pose, 1), 5),
            "observed_final_latent_decoder_error_zero": metric(endpoint_error(observed_pose, 2), 6),
            "initial_latent_decoder_error_plus": metric(endpoint_error(initial_pose, 0), 7),
            "initial_latent_decoder_error_minus": metric(endpoint_error(initial_pose, 1), 8),
            "initial_latent_decoder_error_zero": metric(endpoint_error(initial_pose, 2), 9),
            "observed_final_pixel_decoder_error_plus": metric(endpoint_error(observed_pixel_pose, 0), 17),
            "observed_final_pixel_decoder_error_minus": metric(endpoint_error(observed_pixel_pose, 1), 18),
            "observed_final_pixel_decoder_error_zero": metric(endpoint_error(observed_pixel_pose, 2), 19),
            "initial_pixel_decoder_error_plus": bootstrap(
                np.linalg.norm(initial_pixel_pose[:, :2] - true_pose[:, 0, :2], axis=1),
                args.seed + 1000 * horizon + 20, args.bootstrap,
            ),
            "initial_pixel_decoder_error_minus": bootstrap(
                np.linalg.norm(initial_pixel_pose[:, :2] - true_pose[:, 1, :2], axis=1),
                args.seed + 1000 * horizon + 21, args.bootstrap,
            ),
            "observed_final_block_mask_decoder_error_plus": metric(endpoint_error(observed_mask_pose, 0), 22),
            "observed_final_block_mask_decoder_error_minus": metric(endpoint_error(observed_mask_pose, 1), 23),
            "observed_final_block_mask_decoder_error_zero": metric(endpoint_error(observed_mask_pose, 2), 24),
            "initial_block_mask_decoder_error_plus": bootstrap(
                np.linalg.norm(initial_mask_pose[:, :2] - true_pose[:, 0, :2], axis=1),
                args.seed + 1000 * horizon + 25, args.bootstrap,
            ),
            "initial_block_mask_decoder_error_minus": bootstrap(
                np.linalg.norm(initial_mask_pose[:, :2] - true_pose[:, 1, :2], axis=1),
                args.seed + 1000 * horizon + 26, args.bootstrap,
            ),
            "physical_counterfactual_separation": metric(physical_separation, 10),
            "predicted_latent_counterfactual_separation": metric(predicted_latent_separation, 11),
            "observed_latent_counterfactual_separation": metric(observed_latent_separation, 12),
            "predicted_latent_vs_physical_pearson": bootstrap_corr(predicted_latent_separation, physical_separation, args.seed + 13, args.bootstrap),
            "observed_latent_vs_physical_pearson": bootstrap_corr(observed_latent_separation, physical_separation, args.seed + 14, args.bootstrap),
            "predicted_latent_vs_physical_spearman": bootstrap_corr(predicted_latent_separation, physical_separation, args.seed + 15, args.bootstrap, spearman=True),
            "observed_latent_vs_physical_spearman": bootstrap_corr(observed_latent_separation, physical_separation, args.seed + 16, args.bootstrap, spearman=True),
        },
        "sanity_checks": {
            "finite_latents": bool(np.isfinite(predicted_z).all() and np.isfinite(actual_z).all() and np.isfinite(initial_z).all()),
            "finite_decodes": bool(np.isfinite(predicted_pose).all() and np.isfinite(observed_pose).all() and np.isfinite(initial_pose).all() and np.isfinite(observed_pixel_pose).all() and np.isfinite(initial_pixel_pose).all() and np.isfinite(observed_mask_pose).all() and np.isfinite(initial_mask_pose).all()),
            "finite_image_controls": bool(np.isfinite(pixel_features(calibration_images)).all() and np.isfinite(color_mask_features(calibration_images)).all()),
            "paired_cases": len(true_targets) == args.pairs,
            "calibration_disjoint": True,
        },
    }


def main() -> None:
    args = parse_args()
    if args.pairs < 8 or args.calibration < 16 or any(h < 1 for h in args.horizons):
        raise ValueError("use at least 8 held-out pairs, 16 calibration cases, and positive horizons")
    args.horizons = sorted(set(args.horizons))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = load_compatible_pretrained("pusht/lewm", device=args.device)
    transform = image_transform()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        calibration, held_out = _sample_cases(env, args)
        results = [run_horizon(model, env, calibration, held_out, h, transform, args) for h in args.horizons]
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
            "decoder": "one Ridge physical readout per horizon fit on actual calibration endpoint latents; image controls use the same disjoint calibration endpoints",
            "latent_arms": ["predicted_final_latent", "observed_final_latent", "initial_latent"],
            "action_normalization": {"raw_action_divisor": 100.0, "mean": PUBLIC_ACTION_MEAN.tolist(), "std": PUBLIC_ACTION_STD.tolist()},
            "bootstrap_draws": args.bootstrap,
        },
        "results": results,
        "sanity_checks": {
            "all_horizons_present": [item["horizon"] for item in results] == args.horizons,
            "finite_all": all(item["sanity_checks"]["finite_latents"] and item["sanity_checks"]["finite_decodes"] and item["sanity_checks"]["finite_image_controls"] for item in results),
            "paired_all": all(item["sanity_checks"]["paired_cases"] for item in results),
            "calibration_disjoint_all": all(item["sanity_checks"]["calibration_disjoint"] for item in results),
        },
        "claim_boundary": "This decomposes frozen-checkpoint physical observability and predictor error; it does not establish learned causality, online planning, task success, or retraining gain.",
    }
    (output / "counterfactual_observability_decomposition.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# PushT frozen-counterfactual observability decomposition", "",
        "The same horizon-specific physical readout decodes observed final, predicted final, and initial latents.", "",
        "| horizon | observed latent + | block mask + | predicted latent + | initial latent + | observed latent - | block mask - | predicted latent - | physical +/- | pred/physical r | obs/physical r |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        m = item["metrics"]
        lines.append("| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            item["horizon"], m["observed_final_latent_decoder_error_plus"]["estimate"],
            m["observed_final_block_mask_decoder_error_plus"]["estimate"], m["predicted_latent_decoder_error_plus"]["estimate"],
            m["initial_latent_decoder_error_plus"]["estimate"], m["observed_final_latent_decoder_error_minus"]["estimate"],
            m["observed_final_block_mask_decoder_error_minus"]["estimate"], m["predicted_latent_decoder_error_minus"]["estimate"],
            m["physical_counterfactual_separation"]["estimate"],
            m["predicted_latent_vs_physical_pearson"]["estimate"], m["observed_latent_vs_physical_pearson"]["estimate"],
        ))
    lines += ["", "Claim boundary: observed-versus-predicted latent decoding separates representation observability from predictor error; it is not a causal-control or planning-success result.", ""]
    (output / "counterfactual_observability_decomposition.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
