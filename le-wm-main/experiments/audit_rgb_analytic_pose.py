"""Audit whether PushT block orientation is recoverable from RGB masks alone.

This is a frozen-observer diagnostic, not a learned controller.  Simulator
state is used only for the offline reference angle and contact strata.  The
mask moments are computed in one CUDA batch; a small calibration split is
allowed to fit the image/world handedness and angular offset, and all reported
test metrics are root-disjoint.

The audit separates two questions that were previously entangled:

* second moments: is the block's principal axis visible at all (orientation
  modulo pi)?
* third moments: does the asymmetric T silhouette identify the directed angle?

If the first succeeds while the second fails, RGB pose is visible but the
directional chart is ambiguous.  If both succeed but learned action ranking
still fails, the remaining bottleneck is action-effect identifiability.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import torch

try:
    from .run_causal_visual_pusht_world_model import (
        MaskMomentGeometryEncoder,
        balanced_source_cases,
        resize_image,
    )
except ImportError:
    from run_causal_visual_pusht_world_model import (  # type: ignore
        MaskMomentGeometryEncoder,
        balanced_source_cases,
        resize_image,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--roots", type=int, default=2048)
    parser.add_argument("--root-rollout-steps", type=int, default=8)
    parser.add_argument("--contact-root-fraction", type=float, default=0.5)
    parser.add_argument("--contact-approach-steps", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def wrap_pi(value: np.ndarray) -> np.ndarray:
    return (value + np.pi / 2.0) % np.pi - np.pi / 2.0


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def circular_mae(value: np.ndarray, period: float) -> float:
    if len(value) == 0:
        return float("nan")
    wrapped = (value + period / 2.0) % period - period / 2.0
    return float(np.mean(np.abs(wrapped)))


def fit_signed_offset(predicted: np.ndarray, target: np.ndarray) -> tuple[int, float]:
    """Fit only image handedness and one angular offset on calibration roots."""
    candidates: list[tuple[float, int, float]] = []
    for sign in (1, -1):
        residual = wrap_angle(target - sign * predicted)
        offset = float(np.angle(np.mean(np.exp(1j * residual))))
        error = circular_mae(wrap_angle(sign * predicted + offset - target), 2.0 * np.pi)
        candidates.append((error, sign, offset))
    error, sign, offset = min(candidates, key=lambda row: row[0])
    if not np.isfinite(error):
        raise RuntimeError("analytic pose calibration produced a non-finite error")
    return sign, offset


def angle_metrics(
    predicted: np.ndarray, target: np.ndarray, calibration: np.ndarray,
) -> dict[str, Any]:
    """Return raw modulo-pi and calibration-only directed metrics."""
    line_error = np.abs(wrap_pi(predicted - target))
    sign, offset = fit_signed_offset(predicted[calibration], target[calibration])
    directed_error = np.abs(wrap_angle(sign * predicted + offset - target))
    return {
        "calibration_sign": int(sign),
        "calibration_offset_rad": float(offset),
        "all": {
            "line_mae_rad": circular_mae(predicted - target, np.pi),
            "line_rmse_rad": float(np.sqrt(np.mean(line_error ** 2))),
            "directed_mae_rad": circular_mae(sign * predicted + offset - target, 2.0 * np.pi),
            "directed_rmse_rad": float(np.sqrt(np.mean(directed_error ** 2))),
            "directed_15deg": float(np.mean(directed_error <= np.deg2rad(15.0))),
            "directed_30deg": float(np.mean(directed_error <= np.deg2rad(30.0))),
        },
        "calibration": {
            "n": int(calibration.sum()),
            "line_mae_rad": circular_mae((predicted - target)[calibration], np.pi),
            "directed_mae_rad": circular_mae((sign * predicted + offset - target)[calibration], 2.0 * np.pi),
        },
        "test": {
            "n": int((~calibration).sum()),
            "line_mae_rad": circular_mae((predicted - target)[~calibration], np.pi),
            "directed_mae_rad": circular_mae((sign * predicted + offset - target)[~calibration], 2.0 * np.pi),
            "directed_rmse_rad": float(np.sqrt(np.mean(directed_error[~calibration] ** 2))),
            "directed_15deg": float(np.mean(directed_error[~calibration] <= np.deg2rad(15.0))),
            "directed_30deg": float(np.mean(directed_error[~calibration] <= np.deg2rad(30.0))),
        },
        "per_sample_directed_error": directed_error,
    }


@torch.inference_mode()
def extract_moments(images: np.ndarray, device: str, batch_size: int) -> dict[str, np.ndarray]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; analytic pose audit refuses CPU fallback")
    all_features: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch = torch.as_tensor(images[start:start + batch_size], dtype=torch.float32, device=device) / 255.0
        _agent, block, _target = MaskMomentGeometryEncoder._masks(batch)
        features = MaskMomentGeometryEncoder._moments(block)
        all_features.append(features.detach().cpu())
        all_masks.append(block.sum(dim=(1, 2, 3)).detach().cpu())
    feature = torch.cat(all_features, dim=0).numpy()
    mass = torch.cat(all_masks, dim=0).numpy()
    # [mass, mean_x, mean_y, mu20, mu11, mu02, mu30, mu21, mu12, mu03]
    mu20, mu11, mu02 = feature[:, 3], feature[:, 4], feature[:, 5]
    mu30, mu21, mu12, mu03 = feature[:, 6], feature[:, 7], feature[:, 8], feature[:, 9]
    covariance_angle = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    axis_x, axis_y = np.cos(covariance_angle), np.sin(covariance_angle)
    # Projection of the third central moment onto the principal axis.  Choosing
    # the direction with positive skew is a label-free directed-angle rule.
    projected_third = (
        mu30 * axis_x ** 3
        + 3.0 * mu21 * axis_x ** 2 * axis_y
        + 3.0 * mu12 * axis_x * axis_y ** 2
        + mu03 * axis_y ** 3
    )
    directed_angle = covariance_angle + np.where(projected_third < 0.0, np.pi, 0.0)
    anisotropy = np.sqrt((mu20 - mu02) ** 2 + (2.0 * mu11) ** 2) / np.maximum(mu20 + mu02, 1e-8)
    return {
        "raw_moments": feature,
        "covariance_angle": covariance_angle,
        "directed_angle": directed_angle,
        "projected_third": projected_third,
        "anisotropy": anisotropy,
        "block_mass": mass,
    }


def summarize_subset(
    name: str, mask: np.ndarray, metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: dict[str, Any] = {"n": int(mask.sum())}
    for representation, metric in metrics.items():
        errors = metric["per_sample_directed_error"][mask]
        line = metric["per_sample_line_error"][mask]
        rows[representation] = {
            "line_mae_rad": circular_mae(line, np.pi),
            "directed_mae_rad": circular_mae(errors, 2.0 * np.pi),
            "directed_15deg": float(np.mean(errors <= np.deg2rad(15.0))) if len(errors) else float("nan"),
            "directed_30deg": float(np.mean(errors <= np.deg2rad(30.0))) if len(errors) else float("nan"),
        }
    return rows


def polynomial_moment_features(features: dict[str, np.ndarray], device: str) -> torch.Tensor:
    """Build a small fixed-capacity, translation-invariant moment chart on CUDA."""
    base = np.column_stack([
        features["block_mass"],
        # Central moments are translation-invariant; centroid is omitted as a
        # nuisance coordinate for block orientation.
        features["raw_moments"][:, 3:10],
        features["anisotropy"],
        features["projected_third"],
    ]).astype(np.float32)
    return torch.as_tensor(base, dtype=torch.float32, device=device)


@torch.inference_mode()
def fit_calibrated_harmonic_ridge(
    features: dict[str, np.ndarray], target: np.ndarray, calibration: np.ndarray,
    device: str, ridge: float = 1.0e-2, artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Fit a frozen low-capacity RGB pose readout on calibration roots only."""
    x = polynomial_moment_features(features, device)
    calibration_tensor = torch.as_tensor(calibration, dtype=torch.bool, device=device)
    mean = x[calibration_tensor].mean(dim=0)
    scale = x[calibration_tensor].std(dim=0).clamp_min(1.0e-4)
    z = (x - mean) / scale
    pieces = [torch.ones((len(z), 1), dtype=torch.float32, device=device), z]
    for i in range(z.shape[1]):
        for j in range(i, z.shape[1]):
            pieces.append(z[:, i:i + 1] * z[:, j:j + 1])
    design = torch.cat(pieces, dim=1)
    target_tensor = torch.as_tensor(
        np.column_stack([np.cos(target), np.sin(target)]),
        dtype=torch.float32, device=device,
    )
    x_calibration = design[calibration_tensor]
    y_calibration = target_tensor[calibration_tensor]
    gram = x_calibration.T @ x_calibration
    regularizer = ridge * torch.eye(gram.shape[0], device=device, dtype=gram.dtype)
    weight = torch.linalg.solve(gram + regularizer, x_calibration.T @ y_calibration)
    if artifact_path is not None:
        torch.save(
            {
                "feature_mean": mean.detach().cpu(),
                "feature_scale": scale.detach().cpu(),
                "weight": weight.detach().cpu(),
                "ridge": ridge,
                "feature_definition": "mass, central moments 2-3, anisotropy, projected third; quadratic basis",
                "calibration_count": int(calibration.sum()),
            },
            artifact_path,
        )
    projected = design @ weight
    predicted = torch.atan2(projected[:, 1], projected[:, 0]).cpu().numpy()
    result = angle_metrics(predicted, target, calibration)
    result["capacity"] = {
        "input_dim": int(x.shape[1]),
        "quadratic_design_dim": int(design.shape[1]),
        "ridge": ridge,
    }
    if artifact_path is not None:
        result["calibration_artifact"] = str(artifact_path)
    result["per_sample_line_error"] = np.abs(wrap_pi(predicted - target))
    return result


@torch.inference_mode()
def apply_calibrated_harmonic_ridge(
    features: dict[str, np.ndarray], artifact_path: Path, device: str,
) -> np.ndarray:
    """Apply a frozen calibration artifact without refitting on the target set."""
    payload = torch.load(artifact_path, map_location=device, weights_only=False)
    x = polynomial_moment_features(features, device)
    z = (x - payload["feature_mean"].to(device)) / payload["feature_scale"].to(device)
    pieces = [torch.ones((len(z), 1), dtype=z.dtype, device=device), z]
    for i in range(z.shape[1]):
        for j in range(i, z.shape[1]):
            pieces.append(z[:, i:i + 1] * z[:, j:j + 1])
    projected = torch.cat(pieces, dim=1) @ payload["weight"].to(device)
    return torch.atan2(projected[:, 1], projected[:, 0]).cpu().numpy()


def main() -> None:
    args = parse_args()
    if args.roots < 32 or not 0.1 <= args.calibration_fraction <= 0.9:
        raise ValueError("roots must be >= 32 and calibration fraction must be in [0.1, 0.9]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass a working CUDA environment or do not run this audit")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"[purpose] RGB analytic pose audit; device={args.device}; roots={args.roots}")
    print("[protocol] simulator state is offline label only; calibration is root-disjoint from test")

    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        cases = balanced_source_cases(
            env, args.roots, args.root_rollout_steps, args.seed,
            args.contact_root_fraction, args.contact_approach_steps,
        )
        if len(cases) != args.roots:
            raise RuntimeError(f"collected {len(cases)} roots, expected {args.roots}")
        images = np.asarray([resize_image(case[2], args.image_size) for case in cases], dtype=np.uint8)
        goal_images = np.asarray([resize_image(case[4], args.image_size) for case in cases], dtype=np.uint8)
        states = np.asarray([case[0] for case in cases], dtype=np.float32)
        goals = np.asarray([case[1] for case in cases], dtype=np.float32)
        contact = np.asarray([case[6] for case in cases], dtype=bool)
    finally:
        env.close()

    features = extract_moments(images, args.device, args.batch_size)
    target = states[:, 4]
    permutation = np.random.default_rng(args.seed + 1).permutation(len(target))
    calibration = np.zeros(len(target), dtype=bool)
    calibration[permutation[:int(round(args.calibration_fraction * len(target)))]] = True

    metrics: dict[str, dict[str, Any]] = {}
    for representation in ("covariance_angle", "directed_angle"):
        result = angle_metrics(features[representation], target, calibration)
        result["per_sample_line_error"] = np.abs(wrap_pi(features[representation] - target))
        metrics[representation] = result
    metrics["calibrated_harmonic_ridge"] = fit_calibrated_harmonic_ridge(
        features, target, calibration, args.device,
        artifact_path=output / "calibrated_harmonic_ridge.pt",
    )
    goal_features = extract_moments(goal_images, args.device, args.batch_size)
    goal_metric = fit_calibrated_harmonic_ridge(
        goal_features, goals[:, 4], calibration, args.device,
        artifact_path=output / "calibrated_goal_harmonic_ridge.pt",
    )

    test_mask = ~calibration
    anisotropy_q = np.quantile(features["anisotropy"], [0.25, 0.5, 0.75])
    mass_q = np.quantile(features["block_mass"], [0.25, 0.5, 0.75])
    strata: dict[str, np.ndarray] = {
        "all_test": test_mask,
        "free_test": test_mask & ~contact,
        "contact_test": test_mask & contact,
        "low_anisotropy_test": test_mask & (features["anisotropy"] <= anisotropy_q[0]),
        "high_anisotropy_test": test_mask & (features["anisotropy"] > anisotropy_q[2]),
        "low_block_mass_test": test_mask & (features["block_mass"] <= mass_q[0]),
        "high_block_mass_test": test_mask & (features["block_mass"] > mass_q[2]),
    }
    stratum_report = {
        name: summarize_subset(name, mask, metrics) for name, mask in strata.items()
    }
    report: dict[str, Any] = {
        "protocol": {
            "seed": args.seed,
            "roots": len(cases),
            "root_disjoint_calibration_test": True,
            "calibration_fraction": float(args.calibration_fraction),
            "image_size": args.image_size,
            "device": args.device,
            "cuda_device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            "runtime_state_access": False,
            "state_role": "offline reference angle and contact strata only",
            "mask_chart": "MaskMomentGeometryEncoder RGB thresholds",
        },
        "calibration": {
            "n": int(calibration.sum()),
            "test_n": int(test_mask.sum()),
            "anisotropy_quantiles": anisotropy_q.tolist(),
            "block_mass_quantiles": mass_q.tolist(),
        },
        "global": {
            name: {
                key: value for key, value in metric.items()
                if key != "per_sample_directed_error" and key != "per_sample_line_error"
            }
            for name, metric in metrics.items()
        },
        "goal_image_transfer": {
            key: value for key, value in goal_metric.items()
            if key != "per_sample_directed_error" and key != "per_sample_line_error"
        },
        "strata": stratum_report,
        "raw_summary": {
            "valid_block_mask_fraction": float(np.mean(features["block_mass"] > 1.0)),
            "anisotropy_median": float(np.median(features["anisotropy"])),
            "block_mass_median": float(np.median(features["block_mass"])),
        },
        "claim_boundary": "Frozen RGB pose-interface diagnosis only; not action-effect identifiability or online-planning evidence.",
    }
    (output / "rgb_analytic_pose_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# RGB analytic pose audit", "",
        f"Device: `{args.device}` ({report['protocol']['cuda_device']})", "",
        "Calibration is root-disjoint from test; simulator state is offline reference only.", "",
        "| representation | test line MAE (rad) | test directed MAE (rad) | test <=15deg | test <=30deg |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metric in metrics.items():
        test = metric["test"]
        lines.append(f"| {name} | {test['line_mae_rad']:.5f} | {test['directed_mae_rad']:.5f} | {test['directed_15deg']:.4f} | {test['directed_30deg']:.4f} |")
    lines += ["", "## Root-disjoint strata", "", "| stratum | n | representation | line MAE | directed MAE | <=15deg |", "|---|---:|---|---:|---:|---:|"]
    for stratum, values in stratum_report.items():
        for representation in ("covariance_angle", "directed_angle", "calibrated_harmonic_ridge"):
            row = values[representation]
            lines.append(f"| {stratum} | {values['n']} | {representation} | {row['line_mae_rad']:.5f} | {row['directed_mae_rad']:.5f} | {row['directed_15deg']:.4f} |")
    lines += ["", report["claim_boundary"], ""]
    (output / "rgb_analytic_pose_audit.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "device": args.device, "global_test": {name: metric["test"] for name, metric in metrics.items()}, "strata": stratum_report}, indent=2))


if __name__ == "__main__":
    main()
