"""Residual-spectrum audit for external frozen LeWM temporal vectors.

The input vectors are produced by the external public PushT temporal audit.
For each action mode and horizon, this script decomposes the squared
prediction residual into the eigenbasis of the encoded future-target
covariance.  A matched random orthonormal-basis null tests whether the band
profile is more structured than an arbitrary coordinate system.

This is a post-hoc frozen temporal diagnostic.  It does not turn the public
validation trajectories into online control or planning evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


BANDS = ((0, 8), (8, 32), (32, 128), (128, 192))


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def _band_key(start: int, end: int) -> str:
    return f"{start}:{end}"


def _orthonormal_basis(dim: int, seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(dim, dim)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs[None, :]


def _energy_profile(residual: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if residual.ndim != 2 or basis.shape != (residual.shape[1], residual.shape[1]):
        raise ValueError("residual and basis dimensions do not match")
    coordinates = residual @ basis
    energy = np.mean(np.square(coordinates), axis=0)
    shares = energy / max(float(energy.sum()), 1e-12)
    return energy, shares


def _spectral_profile(
    target: np.ndarray,
    prediction: np.ndarray,
    random_trials: int,
    seed: int,
) -> dict[str, Any]:
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[0] < 3:
        raise ValueError("target and prediction must be aligned matrices with at least 3 rows")
    centered_target = target - target.mean(axis=0, keepdims=True)
    covariance = centered_target.T @ centered_target / max(target.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    basis = np.asarray(eigenvectors[:, order], dtype=np.float64)
    effective_rank = int(np.sum(eigenvalues > max(float(eigenvalues[0]), 1e-12) * 1e-8))
    residual = np.asarray(prediction - target, dtype=np.float64)
    energy, shares = _energy_profile(residual, basis)
    bands = {
        _band_key(start, end): {
            "energy": float(energy[start:end].sum()),
            "share": float(shares[start:end].sum()),
        }
        for start, end in BANDS
        if end <= target.shape[1]
    }

    null_top_shares = []
    for trial in range(int(random_trials)):
        random_basis = _orthonormal_basis(target.shape[1], seed + trial)
        _, random_shares = _energy_profile(residual, random_basis)
        null_top_shares.append(float(random_shares[: BANDS[0][1]].sum()))
    null_top_shares = np.asarray(null_top_shares, dtype=np.float64)
    top_share = bands[_band_key(*BANDS[0])]["share"]
    return {
        "rows": int(target.shape[0]),
        "latent_dim": int(target.shape[1]),
        "residual_mse": float(np.square(residual).mean()),
        "target_covariance_trace": float(eigenvalues.sum()),
        "target_covariance_effective_rank": effective_rank,
        "target_covariance_top8_share": float(eigenvalues[:8].sum() / max(eigenvalues.sum(), 1e-12)),
        "bands": bands,
        "random_basis_null": {
            "trials": int(random_trials),
            "top8_share_mean": float(null_top_shares.mean()),
            "top8_share_ci95": [
                float(np.quantile(null_top_shares, 0.025)),
                float(np.quantile(null_top_shares, 0.975)),
            ],
            "observed_minus_null_mean": float(top_share - null_top_shares.mean()),
            "p_null_top8_ge_observed": float(
                (1.0 + np.sum(null_top_shares >= top_share)) / (null_top_shares.size + 1.0)
            ),
        },
    }


def run_report(
    vector_paths: list[Path],
    output: Path,
    horizons: list[int],
    random_trials: int,
    seed: int,
) -> dict[str, Any]:
    if not vector_paths:
        raise ValueError("at least one vector archive is required")
    arms: dict[str, Any] = {}
    for vector_path in vector_paths:
        arm = vector_path.parent.name
        if arm == "external_public_pusht_temporal_20261003":
            arm = "public"
        with np.load(vector_path, allow_pickle=False) as archive:
            horizon_report: dict[str, Any] = {}
            for horizon in horizons:
                for mode in ("observed", "zero"):
                    target_key = f"{mode}/h{horizon}/target"
                    prediction_key = f"{mode}/h{horizon}/prediction"
                    if target_key not in archive or prediction_key not in archive:
                        raise ValueError(f"missing {target_key} or {prediction_key} in {vector_path}")
                    horizon_report[f"{mode}/h{horizon}"] = _spectral_profile(
                        np.asarray(archive[target_key], dtype=np.float64),
                        np.asarray(archive[prediction_key], dtype=np.float64),
                        random_trials=random_trials,
                        seed=seed + 1000 * len(arms) + horizon,
                    )
        arms[arm] = {
            "source_vectors": str(vector_path),
            "profiles": horizon_report,
        }
    report = {
        "artifact": "external_lewm_residual_spectrum_audit",
        "status": "frozen_external_residual_spectrum",
        "protocol": {
            "horizons": horizons,
            "bands": [_band_key(*band) for band in BANDS],
            "covariance_basis": "eigenbasis of centered encoded future-target covariance",
            "residual": "predicted future embedding minus encoded future embedding",
            "random_null": "independent random orthonormal basis with matched dimension",
            "random_trials": int(random_trials),
            "seed": int(seed),
        },
        "arms": arms,
        "claim_boundary": [
            "This is a post-hoc frozen temporal residual-spectrum diagnostic.",
            "The random-basis null tests coordinate-free band concentration, not causality.",
            "The result does not establish online planning, control success, or a training improvement.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# External frozen LeWM residual spectrum",
        "",
        "Residuals are decomposed in the eigenbasis of the encoded future-target covariance. The random-basis control preserves dimension and residual vectors while removing covariance-relative orientation.",
        "",
        "| arm | mode | h | residual MSE | top 8 share | random-null top 8 | observed-null gap | null p | 8:32 share | 32:128 share | 128:192 share |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, details in report["arms"].items():
        for key, profile in details["profiles"].items():
            mode, horizon = key.split("/h")
            null = profile["random_basis_null"]
            bands = profile["bands"]
            lines.append(
                f"| {arm} | {mode} | {horizon} | {profile['residual_mse']:.6g} | "
                f"{bands['0:8']['share']:.4f} | {null['top8_share_mean']:.4f} | "
                f"{null['observed_minus_null_mean']:.4f} | {null['p_null_top8_ge_observed']:.4f} | "
                f"{bands['8:32']['share']:.4f} | {bands['32:128']['share']:.4f} | "
                f"{bands['128:192']['share']:.4f} |"
            )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--random-trials", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    report = run_report(args.vectors, args.output, args.horizons, args.random_trials, args.seed)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
