"""Non-normality and transient-amplification audit for frozen LeWM operators.

The audit fits a centered ridge operator from frozen ``z`` to the exported
one-step target ``z + predicted_dynamics``.  It reports covariance commutation,
Henrici-style non-normality, eigenvalue radius, singular-value growth, and
finite-horizon operator gains on disjoint held-out splits.

This is frozen one-step operator evidence.  It is not end-to-end retraining,
causal control, rollout, online planning, or task-success evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from experiments.empirical_spectral_audit.run_whitening_invariance_frozen import (
        _commutator,
        _fit_operator,
        _finite,
    )
except ModuleNotFoundError:  # direct invocation from the script directory
    from run_whitening_invariance_frozen import (  # type: ignore
        _commutator,
        _fit_operator,
        _finite,
    )


def _operator_diagnostics(
    covariance: np.ndarray,
    operator: np.ndarray,
    horizons: list[int],
) -> dict[str, Any]:
    operator_norm = max(float(np.linalg.norm(operator, "fro")), 1e-12)
    nonnormal = operator.T @ operator - operator @ operator.T
    eigenvalues = np.linalg.eigvals(operator)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    singular_values = np.linalg.svd(operator, compute_uv=False)
    max_singular = float(singular_values[0])
    powers: dict[str, dict[str, float]] = {}
    for horizon in horizons:
        power = np.linalg.matrix_power(operator, int(horizon))
        gain = float(np.linalg.norm(power, 2))
        denominator = max(spectral_radius ** int(horizon), 1e-12)
        powers[str(horizon)] = {
            "spectral_norm": gain,
            "spectral_radius_power": float(spectral_radius ** int(horizon)),
            "gain_over_radius_power": gain / denominator,
        }
    return {
        "commutator": _commutator(covariance, operator),
        "henrici_nonnormality": float(np.linalg.norm(nonnormal, "fro") / (operator_norm**2)),
        "spectral_radius": spectral_radius,
        "max_singular_value": max_singular,
        "singular_to_radius_ratio": max_singular / max(spectral_radius, 1e-12),
        "operator_powers": powers,
    }


def _one_record(
    source: np.ndarray,
    target: np.ndarray,
    seed: int,
    ridge: float,
    horizons: list[int],
) -> dict[str, Any]:
    order = np.random.default_rng(seed + 9101).permutation(source.shape[0])
    cut = max(2, source.shape[0] // 2)
    train, test = order[:cut], order[cut:]
    covariance, operator = _fit_operator(source[train], target[train], ridge)
    source_mean = source[train].mean(axis=0)
    target_mean = target[train].mean(axis=0)
    prediction = (source[test] - source_mean) @ operator.T + target_mean
    residual_mse = float(np.square(prediction - target[test]).mean())
    return {
        "seed": int(seed),
        "num_samples": int(source.shape[0]),
        "train_samples": int(train.size),
        "test_samples": int(test.size),
        "ridge": float(ridge),
        "heldout_residual_mse": residual_mse,
        "operator": _operator_diagnostics(covariance, operator, horizons),
    }


def run_report(
    source: Path,
    output: Path,
    sample_caps: list[int],
    seeds: list[int],
    ridge: float,
    horizons: list[int],
) -> dict[str, Any]:
    with np.load(source, allow_pickle=False) as archive:
        required = {"latents", "predicted_dynamics"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"missing arrays: {sorted(missing)}")
        latents = np.asarray(archive["latents"], dtype=np.float64)
        predicted = np.asarray(archive["predicted_dynamics"], dtype=np.float64)
    if latents.shape != predicted.shape:
        raise ValueError("latents and predicted_dynamics must have matching shapes")
    target = latents + predicted
    records = []
    for seed in seeds:
        for cap in sample_caps:
            if cap <= 1 or cap > latents.shape[0]:
                raise ValueError(f"invalid sample cap {cap}")
            rows = np.sort(np.random.default_rng(seed).choice(latents.shape[0], size=cap, replace=False))
            record = _one_record(latents[rows], target[rows], seed, ridge, horizons)
            records.append(record)
            operator = record["operator"]
            print(
                f"seed={seed} rows={cap} C={operator['commutator']['normalized_frobenius']:.6g} "
                f"nonnormal={operator['henrici_nonnormality']:.6g} "
                f"rho={operator['spectral_radius']:.6g} sigma_max={operator['max_singular_value']:.6g}",
                flush=True,
            )
    report = {
        "artifact": "frozen_nonnormality_transient_control",
        "status": "frozen_offline_boundary_evidence",
        "source": str(source),
        "protocol": {
            "sample_caps": [int(value) for value in sample_caps],
            "seeds": [int(value) for value in seeds],
            "horizons": [int(value) for value in horizons],
            "target": "latents + predicted_dynamics",
            "fit": "centered ridge operator on a disjoint train half",
            "heldout_metric": "one-step target residual MSE",
            "rollout_and_planning_available": False,
        },
        "records": records,
        "claim_boundary": [
            "The operator diagnostics distinguish eigenvalue radius from singular-value and transient amplification behavior.",
            "A gain-over-radius-power ratio above one is evidence of finite-horizon amplification in the fitted linear summary, not a causal rollout claim.",
            "This is frozen one-step operator evidence and does not establish retraining gains, online control, planning, or task success.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    horizons = report["protocol"]["horizons"]
    lines = [
        "# Non-normality and transient amplification on frozen LeWM latents",
        "",
        "The fitted operator is evaluated on a disjoint held-out half. The table separates covariance commutation from non-normality and finite-horizon gain.",
        "",
        "| seed | rows | normalized C | Henrici non-normality | spectral radius | max singular value | sigma/radius | held-out MSE | "
        + " | ".join(f"gain/radius^{h}" for h in horizons)
        + " |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        + " | ".join("---:" for _ in horizons)
        + " |",
    ]
    for record in report["records"]:
        operator = record["operator"]
        gains = " | ".join(
            f"{operator['operator_powers'][str(h)]['gain_over_radius_power']:.4f}" for h in horizons
        )
        lines.append(
            f"| {record['seed']} | {record['num_samples']} | "
            f"{operator['commutator']['normalized_frobenius']:.6g} | "
            f"{operator['henrici_nonnormality']:.6g} | {operator['spectral_radius']:.6g} | "
            f"{operator['max_singular_value']:.6g} | {operator['singular_to_radius_ratio']:.4f} | "
            f"{record['heldout_residual_mse']:.6g} | {gains} |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-caps", type=int, nargs="+", default=[100000, 200000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[3072, 3073, 3074])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()
    run_report(args.source, args.output, args.sample_caps, args.seeds, args.ridge, args.horizons)
    print(args.output)
    print(args.output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
