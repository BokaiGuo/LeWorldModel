"""Whitening and orthogonal-reparameterization controls for frozen LeWM latents.

The experiment tests two claims about the covariance--predictor diagnostic:

1. simultaneous orthogonal reparameterization should preserve normalized
   commutator metrics and held-out linear residuals;
2. whitening the global latent covariance should collapse the global
   commutator, demonstrating why low global commutation is not by itself a
   sufficient learning objective.

This is a frozen one-step operator diagnostic.  It is not retraining, causal
intervention, rollout, online planning, or task-performance evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _fit_operator(source: np.ndarray, target: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ source_centered / max(source.shape[0] - 1, 1)
    cross_covariance = source_centered.T @ target_centered / max(source.shape[0] - 1, 1)
    scale = max(float(np.trace(covariance)) / source.shape[1], 1e-12)
    row_operator = np.linalg.solve(
        covariance + float(ridge) * scale * np.eye(source.shape[1]), cross_covariance
    )
    return covariance, row_operator.T


def _commutator(covariance: np.ndarray, operator: np.ndarray) -> dict[str, float]:
    commutator = covariance @ operator - operator @ covariance
    denominator_fro = max(float(np.linalg.norm(covariance, "fro") * np.linalg.norm(operator, "fro")), 1e-12)
    denominator_spec = max(float(np.linalg.norm(covariance, 2) * np.linalg.norm(operator, 2)), 1e-12)
    return {
        "normalized_frobenius": float(np.linalg.norm(commutator, "fro") / denominator_fro),
        "normalized_spectral": float(np.linalg.norm(commutator, 2) / denominator_spec),
        "absolute_frobenius": float(np.linalg.norm(commutator, "fro")),
    }


def _stable_rank(covariance: np.ndarray) -> float:
    return float(np.trace(covariance) ** 2 / max(np.square(covariance).sum(), 1e-12))


def _orthogonal(seed: int, dimension: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q @ np.diag(signs)


def _whitener(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(covariance)
    floor = max(float(values.max()) * 1e-8, 1e-10)
    clipped = np.maximum(values, floor)
    whitening = (vectors * (1.0 / np.sqrt(clipped))) @ vectors.T
    coloring = (vectors * np.sqrt(clipped)) @ vectors.T
    return whitening, coloring


def _evaluate_record(
    source: np.ndarray,
    target: np.ndarray,
    seed: int,
    ridge: float,
) -> dict[str, Any]:
    order = np.random.default_rng(seed + 7001).permutation(source.shape[0])
    cut = max(2, source.shape[0] // 2)
    train = order[:cut]
    test = order[cut:]
    train_covariance, train_operator = _fit_operator(source[train], target[train], ridge)

    source_test_centered = source[test] - source[train].mean(axis=0)
    target_test_centered = target[test] - target[train].mean(axis=0)
    prediction = source_test_centered @ train_operator.T
    residual_mse = float(np.square(prediction - target_test_centered).mean())

    q = _orthogonal(seed + 31, source.shape[1])
    orth_covariance = q @ train_covariance @ q.T
    orth_operator = q @ train_operator @ q.T
    original_commutator = _commutator(train_covariance, train_operator)
    orth_commutator = _commutator(orth_covariance, orth_operator)

    whitening, coloring = _whitener(train_covariance)
    white_covariance = whitening @ train_covariance @ whitening.T
    white_operator = whitening @ train_operator @ coloring
    white_commutator = _commutator(white_covariance, white_operator)

    return {
        "seed": int(seed),
        "num_samples": int(source.shape[0]),
        "latent_dim": int(source.shape[1]),
        "ridge": float(ridge),
        "train_samples": int(train.size),
        "test_samples": int(test.size),
        "original": {
            "commutator": original_commutator,
            "stable_rank": _stable_rank(train_covariance),
            "residual_mse": residual_mse,
        },
        "orthogonal_reparameterization": {
            "commutator": orth_commutator,
            "normalized_frobenius_abs_difference": abs(
                orth_commutator["normalized_frobenius"] - original_commutator["normalized_frobenius"]
            ),
            "normalized_spectral_abs_difference": abs(
                orth_commutator["normalized_spectral"] - original_commutator["normalized_spectral"]
            ),
            "expected_invariant": True,
        },
        "global_whitening": {
            "commutator": white_commutator,
            "whitened_covariance_deviation_from_identity": float(
                np.linalg.norm(white_covariance - np.eye(source.shape[1]), "fro")
            ),
            "stable_rank": _stable_rank(white_covariance),
            "expected_global_commutator_collapse": True,
        },
    }


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def run_report(
    source: Path,
    output: Path,
    sample_caps: list[int],
    seeds: list[int],
    ridge: float,
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
            record = _evaluate_record(latents[rows], target[rows], seed, ridge)
            records.append(record)
            print(
                f"seed={seed} rows={cap} original_C={record['original']['commutator']['normalized_frobenius']:.6g} "
                f"orth_delta={record['orthogonal_reparameterization']['normalized_frobenius_abs_difference']:.3g} "
                f"white_C={record['global_whitening']['commutator']['normalized_frobenius']:.3g}",
                flush=True,
            )
    report = {
        "artifact": "frozen_whitening_invariance_control",
        "status": "frozen_offline_boundary_evidence",
        "source": str(source),
        "protocol": {
            "sample_caps": [int(value) for value in sample_caps],
            "seeds": [int(value) for value in seeds],
            "operator_target": "latents + predicted_dynamics",
            "fit": "centered ridge linear operator on a disjoint train half; metrics include held-out residual",
            "orthogonal_control": "random Q with Q^T Q = I applied jointly to covariance and operator",
            "whitening_control": "global train covariance inverse square root applied jointly to latent coordinates",
            "rollout_and_planning_available": False,
        },
        "records": records,
        "claim_boundary": [
            "Orthogonal reparameterization should preserve the normalized commutator diagnostic up to numerical error.",
            "Global whitening is an explicit identifiability control: it can make the global commutator nearly zero without demonstrating useful dynamics.",
            "This is frozen one-step operator evidence and does not establish retraining gains, causal controllability, rollout, or planning.",
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
        "# Whitening and orthogonal invariance control on frozen LeWM latents",
        "",
        "The orthogonal control checks coordinate invariance; the whitening control checks whether global commutator minimization can be made trivial by flattening the covariance.",
        "",
        "| seed | rows | original normalized commutator | orthogonal absolute difference | whitened normalized commutator | original stable rank | whitened stable rank | covariance identity error |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in report["records"]:
        lines.append(
            f"| {record['seed']} | {record['num_samples']} | "
            f"{record['original']['commutator']['normalized_frobenius']:.6g} | "
            f"{record['orthogonal_reparameterization']['normalized_frobenius_abs_difference']:.3g} | "
            f"{record['global_whitening']['commutator']['normalized_frobenius']:.3g} | "
            f"{record['original']['stable_rank']:.3f} | {record['global_whitening']['stable_rank']:.3f} | "
            f"{record['global_whitening']['whitened_covariance_deviation_from_identity']:.3g} |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-caps", type=int, nargs="+", default=[100000, 200000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[3072, 3073, 3074])
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()
    run_report(args.source, args.output, args.sample_caps, args.seeds, args.ridge)
    print(args.output)
    print(args.output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
