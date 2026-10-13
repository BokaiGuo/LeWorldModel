"""Measure horizon-conditioned operator geometry in the synthetic vector world.

For each trained condition, a linear operator is cross-fitted from the initial
latent to the encoded true state at horizons 1/2/4/8.  The report separates
the horizon-dependent operator residual from covariance--operator commutator
and top-subspace overlap.  This is synthetic mechanism evidence; it is not a
LeWM, PushT, planning, or causal-control result.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from experiments.empirical_spectral_audit.run_alignment_equivariance_joint_synthetic import (
        EquivarianceBenchmarkConfig,
        _fit_joint,
        _seed_everything,
    )
except ModuleNotFoundError:  # direct script invocation
    from run_alignment_equivariance_joint_synthetic import EquivarianceBenchmarkConfig, _fit_joint, _seed_everything


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def _split_fit(
    source: np.ndarray, target: np.ndarray, ridge: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must be matching two-dimensional arrays")
    rng = np.random.default_rng(seed)
    indices = np.arange(source.shape[0])
    rng.shuffle(indices)
    midpoint = max(1, source.shape[0] // 2)
    train, test = indices[:midpoint], indices[midpoint:]
    source_train = source[train]
    target_train = target[train]
    source_mean = source_train.mean(axis=0)
    target_mean = target_train.mean(axis=0)
    source_centered = source_train - source_mean
    target_centered = target_train - target_mean
    covariance = source_centered.T @ source_centered / max(source_train.shape[0] - 1, 1)
    cross = source_centered.T @ target_centered / max(source_train.shape[0] - 1, 1)
    scale = max(float(np.trace(covariance)) / source.shape[1], 1e-6)
    row_operator = np.linalg.solve(covariance + ridge * scale * np.eye(source.shape[1]), cross)
    source_test = source[test]
    target_test = target[test]
    prediction = (source_test - source_mean) @ row_operator + target_mean
    test_covariance = np.cov(source_test, rowvar=False, ddof=1)
    return prediction - target_test, test_covariance, row_operator.T


def _operator_metrics(source: np.ndarray, target: np.ndarray, ridge: float, seed: int, top_k: int) -> dict[str, float]:
    residual, test_covariance, column_operator = _split_fit(source, target, ridge=ridge, seed=seed)
    commutator = test_covariance @ column_operator - column_operator @ test_covariance
    denominator = np.square(test_covariance).sum() * np.square(column_operator).sum()
    cov_values, cov_vectors = np.linalg.eigh(test_covariance)
    cov_basis = cov_vectors[:, np.argsort(cov_values)[::-1][:top_k]]
    _, _, vh = np.linalg.svd(column_operator, full_matrices=False)
    op_basis = vh[:top_k].T
    singular_values = np.linalg.svd(cov_basis.T @ op_basis, compute_uv=False)
    min_singular = float(np.clip(singular_values.min(), 0.0, 1.0))
    overlap = float(np.square(cov_basis.T @ op_basis).sum() / top_k)
    return {
        "commutator_sq": float(np.square(commutator).sum() / max(denominator, 1e-12)),
        "heldout_residual_mse": float(np.square(residual).mean()),
        "covariance_operator_overlap": overlap,
        "max_principal_angle_degrees": float(np.degrees(np.arccos(min_singular))),
        "stable_rank": float(np.trace(test_covariance) ** 2 / max(np.square(test_covariance).sum(), 1e-12)),
    }


def _evaluate_condition(mode: str, predictor_weight: float, alignment_weight: float, seed: int, config: EquivarianceBenchmarkConfig, horizons: tuple[int, ...]) -> dict[str, Any]:
    model, _, _, _, rollout, _, best_epoch, best_val = _fit_joint(
        mode, predictor_weight, alignment_weight, seed, config
    )
    with torch.no_grad():
        frames = rollout["states"]
        encoded = model.encode(frames.reshape(-1, frames.size(-1))).reshape(frames.size(0), frames.size(1), -1)
    source = encoded[:, 0].cpu().numpy().astype(np.float64)
    horizon_records = {}
    for horizon in horizons:
        target = encoded[:, horizon].cpu().numpy().astype(np.float64)
        horizon_records[str(horizon)] = _operator_metrics(
            source, target, ridge=1e-3, seed=seed + 80_000 + horizon, top_k=min(4, source.shape[1])
        )
    return {
        "mode": mode,
        "predictor_weight": float(predictor_weight),
        "alignment_weight": float(alignment_weight),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_mse": float(best_val),
        "horizon_metrics": horizon_records,
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1 if arr.size > 1 else 0))}


def _summary(records: list[dict[str, Any]], horizons: tuple[int, ...]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = f"{record['mode']}:predictor={record['predictor_weight']}:alignment={record['alignment_weight']}"
        groups.setdefault(key, []).append(record)
    result = {}
    for key, rows in groups.items():
        result[key] = {
            "n_seeds": len(rows),
            "mode": rows[0]["mode"],
            "predictor_weight": rows[0]["predictor_weight"],
            "alignment_weight": rows[0]["alignment_weight"],
            "horizons": {
                str(horizon): {
                    metric: _mean_std([float(row["horizon_metrics"][str(horizon)][metric]) for row in rows])
                    for metric in rows[0]["horizon_metrics"][str(horizon)]
                }
                for horizon in horizons
            },
        }
    return result


def _rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    if x.size != y.size or x.size < 3:
        raise ValueError("correlation requires at least three paired values")
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    rx, ry = _rank_values(x), _rank_values(y)
    spearman = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else 0.0
    return {"n": int(x.size), "pearson": pearson, "spearman": spearman}


def run_report(seeds: list[int], horizons: tuple[int, ...] = (1, 2, 4, 8), epochs: int = 120) -> dict[str, Any]:
    config = EquivarianceBenchmarkConfig(epochs=epochs)
    conditions = [
        ("none", 0.0, 0.0),
        ("predictor", 1.0, 0.0),
        ("both", 1.0, 0.003),
    ]
    records = []
    started = time.time()
    for mode, predictor_weight, alignment_weight in conditions:
        for seed in seeds:
            print(f"[horizon-geometry] mode={mode} alignment={alignment_weight} seed={seed}")
            records.append(_evaluate_condition(mode, predictor_weight, alignment_weight, seed, config, horizons))
    report = {
        "protocol": {
            "transition": "s_next = s + a",
            "conditions": conditions,
            "seeds": [int(seed) for seed in seeds],
            "horizons": [int(horizon) for horizon in horizons],
            "operator_fit": "ridge linear map from encoded initial state to encoded true h-step state; disjoint half cross-fit",
            "top_k": min(4, config.latent_dim),
            "synthetic_vector_world_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "records": records,
        "summary": _summary(records, horizons),
        "h8_cross_cell_correlations": {
            "commutator_vs_heldout_residual_mse": _correlation(
                np.asarray([row["horizon_metrics"]["8"]["commutator_sq"] for row in records], dtype=np.float64),
                np.asarray([row["horizon_metrics"]["8"]["heldout_residual_mse"] for row in records], dtype=np.float64),
            ),
            "principal_angle_vs_heldout_residual_mse": _correlation(
                np.asarray([row["horizon_metrics"]["8"]["max_principal_angle_degrees"] for row in records], dtype=np.float64),
                np.asarray([row["horizon_metrics"]["8"]["heldout_residual_mse"] for row in records], dtype=np.float64),
            ),
        },
        "claim_boundary": [
            "horizon-conditioned linear summaries are mechanism diagnostics, not the nonlinear predictor itself",
            "no LeWM, PushT, online planning, or causal-control evidence",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    return report


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Horizon-conditioned operator geometry in the synthetic vector world",
        "",
        "Operators map the encoded initial state to the encoded true state at each horizon. Values are mean ± standard deviation over seeds.",
        "",
        "| condition | h | commutator | held-out residual | covariance/operator overlap | max principal angle | stable rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in report["summary"].items():
        for horizon, metrics in values["horizons"].items():
            fmt = lambda name: f"{metrics[name]['mean']:.6g} ± {metrics[name]['std']:.2g}"
            lines.append(
                f"| {key} | {horizon} | {fmt('commutator_sq')} | {fmt('heldout_residual_mse')} | "
                f"{fmt('covariance_operator_overlap')} | {fmt('max_principal_angle_degrees')} | {fmt('stable_rank')} |"
            )
    lines.extend(["", "## Boundary", "", "This is synthetic mechanism evidence only."])
    correlations = report["h8_cross_cell_correlations"]
    lines.extend(
        [
            "",
            "## Descriptive h=8 correlations",
            "",
            f"Across {correlations['commutator_vs_heldout_residual_mse']['n']} condition-seed cells, commutator vs residual Pearson/Spearman = "
            f"{correlations['commutator_vs_heldout_residual_mse']['pearson']:.4f}/{correlations['commutator_vs_heldout_residual_mse']['spearman']:.4f}; "
            f"principal-angle vs residual = {correlations['principal_angle_vs_heldout_residual_mse']['pearson']:.4f}/"
            f"{correlations['principal_angle_vs_heldout_residual_mse']['spearman']:.4f}. These are descriptive cross-cell associations, not causal estimates.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    report = run_report(args.seeds, epochs=args.epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
