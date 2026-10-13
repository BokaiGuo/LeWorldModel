"""Bootstrap held-out residual differences for local frozen LeWM operators.

This audit reuses the cross-fit protocol from ``run_local_operator_geometry_frozen``.
For each action-bin partition, pooled and local ridge operators are fitted on
one half of each group and evaluated on the other half.  The bootstrap resamples
held-out rows within each group and uses the same resampled rows for pooled and
local predictions, producing a paired uncertainty interval for the residual
difference.

The interval is conditional on the fitted frozen operators.  It is therefore
stronger than a point estimate of held-out residual reduction, but it is not a
full uncertainty interval over operator fitting, retraining, trajectories, or
online planning.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from experiments.empirical_spectral_audit.run_local_operator_geometry_frozen import (
        _action_labels,
        _fit_operator,
        _load_arrays,
        _split_indices,
    )
except ModuleNotFoundError:  # direct script invocation
    from run_local_operator_geometry_frozen import _action_labels, _fit_operator, _load_arrays, _split_indices


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


def _predict_squared_errors(
    fit: dict[str, np.ndarray], source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    source_centered = source - fit["source_mean"]
    prediction = source_centered @ fit["row_operator"] + fit["target_mean"]
    return np.square(prediction - target).mean(axis=1)


def _bootstrap_paired(
    pooled_errors: list[np.ndarray],
    local_errors: list[np.ndarray],
    bootstrap_repeats: int,
    seed: int,
    group_weights: np.ndarray,
) -> dict[str, Any]:
    if len(pooled_errors) != len(local_errors) or len(pooled_errors) != len(group_weights):
        raise ValueError("pooled/local errors and weights must have matching group counts")
    rng = np.random.default_rng(seed)
    pooled_values = np.empty(int(bootstrap_repeats), dtype=np.float64)
    local_values = np.empty(int(bootstrap_repeats), dtype=np.float64)
    for repeat in range(int(bootstrap_repeats)):
        pooled_group_means = []
        local_group_means = []
        for pooled, local in zip(pooled_errors, local_errors):
            indices = rng.integers(0, pooled.size, size=pooled.size)
            pooled_group_means.append(float(pooled[indices].mean()))
            local_group_means.append(float(local[indices].mean()))
        pooled_values[repeat] = float(np.dot(group_weights, pooled_group_means))
        local_values[repeat] = float(np.dot(group_weights, local_group_means))
    differences = local_values - pooled_values
    pooled_mean = float(np.dot(group_weights, [values.mean() for values in pooled_errors]))
    local_mean = float(np.dot(group_weights, [values.mean() for values in local_errors]))
    difference = local_mean - pooled_mean
    reduction = 100.0 * (pooled_mean - local_mean) / max(pooled_mean, 1e-12)
    return {
        "bootstrap_repeats": int(bootstrap_repeats),
        "seed": int(seed),
        "pooled_residual_mse": pooled_mean,
        "local_residual_mse": local_mean,
        "local_minus_pooled": difference,
        "relative_reduction_percent": reduction,
        "pooled_ci95": [float(np.quantile(pooled_values, 0.025)), float(np.quantile(pooled_values, 0.975))],
        "local_ci95": [float(np.quantile(local_values, 0.025)), float(np.quantile(local_values, 0.975))],
        "local_minus_pooled_ci95": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "probability_local_better": float(np.mean(differences < 0.0)),
        "group_pooled_residual_mse": [float(values.mean()) for values in pooled_errors],
        "group_local_residual_mse": [float(values.mean()) for values in local_errors],
    }


def _one_record(
    source: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    bins: int,
    seed: int,
    ridge: float,
    min_samples: int,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    groups = [np.flatnonzero(labels == label) for label in sorted(np.unique(labels))]
    if any(group.size < int(min_samples) for group in groups):
        raise ValueError(f"a group has fewer than min_samples={min_samples}")
    split_rng = np.random.default_rng(seed + 100_003)
    local_train_indices: list[np.ndarray] = []
    local_test_indices: list[np.ndarray] = []
    for group in groups:
        train, test = _split_indices(group, split_rng)
        local_train_indices.append(train)
        local_test_indices.append(test)
    pooled_train = np.concatenate(local_train_indices)
    pooled_fit = _fit_operator(source[pooled_train], target[pooled_train], ridge=ridge)

    local_test_errors: list[np.ndarray] = []
    pooled_group_errors: list[np.ndarray] = []
    local_group_sizes = []
    for group, train, test in zip(groups, local_train_indices, local_test_indices):
        local_fit = _fit_operator(source[train], target[train], ridge=ridge)
        local_test_errors.append(_predict_squared_errors(local_fit, source[test], target[test]))
        pooled_group_errors.append(_predict_squared_errors(pooled_fit, source[test], target[test]))
        local_test_sizes = int(test.size)
        local_test_sizes and local_group_sizes.append(local_test_sizes)
    group_weights = np.asarray(local_group_sizes, dtype=np.float64)
    group_weights /= group_weights.sum()
    result = _bootstrap_paired(
        pooled_group_errors,
        local_test_errors,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed + 400_003,
        group_weights=group_weights,
    )
    result.update(
        {
            "seed": int(seed),
            "bins": int(bins),
            "group_sizes": [int(group.size) for group in groups],
            "test_group_sizes": local_group_sizes,
            "operator_fit": "pooled and action-conditional ridge operators fit on disjoint cross-fit halves",
            "bootstrap": "stratified held-out row bootstrap, paired indices for pooled/local predictions",
        }
    )
    return result


def run_report(
    latents: Path,
    max_samples: int,
    sample_seed: int,
    seeds: list[int],
    bins: list[int],
    ridge: float,
    min_samples: int,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    arrays = _load_arrays(latents, max_samples=max_samples, sample_seed=sample_seed)
    source = arrays["latents"]
    target = source + arrays["predicted_dynamics"]
    records = []
    for seed in seeds:
        for num_bins in bins:
            labels = _action_labels(arrays["actions"], num_bins)
            records.append(
                _one_record(
                    source,
                    target,
                    labels,
                    bins=num_bins,
                    seed=seed,
                    ridge=ridge,
                    min_samples=min_samples,
                    bootstrap_repeats=bootstrap_repeats,
                )
            )
    report = {
        "protocol": {
            "source": str(latents),
            "source_fields": ["latents", "predicted_dynamics", "actions"],
            "target": "latents + predicted_dynamics",
            "max_samples": int(max_samples),
            "num_loaded_samples": int(source.shape[0]),
            "latent_dim": int(source.shape[1]),
            "sample_seed": int(sample_seed),
            "seeds": [int(seed) for seed in seeds],
            "bins": [int(value) for value in bins],
            "ridge": float(ridge),
            "min_samples": int(min_samples),
            "bootstrap_repeats": int(bootstrap_repeats),
            "execution": "numpy_cpu",
            "frozen_checkpoint": True,
            "conditional_on_fitted_operators": True,
            "end_to_end_retraining": False,
            "rollout_and_planning_available": False,
        },
        "records": records,
        "claim_boundary": [
            "bootstrap uncertainty is conditional on frozen cross-fit operators",
            "paired held-out residual evidence only; no causal intervention",
            "no episode/timestep or multi-horizon evidence is available in the source artifact",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Held-out residual bootstrap for frozen LeWM local operators",
        "",
        "Pooled and action-conditional ridge operators are fit on disjoint cross-fit halves. The held-out residual comparison uses a stratified paired bootstrap, so each replicate evaluates pooled and local predictions on the same resampled rows.",
        "",
        "| seed | bins | pooled MSE | local MSE | local minus pooled | relative reduction | 95% CI for difference | P(local better) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for record in report["records"]:
        ci = record["local_minus_pooled_ci95"]
        lines.append(
            f"| {record['seed']} | {record['bins']} | {record['pooled_residual_mse']:.6g} | "
            f"{record['local_residual_mse']:.6g} | {record['local_minus_pooled']:.6g} | "
            f"{record['relative_reduction_percent']:.3f}% | [{ci[0]:.6g}, {ci[1]:.6g}] | "
            f"{record['probability_local_better']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "The bootstrap interval conditions on the fitted frozen operators and therefore does not quantify retraining or operator-fit uncertainty. This is frozen one-step evidence only; it does not establish causal action effects, trajectory rollout, online planning, or task success.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--sample-seed", type=int, default=3072)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3072, 3073, 3074])
    parser.add_argument("--bins", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--min-samples", type=int, default=256)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    started = time.time()
    report = run_report(
        latents=args.latents,
        max_samples=args.max_samples,
        sample_seed=args.sample_seed,
        seeds=args.seeds,
        bins=args.bins,
        ridge=args.ridge,
        min_samples=args.min_samples,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    report["protocol"]["elapsed_seconds"] = float(time.time() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
