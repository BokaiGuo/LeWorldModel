"""Permutation control for action-conditional frozen LeWM operators.

The local residual bootstrap establishes that action-quantile operators predict
the frozen one-step target better than a pooled operator.  This audit asks the
necessary capacity-control question: does the gain remain after replacing the
action labels with random labels having exactly the same group sizes?

All methods use the same global train/holdout rows and ridge strength.  The
pooled operator is refit on the common training rows for each configuration;
only the local partition is changed.  This is frozen one-step evidence, not a
causal action intervention, rollout, online planning, or task-success test.
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
    )
except ModuleNotFoundError:  # direct invocation from the script directory
    from run_local_operator_geometry_frozen import _action_labels, _fit_operator, _load_arrays


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


def _split_global(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 4:
        raise ValueError("at least four rows are required")
    order = np.random.default_rng(seed).permutation(n)
    cut = max(1, min(n - 1, n // 2))
    return order[:cut], order[cut:]


def _partition_mse(
    source: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    ridge: float,
    min_samples: int,
) -> dict[str, float]:
    if labels.shape != (source.shape[0],):
        raise ValueError("labels must have one entry per row")
    pooled_fit = _fit_operator(source[train], target[train], ridge=ridge)
    pooled_prediction = (
        source[test] - pooled_fit["source_mean"]
    ) @ pooled_fit["row_operator"] + pooled_fit["target_mean"]
    pooled_mse = float(np.square(pooled_prediction - target[test]).mean())

    local_predictions = np.empty_like(target[test])
    group_sizes: list[int] = []
    for label in sorted(np.unique(labels).tolist()):
        train_group = train[labels[train] == label]
        test_mask = labels[test] == label
        if train_group.size < int(min_samples) or not bool(test_mask.any()):
            raise ValueError(f"partition group {label} is too small")
        fit = _fit_operator(source[train_group], target[train_group], ridge=ridge)
        local_predictions[test_mask] = (
            source[test[test_mask]] - fit["source_mean"]
        ) @ fit["row_operator"] + fit["target_mean"]
        group_sizes.append(int(train_group.size))
    local_mse = float(np.square(local_predictions - target[test]).mean())
    return {
        "pooled_mse": pooled_mse,
        "local_mse": local_mse,
        "local_minus_pooled": local_mse - pooled_mse,
        "relative_reduction_percent": 100.0 * (pooled_mse - local_mse) / max(pooled_mse, 1e-12),
        "train_samples": int(train.size),
        "test_samples": int(test.size),
        "train_group_sizes": group_sizes,
    }


def _one_record(
    source: np.ndarray,
    target: np.ndarray,
    action_labels: np.ndarray,
    bins: int,
    seed: int,
    ridge: float,
    min_samples: int,
    permutation_repeats: int,
) -> dict[str, Any]:
    train, test = _split_global(source.shape[0], seed + 100_003)
    observed = _partition_mse(
        source, target, action_labels, train, test, ridge=ridge, min_samples=min_samples
    )
    null_records: list[dict[str, float]] = []
    for repeat in range(int(permutation_repeats)):
        labels = action_labels.copy()
        np.random.default_rng(seed + 200_003 + repeat).shuffle(labels)
        result = _partition_mse(
            source, target, labels, train, test, ridge=ridge, min_samples=min_samples
        )
        null_records.append(result)
    null_reductions = np.asarray(
        [item["relative_reduction_percent"] for item in null_records], dtype=np.float64
    )
    observed_reduction = float(observed["relative_reduction_percent"])
    excess = observed_reduction - null_reductions
    return {
        "seed": int(seed),
        "bins": int(bins),
        "action_group_sizes": [int(v) for v in np.bincount(action_labels, minlength=bins)],
        "observed_action_partition": observed,
        "permutation_null": {
            "repeats": int(permutation_repeats),
            "reduction_mean_percent": float(null_reductions.mean()),
            "reduction_sd_percent": float(null_reductions.std(ddof=1)) if null_reductions.size > 1 else 0.0,
            "reduction_ci95_percent": [
                float(np.quantile(null_reductions, 0.025)),
                float(np.quantile(null_reductions, 0.975)),
            ],
            "excess_gain_mean_percent": float(excess.mean()),
            "excess_gain_ci95_percent": [
                float(np.quantile(excess, 0.025)),
                float(np.quantile(excess, 0.975)),
            ],
            "permutation_p_observed_not_exceeding_null": float(
                (1.0 + np.sum(null_reductions >= observed_reduction))
                / (null_reductions.size + 1.0)
            ),
            "all_reduction_percent": [float(v) for v in null_reductions],
        },
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Action-label permutation null for frozen LeWM local operators",
        "",
        "The observed action-quantile partition is compared with equal-size random label partitions. All configurations use the same global train/holdout rows and ridge operator family.",
        "",
        "| seed | bins | observed reduction | random-label mean | random-label 95% interval | excess over null | permutation p |",
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for record in report["records"]:
        observed = record["observed_action_partition"]["relative_reduction_percent"]
        null = record["permutation_null"]
        lines.append(
            f"| {record['seed']} | {record['bins']} | {observed:.4f}% | "
            f"{null['reduction_mean_percent']:.4f}% | "
            f"[{null['reduction_ci95_percent'][0]:.4f}, {null['reduction_ci95_percent'][1]:.4f}]% | "
            f"{observed - null['reduction_mean_percent']:.4f} pp | "
            f"{null['permutation_p_observed_not_exceeding_null']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "The permutation null controls equal-size partition capacity, not causal action assignment. Operators and targets are frozen one-step exports; the source lacks episode/timestep and multi-horizon fields, so this does not establish rollout, online planning, control, or task success.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run_report(
    latents: Path,
    max_samples: int,
    sample_seed: int,
    seeds: list[int],
    bins: list[int],
    ridge: float,
    min_samples: int,
    permutation_repeats: int,
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
                    permutation_repeats=permutation_repeats,
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
            "permutation_repeats": int(permutation_repeats),
            "split": "one common global 50/50 train/holdout split per seed",
            "null": "uniform permutation of action labels preserving exact group sizes",
            "execution": "numpy_cpu",
            "frozen_checkpoint": True,
            "end_to_end_retraining": False,
            "rollout_and_planning_available": False,
        },
        "records": records,
        "claim_boundary": [
            "the permutation null controls equal-size partition capacity, not causality",
            "results are frozen one-step prediction evidence",
            "no episode/timestep or multi-horizon evidence is available in the source artifact",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    return report


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
    parser.add_argument("--permutation-repeats", type=int, default=50)
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
        permutation_repeats=args.permutation_repeats,
    )
    report["protocol"]["elapsed_seconds"] = float(time.time() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
