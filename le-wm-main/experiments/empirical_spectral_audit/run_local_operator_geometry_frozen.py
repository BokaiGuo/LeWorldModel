"""Cross-fit local operator geometry on a frozen LeWM latent export.

The audit fits a linear operator from ``z`` to the frozen predictor's
one-step latent target ``z + predicted_dynamics``.  It compares a pooled
covariance/operator pair with action-quantile local pairs and with shuffled
action-label controls.  Operators are fitted on one within-group half and
commutators/residuals are evaluated on the other half.

This is frozen-representation evidence.  It is not end-to-end training,
multi-step rollout, online control, or planning success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


def _load_arrays(path: Path, max_samples: int, sample_seed: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"latents", "predicted_dynamics", "actions"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"missing required arrays: {missing}")
        n = int(archive["latents"].shape[0])
        if any(int(archive[name].shape[0]) != n for name in required):
            raise ValueError("required arrays have inconsistent row counts")
        if max_samples and max_samples < n:
            rng = np.random.default_rng(sample_seed)
            indices = np.sort(rng.choice(n, size=int(max_samples), replace=False))
        else:
            indices = np.arange(n)
        arrays = {
            name: np.asarray(archive[name][indices], dtype=np.float64)
            for name in required
        }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("required arrays contain non-finite values")
    return arrays


def _action_labels(actions: np.ndarray, bins: int) -> np.ndarray:
    score = np.linalg.norm(np.asarray(actions, dtype=np.float64).reshape(actions.shape[0], -1), axis=1)
    thresholds = np.quantile(score, np.linspace(0.0, 1.0, int(bins) + 1)[1:-1])
    labels = np.digitize(score, thresholds, right=False).astype(np.int64)
    return labels


def _fit_operator(
    source: np.ndarray,
    target: np.ndarray,
    ridge: float,
) -> dict[str, np.ndarray]:
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError("source and target must be matching two-dimensional arrays")
    if source.shape[0] < 2:
        raise ValueError("at least two rows are required")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    denom = float(max(source.shape[0] - 1, 1))
    covariance = source_centered.T @ source_centered / denom
    cross_covariance = source_centered.T @ target_centered / denom
    scale = max(float(np.trace(covariance)) / source.shape[1], 1e-6)
    regularized = covariance + float(ridge) * scale * np.eye(source.shape[1])
    row_operator = np.linalg.solve(regularized, cross_covariance)
    return {
        "source_mean": source_mean,
        "target_mean": target_mean,
        "covariance": covariance,
        "row_operator": row_operator,
    }


def _commutator_sq(covariance: np.ndarray, row_operator: np.ndarray) -> float:
    operator = row_operator.T
    commutator = covariance @ operator - operator @ covariance
    denominator = float(np.square(covariance).sum() * np.square(operator).sum())
    return float(np.square(commutator).sum() / max(denominator, 1e-12))


def _evaluate_fit(
    fit: dict[str, np.ndarray],
    source: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    source_centered = source - fit["source_mean"]
    target_centered = target - fit["target_mean"]
    prediction = source_centered @ fit["row_operator"] + fit["target_mean"]
    covariance = (source - source.mean(axis=0)).T @ (source - source.mean(axis=0))
    covariance /= float(max(source.shape[0] - 1, 1))
    return {
        "commutator_sq": _commutator_sq(covariance, fit["row_operator"]),
        "residual_mse": float(np.square(prediction - target).mean()),
        "target_centered_mse": float(np.square(prediction - fit["target_mean"] - target_centered).mean()),
        "n_samples": int(source.shape[0]),
    }


def _split_indices(indices: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    midpoint = max(1, shuffled.size // 2)
    if midpoint >= shuffled.size:
        midpoint = shuffled.size - 1
    return shuffled[:midpoint], shuffled[midpoint:]


def _crossfit(
    source: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray | None,
    rng: np.random.Generator,
    ridge: float,
    min_samples: int,
) -> dict[str, Any]:
    if labels is None:
        groups = [np.arange(source.shape[0], dtype=np.int64)]
    else:
        groups = [np.flatnonzero(labels == label) for label in sorted(np.unique(labels))]
    if any(group.size < int(min_samples) for group in groups):
        raise ValueError(f"a group has fewer than min_samples={min_samples}")

    evaluations: list[dict[str, float]] = []
    group_sizes: list[dict[str, int]] = []
    for group_index, group in enumerate(groups):
        train, test = _split_indices(group, rng)
        fit = _fit_operator(source[train], target[train], ridge=ridge)
        result = _evaluate_fit(fit, source[test], target[test])
        result["group"] = int(group_index)
        result["train_samples"] = int(train.size)
        result["test_samples"] = int(test.size)
        evaluations.append(result)
        group_sizes.append({"group": int(group_index), "train": int(train.size), "test": int(test.size)})

    weights = np.asarray([item["test_samples"] for item in evaluations], dtype=np.float64)
    weights /= weights.sum()
    return {
        "groups": evaluations,
        "group_sizes": group_sizes,
        "commutator_sq_mean": float(np.mean([item["commutator_sq"] for item in evaluations])),
        "commutator_sq_weighted": float(np.sum(weights * [item["commutator_sq"] for item in evaluations])),
        "residual_mse_mean": float(np.mean([item["residual_mse"] for item in evaluations])),
        "residual_mse_weighted": float(np.sum(weights * [item["residual_mse"] for item in evaluations])),
    }


def _full_fit(
    source: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray | None,
    ridge: float,
    min_samples: int,
) -> dict[str, Any]:
    fit = _fit_operator(source, target, ridge=ridge)
    pooled = _evaluate_fit(fit, source, target)
    if labels is None:
        groups = [np.arange(source.shape[0], dtype=np.int64)]
    else:
        groups = [np.flatnonzero(labels == label) for label in sorted(np.unique(labels))]
    if any(group.size < int(min_samples) for group in groups):
        raise ValueError(f"a group has fewer than min_samples={min_samples}")
    local = []
    for group_index, group in enumerate(groups):
        local_fit = _fit_operator(source[group], target[group], ridge=ridge)
        item = _evaluate_fit(local_fit, source[group], target[group])
        item["group"] = int(group_index)
        local.append(item)
    sizes = np.asarray([item["n_samples"] for item in local], dtype=np.float64)
    sizes /= sizes.sum()
    return {
        "pooled": pooled,
        "local": local,
        "local_commutator_sq_mean": float(np.mean([item["commutator_sq"] for item in local])),
        "local_commutator_sq_weighted": float(np.sum(sizes * [item["commutator_sq"] for item in local])),
        "local_residual_mse_weighted": float(np.sum(sizes * [item["residual_mse"] for item in local])),
        "group_sizes": [int(group.size) for group in groups],
    }


def _one_record(
    source: np.ndarray,
    target: np.ndarray,
    action_labels: np.ndarray,
    bins: int,
    seed: int,
    ridge: float,
    min_samples: int,
    shuffle_repeats: int,
) -> dict[str, Any]:
    split_rng = np.random.default_rng(seed + 100_003)
    conditional_full = _full_fit(source, target, action_labels, ridge, min_samples)
    conditional_crossfit = _crossfit(source, target, action_labels, split_rng, ridge, min_samples)
    pooled_full = _full_fit(source, target, None, ridge, min_samples)
    pooled_crossfit = _crossfit(source, target, None, split_rng, ridge, min_samples)

    shuffle_values = []
    for repeat in range(int(shuffle_repeats)):
        shuffle_rng = np.random.default_rng(seed + 200_003 + repeat)
        shuffled_labels = action_labels.copy()
        shuffle_rng.shuffle(shuffled_labels)
        shuffled = _crossfit(
            source,
            target,
            shuffled_labels,
            np.random.default_rng(seed + 300_003 + repeat),
            ridge,
            min_samples,
        )
        shuffle_values.append(shuffled)

    return {
        "seed": int(seed),
        "bins": int(bins),
        "action_group_sizes": [int(value) for value in np.bincount(action_labels, minlength=bins)],
        "pooled_full": pooled_full,
        "pooled_crossfit": pooled_crossfit,
        "conditional_full": conditional_full,
        "conditional_crossfit": conditional_crossfit,
        "action_shuffle_null": {
            "repeats": int(shuffle_repeats),
            "commutator_sq_weighted_mean": float(np.mean([item["commutator_sq_weighted"] for item in shuffle_values])),
            "commutator_sq_weighted_values": [item["commutator_sq_weighted"] for item in shuffle_values],
            "residual_mse_weighted_mean": float(np.mean([item["residual_mse_weighted"] for item in shuffle_values])),
        },
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Cross-fit local operator geometry on frozen LeWM latents",
        "",
        "This audit compares pooled and action-quantile local covariance--operator pairs on the frozen one-step target `z + predicted_dynamics`.",
        "It is frozen-representation evidence, not end-to-end training, rollout, online control, or planning success.",
        "",
        "## Cross-fit summary",
        "",
        "| seed | bins | pooled commutator | local commutator | shuffled-label null | pooled residual MSE | local residual MSE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in report["records"]:
        lines.append(
            f"| {record['seed']} | {record['bins']} | "
            f"{record['pooled_crossfit']['commutator_sq_weighted']:.6g} | "
            f"{record['conditional_crossfit']['commutator_sq_weighted']:.6g} | "
            f"{record['action_shuffle_null']['commutator_sq_weighted_mean']:.6g} | "
            f"{record['pooled_crossfit']['residual_mse_weighted']:.6g} | "
            f"{record['conditional_crossfit']['residual_mse_weighted']:.6g} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "The local operator is fitted from a frozen predictor export and evaluated by within-group cross-fitting. The action-shuffle result is a label-randomization control, not a causal intervention. Missing episode/timestep and multi-horizon fields prevent claims about trajectory-level rollout or online planning.",
    ])
    path.write_text("\n".join(lines) + "\n")


def run_report(
    latents: Path,
    max_samples: int,
    sample_seed: int,
    seeds: list[int],
    bins: list[int],
    ridge: float,
    min_samples: int,
    shuffle_repeats: int,
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
                    shuffle_repeats=shuffle_repeats,
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
            "shuffle_repeats": int(shuffle_repeats),
            "crossfit": "within-group half split; operator fit on one half, covariance/residual evaluated on the other",
            "execution": "numpy_cpu",
            "frozen_checkpoint": True,
            "end_to_end_retraining": False,
            "rollout_and_planning_available": False,
        },
        "records": records,
        "claim_boundary": [
            "frozen one-step predictor geometry only",
            "action-label shuffle is a null control, not a causal intervention",
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
    parser.add_argument("--shuffle-repeats", type=int, default=3)
    args = parser.parse_args()
    report = run_report(
        latents=args.latents,
        max_samples=args.max_samples,
        sample_seed=args.sample_seed,
        seeds=args.seeds,
        bins=args.bins,
        ridge=args.ridge,
        min_samples=args.min_samples,
        shuffle_repeats=args.shuffle_repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
