"""Bootstrap and permutation audit for a frozen LeWM predictor export.

The compact PushT artifact contains paired one-step latent displacements but
not episode identifiers or enough fields for trajectory-level evaluation. This
audit therefore targets one estimand only: whether the exported predicted
displacement is more aligned with the paired true displacement than with a
row-permuted null. It reports bootstrap confidence intervals and action-norm
stratified summaries. It is frozen-checkpoint evidence, not retraining,
rollout, planning success, or causal evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("left and right must be matching two-dimensional arrays")
    numerator = np.sum(left.astype(np.float64) * right.astype(np.float64), axis=1)
    denominator = np.linalg.norm(left.astype(np.float64), axis=1)
    denominator *= np.linalg.norm(right.astype(np.float64), axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def _mean_std(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    }


def _percentile_interval(values: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "low": float(np.quantile(values, alpha / 2.0)),
        "high": float(np.quantile(values, 1.0 - alpha / 2.0)),
    }


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, repeats: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("bootstrap requires at least two rows")
    result = np.empty(int(repeats), dtype=np.float64)
    for index in range(int(repeats)):
        sample = rng.integers(0, values.size, size=values.size)
        result[index] = float(np.mean(values[sample]))
    return result


def _action_labels(actions: np.ndarray, bins: int) -> np.ndarray:
    score = np.linalg.norm(np.asarray(actions, dtype=np.float64), axis=1)
    if int(bins) <= 1:
        return np.zeros(score.shape[0], dtype=np.int64)
    thresholds = np.quantile(score, np.linspace(0.0, 1.0, int(bins) + 1)[1:-1])
    return np.digitize(score, thresholds, right=False).astype(np.int64)


def _summary_for_rows(
    true_delta: np.ndarray,
    predicted_delta: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    bootstrap_repeats: int,
    permutation_repeats: int,
) -> dict[str, Any]:
    paired_cosine = _cosine_rows(true_delta, predicted_delta)
    paired_mse = np.mean((true_delta.astype(np.float64) - predicted_delta.astype(np.float64)) ** 2, axis=1)

    null_cosines = np.empty(int(permutation_repeats), dtype=np.float64)
    for index in range(int(permutation_repeats)):
        shuffled = rng.permutation(predicted_delta.shape[0])
        null_cosines[index] = float(np.mean(_cosine_rows(true_delta, predicted_delta[shuffled])))
    bootstrap_cosine = _bootstrap_mean(paired_cosine, rng, bootstrap_repeats)
    bootstrap_mse = _bootstrap_mean(paired_mse, rng, bootstrap_repeats)
    null_p_value = float((1.0 + np.sum(null_cosines >= np.mean(paired_cosine))) / (null_cosines.size + 1.0))

    output: dict[str, Any] = {
        "n_rows": int(true_delta.shape[0]),
        "paired_cosine": {
            **_mean_std(paired_cosine),
            "bootstrap_ci95": _percentile_interval(bootstrap_cosine),
        },
        "paired_mse": {
            **_mean_std(paired_mse),
            "bootstrap_ci95": _percentile_interval(bootstrap_mse),
        },
        "row_permutation_null_cosine": {
            **_mean_std(null_cosines),
            "quantile95": _percentile_interval(null_cosines),
            "p_value_mean_null_ge_paired": null_p_value,
            "repeats": int(permutation_repeats),
        },
        "cosine_gap_paired_minus_null_mean": float(np.mean(paired_cosine) - np.mean(null_cosines)),
        "bootstrap_repeats": int(bootstrap_repeats),
    }
    for label in sorted(np.unique(labels)):
        mask = labels == label
        if int(mask.sum()) < 32:
            continue
        group_true = true_delta[mask]
        group_pred = predicted_delta[mask]
        group_paired = _cosine_rows(group_true, group_pred)
        group_null = np.empty(int(permutation_repeats), dtype=np.float64)
        for index in range(int(permutation_repeats)):
            shuffled = rng.permutation(group_pred.shape[0])
            group_null[index] = float(np.mean(_cosine_rows(group_true, group_pred[shuffled])))
        group_bootstrap = _bootstrap_mean(group_paired, rng, bootstrap_repeats)
        output.setdefault("action_groups", {})[str(int(label))] = {
            "n_rows": int(mask.sum()),
            "paired_cosine_mean": float(np.mean(group_paired)),
            "paired_cosine_ci95": _percentile_interval(group_bootstrap),
            "null_cosine_mean": float(np.mean(group_null)),
            "null_cosine_ci95": _percentile_interval(group_null),
            "cosine_gap": float(np.mean(group_paired) - np.mean(group_null)),
        }
    return output


def _load(path: Path, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, mmap_mode="r") as data:
        required = {"dynamics", "predicted_dynamics", "actions"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"missing required fields: {missing}")
        n_rows = int(data["dynamics"].shape[0])
        rng = np.random.default_rng(seed)
        rows = np.arange(n_rows)
        if int(max_samples) > 0 and n_rows > int(max_samples):
            rows = np.sort(rng.choice(n_rows, size=int(max_samples), replace=False))
        true_delta = np.asarray(data["dynamics"][rows], dtype=np.float32)
        predicted_delta = np.asarray(data["predicted_dynamics"][rows], dtype=np.float32)
        actions = np.asarray(data["actions"][rows], dtype=np.float32)
    return true_delta, predicted_delta, actions


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    overall = report["overall"]
    lines = [
        "# Frozen LeWM predictor bootstrap audit",
        "",
        "This is a frozen-checkpoint one-step alignment diagnostic. It compares paired true and predicted latent displacements with a row-permutation null and reports bootstrap intervals. It is not end-to-end training, trajectory rollout, online planning, or causal evidence.",
        "",
        "## Overall result",
        "",
        "| rows | paired cosine | 95% bootstrap CI | permutation-null cosine | null 95% interval | paired-minus-null | null p-value | paired MSE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {overall['n_rows']} | {overall['paired_cosine']['mean']:.6g} | [{overall['paired_cosine']['bootstrap_ci95']['low']:.6g}, {overall['paired_cosine']['bootstrap_ci95']['high']:.6g}] | {overall['row_permutation_null_cosine']['mean']:.6g} | [{overall['row_permutation_null_cosine']['quantile95']['low']:.6g}, {overall['row_permutation_null_cosine']['quantile95']['high']:.6g}] | {overall['cosine_gap_paired_minus_null_mean']:.6g} | {overall['row_permutation_null_cosine']['p_value_mean_null_ge_paired']:.6g} | {overall['paired_mse']['mean']:.6g} |",
        "",
        "## Action-norm strata",
        "",
        "| bin | rows | paired cosine | 95% CI | null cosine | null 95% interval | gap |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, values in report["overall"].get("action_groups", {}).items():
        lines.append(
            f"| {label} | {values['n_rows']} | {values['paired_cosine_mean']:.6g} | "
            f"[{values['paired_cosine_ci95']['low']:.6g}, {values['paired_cosine_ci95']['high']:.6g}] | "
            f"{values['null_cosine_mean']:.6g} | [{values['null_cosine_ci95']['low']:.6g}, {values['null_cosine_ci95']['high']:.6g}] | "
            f"{values['cosine_gap']:.6g} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "The permutation null tests row-level pairing in the frozen export; it does not randomize actions in the environment and does not establish causality. Missing episode/timestep fields still prevent trajectory-level and online-control claims.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--action-bins", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    parser.add_argument("--permutation-repeats", type=int, default=200)
    args = parser.parse_args()
    if args.bootstrap_repeats < 20 or args.permutation_repeats < 20:
        raise ValueError("at least 20 repeats are required for a stable diagnostic")
    true_delta, predicted_delta, actions = _load(args.latents, args.max_samples, args.seed)
    start = time.time()
    rng = np.random.default_rng(args.seed + 1)
    report: dict[str, Any] = {
        "protocol": {
            "source": str(args.latents),
            "max_samples": int(args.max_samples),
            "seed": int(args.seed),
            "action_bins": [int(value) for value in args.action_bins],
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "permutation_repeats": int(args.permutation_repeats),
            "estimand": "paired true-vs-predicted one-step latent displacement cosine",
            "null": "row permutation of predicted displacement",
            "frozen_checkpoint_only": True,
            "trajectory_or_online_evidence": False,
            "elapsed_seconds": None,
        },
        "overall": None,
        "by_action_bins": {},
    }
    labels_for_overall = _action_labels(actions, 2)
    report["overall"] = _summary_for_rows(
        true_delta,
        predicted_delta,
        labels_for_overall,
        rng,
        args.bootstrap_repeats,
        args.permutation_repeats,
    )
    for bins in args.action_bins:
        labels = _action_labels(actions, bins)
        report["by_action_bins"][str(int(bins))] = _summary_for_rows(
            true_delta,
            predicted_delta,
            labels,
            np.random.default_rng(args.seed + 10_000 + int(bins)),
            args.bootstrap_repeats,
            args.permutation_repeats,
        )
    report["protocol"]["elapsed_seconds"] = time.time() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(json.dumps({"overall": report["overall"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
