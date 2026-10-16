"""Stratified aggregation of fresh Reacher capacity-readout replications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stratified_bootstrap(
    batches: list[np.ndarray], *, draws: int, seed: int
) -> dict[str, Any]:
    if len(batches) < 2:
        raise ValueError("stratified aggregation requires at least two batches")
    arrays = [np.asarray(values, dtype=np.float64) for values in batches]
    if any(values.ndim != 1 or len(values) < 2 for values in arrays):
        raise ValueError("each batch must contain at least two one-dimensional rows")
    rng = np.random.default_rng(seed)
    sampled_sums = np.zeros(draws, dtype=np.float64)
    total = 0
    for values in arrays:
        indexes = rng.integers(0, len(values), size=(draws, len(values)))
        sampled_sums += values[indexes].sum(axis=1)
        total += len(values)
    means = sampled_sums / total
    concatenated = np.concatenate(arrays)
    return {
        "n": int(total),
        "batch_ns": [int(len(values)) for values in arrays],
        "mean": float(np.mean(concatenated)),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "bootstrap_unit": "root_within_batch",
    }


def _heterogeneity(
    batches: list[np.ndarray], *, draws: int, seed: int
) -> dict[str, Any]:
    arrays = [np.asarray(values, dtype=np.float64) for values in batches]
    means = np.asarray([np.mean(values) for values in arrays], dtype=np.float64)
    pairwise: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for left in range(len(arrays)):
        for right in range(left + 1, len(arrays)):
            left_values = arrays[left]
            right_values = arrays[right]
            left_indexes = rng.integers(
                0, len(left_values), size=(draws, len(left_values))
            )
            right_indexes = rng.integers(
                0, len(right_values), size=(draws, len(right_values))
            )
            contrasts = (
                right_values[right_indexes].mean(axis=1)
                - left_values[left_indexes].mean(axis=1)
            )
            pairwise.append(
                {
                    "left_batch": left,
                    "right_batch": right,
                    "right_minus_left": float(means[right] - means[left]),
                    "ci95": [
                        float(np.quantile(contrasts, 0.025)),
                        float(np.quantile(contrasts, 0.975)),
                    ],
                }
            )
    return {
        "batch_means": [float(value) for value in means],
        "range": float(np.max(means) - np.min(means)),
        "between_batch_sd": float(np.std(means, ddof=1)),
        "sign_consistent": bool(np.all(means >= 0) or np.all(means <= 0)),
        "pairwise_contrasts": pairwise,
    }


def _planning_arrays(
    comparison: dict[str, Any], comparison_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if comparison.get("schema") != "reacher_capacity_readout_paired_comparison.v1":
        raise ValueError(f"{comparison_path}: unexpected comparison schema")
    protocol = comparison["protocol"]
    baseline_path = Path(protocol["baseline_report"])
    readout_path = Path(protocol["readout_report"])
    baseline = _read_json(baseline_path)
    readout = _read_json(readout_path)
    base_rows = baseline["frozen_lewm"]
    readout_rows = readout["frozen_lewm"]
    if base_rows["seeds"] != readout_rows["seeds"]:
        raise ValueError(f"{comparison_path}: baseline/readout seeds differ")
    if base_rows["seeds"] != protocol["seeds"]:
        raise ValueError(f"{comparison_path}: raw and summarized seeds differ")
    arrays = {
        "success_base": np.asarray(base_rows["episode_successes"], dtype=np.float64),
        "success_readout": np.asarray(
            readout_rows["episode_successes"], dtype=np.float64
        ),
        "final_base": np.asarray(base_rows["final_qpos_distance"], dtype=np.float64),
        "final_readout": np.asarray(
            readout_rows["final_qpos_distance"], dtype=np.float64
        ),
        "min_base": np.asarray(base_rows["min_qpos_distance"], dtype=np.float64),
        "min_readout": np.asarray(
            readout_rows["min_qpos_distance"], dtype=np.float64
        ),
    }
    lengths = {len(values) for values in arrays.values()}
    if lengths != {len(base_rows["seeds"])}:
        raise ValueError(f"{comparison_path}: raw paired arrays have inconsistent lengths")
    return comparison, arrays


def aggregate(
    comparison_paths: list[Path], *, draws: int = 20_000, seed: int = 20261020
) -> dict[str, Any]:
    if len(comparison_paths) < 2:
        raise ValueError("provide at least two independent batch comparisons")
    loaded = [
        _planning_arrays(_read_json(path), path) for path in comparison_paths
    ]
    comparisons = [item[0] for item in loaded]
    arrays = [item[1] for item in loaded]
    all_seeds = [
        int(value)
        for comparison in comparisons
        for value in comparison["protocol"]["seeds"]
    ]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("replication batches must contain disjoint root seeds")
    artifacts = {
        comparison["protocol"].get("capacity_readout_artifact")
        for comparison in comparisons
    }
    if len(artifacts) != 1:
        raise ValueError("all batches must use the same frozen capacity readout")

    success_deltas = [row["success_readout"] - row["success_base"] for row in arrays]
    final_deltas = [row["final_readout"] - row["final_base"] for row in arrays]
    min_deltas = [row["min_readout"] - row["min_base"] for row in arrays]
    diagnostics = [comparison.get("runtime_diagnostics", {}) for comparison in comparisons]
    active = sum(int(row.get("active_rows", 0)) for row in diagnostics)
    evaluated = sum(int(row.get("evaluated_rows", 0)) for row in diagnostics)
    changed = sum(int(row.get("argmin_changes", 0)) for row in diagnostics)
    weighted_support = sum(
        float(row.get("row_support_fraction_mean", row.get("proposal_support_rate", 0.0)))
        * int(row.get("active_rows", 0))
        for row in diagnostics
    )
    support_mins = [
        float(row["row_support_fraction_min"])
        for row in diagnostics
        if int(row.get("active_rows", 0)) > 0
        and "row_support_fraction_min" in row
    ]
    support_maxes = [
        float(row["row_support_fraction_max"])
        for row in diagnostics
        if int(row.get("active_rows", 0)) > 0
        and "row_support_fraction_max" in row
    ]

    batch_summaries = []
    for index, (path, comparison) in enumerate(zip(comparison_paths, comparisons)):
        batch_summaries.append(
            {
                "batch_index": index,
                "comparison_report": str(path.resolve()),
                "n": int(comparison["paired_readout_minus_baseline"]["success"]["n"]),
                "point_estimates": comparison["point_estimates"],
                "paired_readout_minus_baseline": comparison[
                    "paired_readout_minus_baseline"
                ],
                "runtime_diagnostics": comparison.get("runtime_diagnostics", {}),
            }
        )

    return {
        "schema": "reacher_capacity_readout_stratified_aggregate.v1",
        "protocol": {
            "comparison_reports": [str(path.resolve()) for path in comparison_paths],
            "batch_count": len(comparison_paths),
            "root_disjoint": True,
            "seeds": all_seeds,
            "bootstrap_draws": draws,
            "bootstrap_seed": seed,
            "capacity_readout_artifact": next(iter(artifacts)),
            "test_outcome_used_for_gate_tuning": False,
        },
        "batches": batch_summaries,
        "pooled_point_estimates": {
            "n": len(all_seeds),
            "baseline_successes": int(
                sum(np.sum(row["success_base"]) for row in arrays)
            ),
            "readout_successes": int(
                sum(np.sum(row["success_readout"]) for row in arrays)
            ),
        },
        "stratified_paired_readout_minus_baseline": {
            "success": _stratified_bootstrap(
                success_deltas, draws=draws, seed=seed
            ),
            "final_qpos_distance": _stratified_bootstrap(
                final_deltas, draws=draws, seed=seed + 1
            ),
            "min_qpos_distance": _stratified_bootstrap(
                min_deltas, draws=draws, seed=seed + 2
            ),
            "success_discordance": {
                "readout_only_wins": int(
                    sum(
                        np.sum(
                            (row["success_readout"] == 1)
                            & (row["success_base"] == 0)
                        )
                        for row in arrays
                    )
                ),
                "baseline_only_wins": int(
                    sum(
                        np.sum(
                            (row["success_readout"] == 0)
                            & (row["success_base"] == 1)
                        )
                        for row in arrays
                    )
                ),
                "ties": int(
                    sum(
                        np.sum(row["success_readout"] == row["success_base"])
                        for row in arrays
                    )
                ),
            },
        },
        "batch_heterogeneity": {
            "success": _heterogeneity(success_deltas, draws=draws, seed=seed + 10),
            "final_qpos_distance": _heterogeneity(
                final_deltas, draws=draws, seed=seed + 11
            ),
            "min_qpos_distance": _heterogeneity(
                min_deltas, draws=draws, seed=seed + 12
            ),
        },
        "runtime_diagnostics": {
            "active_rows": active,
            "evaluated_rows": evaluated,
            "abstained_low_support_rows": sum(
                int(row.get("abstained_low_support_rows", 0)) for row in diagnostics
            ),
            "row_support_fraction_mean": float(weighted_support / max(active, 1)),
            "row_support_fraction_min": float(min(support_mins)) if support_mins else 0.0,
            "row_support_fraction_max": float(max(support_maxes)) if support_maxes else 0.0,
            "argmin_changes": changed,
            "argmin_change_rate": float(changed / max(evaluated, 1)),
        },
        "claim_boundary": (
            "two-batch fresh Reacher replication of a phase-gated proposal-relative "
            "source-action readout; batch-specific uncertainty and continuous-error "
            "no-gain remain explicit; not predicted-dynamics or cross-environment evidence"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    report = aggregate(args.batch, draws=args.draws, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
