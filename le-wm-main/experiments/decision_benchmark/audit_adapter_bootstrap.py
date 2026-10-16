"""Audit root-level uncertainty for joint RGB/task adapter artifacts.

The adapter report contains repeated seed fits on the same roots.  Treating
all seed-root rows as independent would make the confidence interval too
optimistic, so this script first averages gains within root across seeds and
then bootstraps roots.  It also reports leave-one-root-out stability and the
paired difference between two protocol-compatible artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _objective_root_matrix(report: dict[str, Any], objective: str) -> np.ndarray:
    seeds = report["objectives"][objective]["seeds"]
    rows = [
        np.asarray(seed["outcome"]["ranking"]["paired_regret_gain_values"], dtype=np.float64)
        for seed in seeds
    ]
    matrix = np.stack(rows, axis=0)
    if matrix.ndim != 2:
        raise ValueError(f"expected [seed, root] gain matrix, got {matrix.shape}")
    return matrix


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("root bootstrap needs at least two roots")
    indices = rng.integers(0, values.size, size=(int(draws), values.size))
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _audit_objective(
    report: dict[str, Any],
    objective: str,
    *,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, Any]:
    matrix = _objective_root_matrix(report, objective)
    root_values = matrix.mean(axis=0)
    mean, low, high = _bootstrap_mean(root_values, rng, draws)
    loo = np.asarray(
        [np.mean(np.delete(root_values, index)) for index in range(root_values.size)],
        dtype=np.float64,
    )
    seed_means = matrix.mean(axis=1)
    return {
        "seed_count": int(matrix.shape[0]),
        "root_count": int(matrix.shape[1]),
        "seed_mean_gain": [float(value) for value in seed_means],
        "root_mean_gain": float(mean),
        "root_bootstrap_ci95": [float(low), float(high)],
        "root_gain_values": [float(value) for value in root_values],
        "leave_one_root_out_mean_min": float(loo.min()),
        "leave_one_root_out_mean_max": float(loo.max()),
        "positive_root_fraction": float(np.mean(root_values > 0.0)),
        "stable_after_any_single_root_removal": bool(np.all(loo > 0.0)),
    }


def audit(
    report_path: Path,
    output_path: Path,
    *,
    comparison_report: Path | None = None,
    draws: int = 20000,
    seed: int = 20261016,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(int(seed))
    objectives = sorted(report.get("objectives", {}))
    result: dict[str, Any] = {
        "schema": "adapter_root_bootstrap_audit.v1",
        "report": str(report_path.resolve()),
        "bootstrap_unit": "root_mean_across_seeds",
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "objectives": {
            objective: _audit_objective(report, objective, rng=rng, draws=draws)
            for objective in objectives
        },
    }
    if comparison_report is not None:
        comparison = json.loads(comparison_report.read_text(encoding="utf-8"))
        paired: dict[str, Any] = {}
        for objective in sorted(set(objectives).intersection(comparison.get("objectives", {}))):
            first = _objective_root_matrix(report, objective).mean(axis=0)
            second = _objective_root_matrix(comparison, objective).mean(axis=0)
            if first.shape != second.shape:
                raise ValueError(f"root mismatch for {objective}: {first.shape} vs {second.shape}")
            delta = first - second
            mean, low, high = _bootstrap_mean(delta, rng, draws)
            paired[objective] = {
                "root_count": int(delta.size),
                "first_minus_comparison_mean": float(mean),
                "root_bootstrap_ci95": [float(low), float(high)],
                "root_delta_values": [float(value) for value in delta],
            }
        result["paired_comparison"] = {
            "comparison_report": str(comparison_report.resolve()),
            "semantics": "first_report_gain_minus_comparison_report_gain",
            "objectives": paired,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, default=None)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    result = audit(
        report_path=args.report,
        output_path=args.output,
        comparison_report=args.comparison_report,
        draws=args.draws,
        seed=args.seed,
    )
    for objective, value in result["objectives"].items():
        print(
            objective,
            "root_mean=", f"{value['root_mean_gain']:.6f}",
            "ci95=", [round(number, 6) for number in value["root_bootstrap_ci95"]],
            "loo_min=", f"{value['leave_one_root_out_mean_min']:.6f}",
        )


if __name__ == "__main__":
    main()
