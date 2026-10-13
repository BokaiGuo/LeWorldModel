"""Paired seed inference for the data-efficiency sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = {
    "test_one_step_mse": lambda row: float(row["test_one_step_mse"]),
    "rollout_h8_mse": lambda row: float(row["rollout_h8_mse"]),
    "composition_error": lambda row: float(row["composition_error"]),
    "predictor_equivariance_error": lambda row: float(row["predictor_equivariance_error"]),
    "operator_alignment": lambda row: float(row["operator_alignment"]),
}


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


def _condition(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["sample_count"]), str(row["mode"])


def _paired(rows, target, baseline, metric):
    target_rows = {int(row["seed"]): row for row in rows if _condition(row) == target}
    baseline_rows = {int(row["seed"]): row for row in rows if _condition(row) == baseline}
    seeds = sorted(set(target_rows).intersection(baseline_rows))
    if not seeds:
        raise ValueError(f"no paired seeds for target={target}, baseline={baseline}")
    return np.asarray(
        [METRICS[metric](target_rows[seed]) - METRICS[metric](baseline_rows[seed]) for seed in seeds],
        dtype=np.float64,
    )


def _summary(values: np.ndarray, repeats: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [rng.choice(values, size=values.size, replace=True).mean() for _ in range(int(repeats))],
        dtype=np.float64,
    )
    return {
        "n_paired_seeds": int(values.size),
        "mean_difference_target_minus_baseline": float(values.mean()),
        "std_difference": float(values.std(ddof=1) if values.size > 1 else 0.0),
        "bootstrap_mean_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "fraction_target_lower": float(np.mean(values < 0.0)),
        "paired_differences": [float(value) for value in values],
    }


def run_report(input_path: Path, repeats: int = 10000, seed: int = 161803) -> dict[str, Any]:
    source = json.loads(input_path.read_text())
    rows = list(source["records"])
    sample_counts = sorted({int(row["sample_count"]) for row in rows})
    comparisons = {
        "predictor_vs_none": ("predictor", "none"),
        "joint_vs_none": ("both", "none"),
        "joint_vs_predictor": ("both", "predictor"),
    }
    report: dict[str, Any] = {
        "protocol": {
            "input": str(input_path),
            "paired_unit": "same sample count and seed",
            "bootstrap_unit": "paired seed-level differences",
            "bootstrap_repeats": int(repeats),
            "bootstrap_seed": int(seed),
            "lower_is_better_for_all_metrics": True,
            "synthetic_vector_world_only": True,
        },
        "comparisons": {},
        "claim_boundary": [
            "intervals summarize the observed five-seed differences at each sample count",
            "this is synthetic data-efficiency evidence, not a nonlinear LeWM or planning guarantee",
        ],
    }
    index = 0
    for sample_count in sample_counts:
        for name, (target_mode, baseline_mode) in comparisons.items():
            report["comparisons"][f"n_{sample_count}_{name}"] = {
                "sample_count": sample_count,
                "target_mode": target_mode,
                "baseline_mode": baseline_mode,
                "metrics": {
                    metric: _summary(
                        _paired(rows, (sample_count, target_mode), (sample_count, baseline_mode), metric),
                        repeats=repeats,
                        seed=seed + index * 23 + metric_index,
                    )
                    for metric_index, metric in enumerate(METRICS)
                },
            }
            index += 1
    if not _finite(report):
        raise ValueError("inference report contains non-finite values")
    return report


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Paired inference for the data-efficiency sweep",
        "",
        "Negative differences favor the target condition because all metrics are errors or misalignment diagnostics.",
        "",
        "| samples | comparison | metric | n | mean difference | 95% bootstrap CI | target lower |",
        "| ---: | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for comparison, values in report["comparisons"].items():
        for metric, summary in values["metrics"].items():
            ci = summary["bootstrap_mean_ci95"]
            lines.append(
                f"| {values['sample_count']} | {values['target_mode']} vs {values['baseline_mode']} | {metric} | "
                f"{summary['n_paired_seeds']} | {summary['mean_difference_target_minus_baseline']:.6g} | "
                f"[{ci[0]:.6g}, {ci[1]:.6g}] | {summary['fraction_target_lower']:.2f} |"
            )
    lines.extend(["", "## Boundary", "", "This is synthetic mechanism evidence only."])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=161803)
    args = parser.parse_args()
    report = run_report(args.input, args.bootstrap_repeats, args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
