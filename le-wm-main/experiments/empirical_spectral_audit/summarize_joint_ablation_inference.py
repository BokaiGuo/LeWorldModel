"""Compute paired seed-level uncertainty summaries for the joint ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = {
    "test_one_step_mse": lambda row: float(row["test_one_step_mse"]),
    "rollout_h8_mse": lambda row: float(row["rollout_mse"]["8"]),
    "composition_error": lambda row: float(row["composition_error"]),
    "predictor_equivariance_error": lambda row: float(row["equivariance"]["predictor_equivariance_error"]),
    "operator_alignment": lambda row: float(row["operator_alignment"]["operator_alignment"]),
}


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


def _condition(row: dict[str, Any]) -> tuple[str, float, float]:
    return (str(row["mode"]), float(row["predictor_weight"]), float(row["alignment_weight"]))


def _paired_differences(
    rows: list[dict[str, Any]],
    target: tuple[str, float, float],
    baseline: tuple[str, float, float],
    metric: str,
) -> np.ndarray:
    target_rows = {int(row["seed"]): row for row in rows if _condition(row) == target}
    baseline_rows = {int(row["seed"]): row for row in rows if _condition(row) == baseline}
    seeds = sorted(set(target_rows).intersection(baseline_rows))
    if not seeds:
        raise ValueError(f"no paired seeds for target={target}, baseline={baseline}")
    return np.asarray(
        [METRICS[metric](target_rows[seed]) - METRICS[metric](baseline_rows[seed]) for seed in seeds],
        dtype=np.float64,
    )


def _summarize_difference(
    differences: np.ndarray,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(int(bootstrap_repeats), dtype=np.float64)
    for repeat in range(int(bootstrap_repeats)):
        bootstrap_means[repeat] = float(rng.choice(differences, size=differences.size, replace=True).mean())
    return {
        "n_paired_seeds": int(differences.size),
        "mean_difference_target_minus_baseline": float(differences.mean()),
        "std_difference": float(differences.std(ddof=1) if differences.size > 1 else 0.0),
        "median_difference": float(np.median(differences)),
        "min_difference": float(differences.min()),
        "max_difference": float(differences.max()),
        "bootstrap_repeats": int(bootstrap_repeats),
        "bootstrap_mean_ci95": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "fraction_target_better": float(np.mean(differences < 0.0)),
        "paired_differences": [float(value) for value in differences],
        "lower_is_better": True,
    }


def run_report(input_path: Path, bootstrap_repeats: int = 10000, bootstrap_seed: int = 918273) -> dict[str, Any]:
    source = json.loads(input_path.read_text())
    rows = list(source["records"])
    comparisons = {
        "joint_vs_predictor_weight_1": {
            "target": ("both", 1.0, 0.01),
            "baseline": ("predictor", 1.0, 0.0),
        },
        "joint_vs_none": {
            "target": ("both", 1.0, 0.01),
            "baseline": ("none", 0.0, 0.0),
        },
        "alignment_weight_0.01_vs_none": {
            "target": ("alignment", 0.0, 0.01),
            "baseline": ("none", 0.0, 0.0),
        },
        "predictor_weight_1_vs_none": {
            "target": ("predictor", 1.0, 0.0),
            "baseline": ("none", 0.0, 0.0),
        },
    }
    result: dict[str, Any] = {
        "protocol": {
            "input": str(input_path),
            "paired_unit": "same random seed across conditions",
            "bootstrap_unit": "resample paired seed-level differences",
            "bootstrap_repeats": int(bootstrap_repeats),
            "bootstrap_seed": int(bootstrap_seed),
            "all_metrics_lower_is_better": True,
            "synthetic_vector_world_only": True,
        },
        "comparisons": {},
        "claim_boundary": [
            "uncertainty is over the observed random-seed differences, not a population-level guarantee",
            "the experiment is synthetic mechanism evidence, not LeWM or PushT performance evidence",
            "paired differences do not establish causality beyond this controlled training protocol",
        ],
    }
    for name, specification in comparisons.items():
        target = specification["target"]
        baseline = specification["baseline"]
        result["comparisons"][name] = {
            "target": {"mode": target[0], "predictor_weight": target[1], "alignment_weight": target[2]},
            "baseline": {"mode": baseline[0], "predictor_weight": baseline[1], "alignment_weight": baseline[2]},
            "metrics": {
                metric: _summarize_difference(
                    _paired_differences(rows, target, baseline, metric),
                    bootstrap_repeats=bootstrap_repeats,
                    seed=bootstrap_seed + index,
                )
                for index, metric in enumerate(METRICS)
            },
        }
    if not _finite(result):
        raise ValueError("inference report contains non-finite values")
    return result


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Paired inference summary for the 10-seed joint ablation",
        "",
        "All differences are target minus baseline on the same random seed. Negative values favor the target because every reported metric is an error or misalignment measure.",
        "",
        "| comparison | metric | n | mean difference | 95% bootstrap CI | target better |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for comparison, values in report["comparisons"].items():
        for metric, summary in values["metrics"].items():
            ci = summary["bootstrap_mean_ci95"]
            lines.append(
                f"| {comparison} | {metric} | {summary['n_paired_seeds']} | "
                f"{summary['mean_difference_target_minus_baseline']:.6g} | "
                f"[{ci[0]:.6g}, {ci[1]:.6g}] | {summary['fraction_target_better']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "The bootstrap resamples paired random-seed differences and is not a population-level confidence guarantee. This is synthetic mechanism evidence only; it does not establish LeWM/PushT performance, online planning, or causality.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=918273)
    args = parser.parse_args()
    report = run_report(args.input, args.bootstrap_repeats, args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
