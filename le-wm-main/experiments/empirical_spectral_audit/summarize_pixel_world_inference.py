"""Paired seed-level inference for the learned pixel-world alignment sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = {
    "test_one_step_mse": lambda row: float(row["test_one_step_mse"]),
    "rollout_h8_mse": lambda row: float(row["rollout_mse"]["8"]),
    "composition_error": lambda row: float(row["action_composition_error_mean"]),
    "global_commutator": lambda row: float(row["diagnostics"]["global_commutator"]),
    "conditional_commutator": lambda row: float(row["diagnostics"]["conditional_commutator"]),
}


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


def _condition(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["mode"]), float(row["alignment_weight"])


def _paired_differences(
    rows: list[dict[str, Any]], target: tuple[str, float], baseline: tuple[str, float], metric: str
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


def _summary(differences: np.ndarray, repeats: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [rng.choice(differences, size=differences.size, replace=True).mean() for _ in range(int(repeats))],
        dtype=np.float64,
    )
    return {
        "n_paired_seeds": int(differences.size),
        "mean_difference_target_minus_baseline": float(differences.mean()),
        "std_difference": float(differences.std(ddof=1) if differences.size > 1 else 0.0),
        "bootstrap_repeats": int(repeats),
        "bootstrap_mean_ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "fraction_target_lower": float(np.mean(differences < 0.0)),
        "paired_differences": [float(value) for value in differences],
    }


def run_report(input_path: Path, repeats: int = 10000, seed: int = 104729) -> dict[str, Any]:
    source = json.loads(input_path.read_text())
    rows = list(source["records"])
    comparisons = {
        "global_0.01_vs_none": {"target": ("global", 0.01), "baseline": ("none", 0.0)},
        "conditional_0.01_vs_none": {"target": ("conditional", 0.01), "baseline": ("none", 0.0)},
        "global_0.1_vs_none": {"target": ("global", 0.1), "baseline": ("none", 0.0)},
        "conditional_0.1_vs_none": {"target": ("conditional", 0.1), "baseline": ("none", 0.0)},
    }
    report: dict[str, Any] = {
        "protocol": {
            "input": str(input_path),
            "paired_unit": "same generated split and initialization seed",
            "bootstrap_unit": "resample paired seed-level differences",
            "bootstrap_repeats": int(repeats),
            "bootstrap_seed": int(seed),
            "learned_pixel_encoder": True,
            "synthetic_pixel_world_only": True,
        },
        "comparisons": {},
        "claim_boundary": [
            "uncertainty is over the ten observed training seeds, not a population-level guarantee",
            "commutator reduction is a diagnostic consequence and is not assumed to imply lower rollout error",
            "this is synthetic pixel-world evidence, not nonlinear LeWM, PushT, online planning, or causal-control evidence",
        ],
    }
    for comparison, specification in comparisons.items():
        target = specification["target"]
        baseline = specification["baseline"]
        report["comparisons"][comparison] = {
            "target": {"mode": target[0], "alignment_weight": target[1]},
            "baseline": {"mode": baseline[0], "alignment_weight": baseline[1]},
            "metrics": {
                metric: _summary(
                    _paired_differences(rows, target, baseline, metric),
                    repeats=repeats,
                    seed=seed + index,
                )
                for index, metric in enumerate(METRICS)
            },
        }
    if not _finite(report):
        raise ValueError("inference report contains non-finite values")
    return report


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Paired inference for the ten-seed learned pixel-world sweep",
        "",
        "Values are target minus no-intervention baseline on the same seed. Negative values lower the corresponding error or commutator.",
        "",
        "| comparison | metric | n | mean difference | 95% bootstrap CI | target lower |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for comparison, values in report["comparisons"].items():
        for metric, summary in values["metrics"].items():
            ci = summary["bootstrap_mean_ci95"]
            lines.append(
                f"| {comparison} | {metric} | {summary['n_paired_seeds']} | "
                f"{summary['mean_difference_target_minus_baseline']:.6g} | "
                f"[{ci[0]:.6g}, {ci[1]:.6g}] | {summary['fraction_target_lower']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "The paired bootstrap summarizes observed seed variation. It does not establish that lower commutator produces better rollout; the learned pixel benchmark is a synthetic bridge toward, not a substitute for, end-to-end LeWM evidence.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=104729)
    args = parser.parse_args()
    report = run_report(args.input, args.bootstrap_repeats, args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
