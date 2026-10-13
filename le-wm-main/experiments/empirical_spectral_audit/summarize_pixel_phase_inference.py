"""Paired seed inference for each cell of the SIGReg x alignment phase diagram."""

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


def _condition(row: dict[str, Any]) -> tuple[float, str, float]:
    return float(row["sigreg_weight"]), str(row["mode"]), float(row["alignment_weight"])


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
        "mean_difference_target_minus_same_sigreg_baseline": float(values.mean()),
        "std_difference": float(values.std(ddof=1) if values.size > 1 else 0.0),
        "bootstrap_mean_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "fraction_target_lower": float(np.mean(values < 0.0)),
        "paired_differences": [float(value) for value in values],
    }


def run_report(input_path: Path, repeats: int = 10000, seed: int = 271828) -> dict[str, Any]:
    source = json.loads(input_path.read_text())
    rows = list(source["records"])
    sigregs = sorted({float(row["sigreg_weight"]) for row in rows})
    interventions = sorted(
        {
            _condition(row)[1:]
            for row in rows
            if str(row["mode"]) != "none"
        }
    )
    report: dict[str, Any] = {
        "protocol": {
            "input": str(input_path),
            "paired_unit": "same seed within the same SIGReg weight",
            "bootstrap_unit": "paired seed-level differences",
            "bootstrap_repeats": int(repeats),
            "bootstrap_seed": int(seed),
            "synthetic_pixel_world_only": True,
        },
        "comparisons": {},
        "claim_boundary": [
            "each intervention is compared only with the no-alignment condition at the same SIGReg weight",
            "intervals quantify observed seed variation, not a population-level guarantee",
            "lower commutator is not treated as evidence of better rollout or composition",
        ],
    }
    comparison_index = 0
    for sigreg in sigregs:
        baseline = (sigreg, "none", 0.0)
        for mode, weight in interventions:
            target = (sigreg, mode, weight)
            name = f"sigreg_{sigreg:g}_{mode}_{weight:g}_vs_none"
            report["comparisons"][name] = {
                "target": {"sigreg_weight": sigreg, "mode": mode, "alignment_weight": weight},
                "baseline": {"sigreg_weight": sigreg, "mode": "none", "alignment_weight": 0.0},
                "metrics": {
                    metric: _summary(
                        _paired(rows, target, baseline, metric),
                        repeats=repeats,
                        seed=seed + comparison_index * 17 + metric_index,
                    )
                    for metric_index, metric in enumerate(METRICS)
                },
            }
            comparison_index += 1
    if not _finite(report):
        raise ValueError("phase inference report contains non-finite values")
    return report


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Paired inference for the SIGReg × alignment phase diagram",
        "",
        "Each intervention is compared with the no-alignment baseline at the same SIGReg weight. Negative differences lower the corresponding error or commutator.",
        "",
        "| comparison | metric | n | mean difference | 95% bootstrap CI | target lower |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for comparison, values in report["comparisons"].items():
        for metric, summary in values["metrics"].items():
            ci = summary["bootstrap_mean_ci95"]
            lines.append(
                f"| {comparison} | {metric} | {summary['n_paired_seeds']} | "
                f"{summary['mean_difference_target_minus_same_sigreg_baseline']:.6g} | "
                f"[{ci[0]:.6g}, {ci[1]:.6g}] | {summary['fraction_target_lower']:.2f} |"
            )
    lines.extend(["", "## Boundary", "", "This is synthetic learned pixel-world evidence only."])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=271828)
    args = parser.parse_args()
    report = run_report(args.input, args.bootstrap_repeats, args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
