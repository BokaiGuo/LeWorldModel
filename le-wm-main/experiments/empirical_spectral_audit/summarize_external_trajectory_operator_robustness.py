"""Summarize PCA-dimension robustness for the external trajectory operator audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


def _normalization_mode(report: dict[str, Any]) -> str:
    value = report["protocol"]["action_normalization"]
    return value if isinstance(value, str) else str(value["mode"])


def run_report(paths: list[Path], output: Path) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one report is required")
    reports = [json.loads(path.read_text()) for path in paths]
    reference = reports[0]
    rows = []
    for path, report in zip(paths, reports):
        if report["source"]["windows"] != reference["source"]["windows"]:
            raise ValueError("robustness reports use different window counts")
        if _normalization_mode(report) != _normalization_mode(reference):
            raise ValueError("robustness reports use different action normalization")
        dimension = int(report["protocol"]["pca_dimension_requested"])
        summary = report["crossfit"]["summary"]
        rows.append(
            {
                "report": str(path),
                "pca_dimension": dimension,
                "windows": int(report["source"]["windows"]),
                "episodes": int(report["source"]["episodes"]),
                "true_one_step_r2": summary["operator_by_source"]["true"]["one_step_r2"],
                "predicted_one_step_r2": summary["operator_by_source"]["predicted"]["one_step_r2"],
                "true_henrici": summary["operator_by_source"]["true"]["henrici_nonnormality"],
                "predicted_henrici": summary["operator_by_source"]["predicted"]["henrici_nonnormality"],
                "true_h8_gain_over_radius": summary["operator_by_source"]["true"]["power_gain_over_radius"]["8"],
                "predicted_h8_gain_over_radius": summary["operator_by_source"]["predicted"]["power_gain_over_radius"]["8"],
                "predicted_minus_true_h8_composition": summary["paired_operator_difference"]["composition_defect/h8"],
                "h8_error_mse": summary["error_amplification"]["h8"]["mse"],
                "h8_error_ratio": summary["error_amplification"]["h8"]["ratio_to_h1"],
            }
        )
    report = {
        "artifact": "external_frozen_lewm_trajectory_operator_pca_robustness",
        "status": "complete",
        "protocol": {
            "reports": [str(path) for path in paths],
            "comparison": "same external windows, action normalization, leave-one-episode-out folds, ridge, and bootstrap protocol; only requested PCA dimension varies",
            "dimensions": [row["pca_dimension"] for row in rows],
        },
        "rows": rows,
        "claim_boundary": [
            "The robustness table tests sensitivity to the reduced-coordinate dimension; it does not turn the diagnostic into full-rank nonlinear Jacobian evidence.",
            "All values remain frozen external trajectory diagnostics and do not establish causal control, online planning, task success, or retraining improvement.",
        ],
    }
    if not _finite(report):
        raise ValueError("robustness report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Robustness of the external trajectory-level operator audit",
        "",
        "Only the requested PCA dimension changes across rows; the external windows, episode folds, ridge, action normalization, and bootstrap protocol are held fixed.",
        "",
        "| PCA dim | true R² | predicted R² | true Henrici | predicted Henrici | true h=8 gain/radius⁸ | predicted h=8 gain/radius⁸ | predicted − true h=8 defect | h=8 error / h=1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        fmt = lambda value: f"{value['mean']:.6g} ± {value['std']:.3g}"
        ci = row["predicted_minus_true_h8_composition"]
        ratio = row["h8_error_ratio"]
        lines.append(
            f"| {row['pca_dimension']} | {fmt(row['true_one_step_r2'])} | {fmt(row['predicted_one_step_r2'])} | "
            f"{fmt(row['true_henrici'])} | {fmt(row['predicted_henrici'])} | {fmt(row['true_h8_gain_over_radius'])} | "
            f"{fmt(row['predicted_h8_gain_over_radius'])} | {ci['mean']:.6g} [{ci['ci95'][0]:.6g}, {ci['ci95'][1]:.6g}] | "
            f"{ratio['mean']:.6g} [{ratio['ci95'][0]:.6g}, {ratio['ci95'][1]:.6g}] |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_report(args.reports, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
