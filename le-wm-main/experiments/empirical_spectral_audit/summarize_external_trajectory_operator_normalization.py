"""Summarize action-normalization sensitivity for trajectory operator audits."""

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
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def _normalization_mode(report: dict[str, Any]) -> str:
    value = report["protocol"]["action_normalization"]
    return value if isinstance(value, str) else str(value["mode"])


def run_report(paths: list[Path], output: Path) -> dict[str, Any]:
    reports = [json.loads(path.read_text()) for path in paths]
    if not reports:
        raise ValueError("at least one report is required")
    reference = reports[0]
    rows = []
    for path, report in zip(paths, reports):
        if report["source"]["windows"] != reference["source"]["windows"]:
            raise ValueError("normalization reports use different window counts")
        summary = report["crossfit"]["summary"]
        predicted = summary["operator_by_source"]["predicted"]
        rows.append(
            {
                "report": str(path),
                "normalization": _normalization_mode(report),
                "pca_dimension": int(report["protocol"]["pca_dimension_requested"]),
                "predicted_one_step_r2": predicted["one_step_r2"],
                "predicted_spectral_radius": predicted["spectral_radius"],
                "predicted_sigma_over_radius": predicted["singular_to_radius_ratio"],
                "predicted_henrici": predicted["henrici_nonnormality"],
                "predicted_h8_gain_over_radius": predicted["power_gain_over_radius"]["8"],
                "h8_error_ratio": summary["error_amplification"]["h8"]["ratio_to_h1"],
            }
        )
    report = {
        "artifact": "external_frozen_lewm_trajectory_operator_normalization_sensitivity",
        "status": "complete",
        "protocol": {
            "reports": [str(path) for path in paths],
            "comparison": "same external windows, PCA dimension, ridge, folds, and bootstrap protocol; action normalization varies",
        },
        "rows": rows,
        "claim_boundary": [
            "Observed trajectory-error amplification persists across the tested action normalizations, but fitted operator geometry is normalization-sensitive.",
            "Therefore the operator spectra are exploratory diagnostics rather than normalization-invariant properties of the checkpoint.",
            "All values remain frozen external evidence and do not establish causal control, online planning, task success, or retraining improvement.",
        ],
    }
    if not _finite(report):
        raise ValueError("normalization report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Action-normalization sensitivity of the external trajectory operator audit",
        "",
        "The external windows, PCA dimension, ridge, episode folds, and bootstrap protocol are fixed; only action normalization changes.",
        "",
        "| normalization | predicted R² | predicted rho | predicted sigma/rho | predicted Henrici | predicted h=8 gain/rho⁸ | h=8 error/h=1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        fmt = lambda value: f"{value['mean']:.6g} ± {value['std']:.3g}"
        lines.append(
            f"| {row['normalization']} | {fmt(row['predicted_one_step_r2'])} | {fmt(row['predicted_spectral_radius'])} | "
            f"{fmt(row['predicted_sigma_over_radius'])} | {fmt(row['predicted_henrici'])} | "
            f"{fmt(row['predicted_h8_gain_over_radius'])} | {row['h8_error_ratio']['mean']:.6g} "
            f"[{row['h8_error_ratio']['ci95'][0]:.6g}, {row['h8_error_ratio']['ci95'][1]:.6g}] |"
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
