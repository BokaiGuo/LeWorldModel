"""Summarize physical-state observability across action normalizations."""

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


def run_report(paths: list[Path], output: Path) -> dict[str, Any]:
    reports = [json.loads(path.read_text()) for path in paths]
    if not reports:
        raise ValueError("at least one report is required")
    rows = []
    for path, report in zip(paths, reports):
        summary = report["crossfit"]["summary"]["by_horizon"]
        mode = report["protocol"]["action_normalization"]["mode"]
        rows.append(
            {
                "report": str(path),
                "normalization": mode,
                "episodes": int(report["source"]["episodes"]),
                "windows": int(report["source"]["windows"]),
                "h6_delta_mse": summary["6"]["predicted_minus_true_mse"],
                "h8_delta_mse": summary["8"]["predicted_minus_true_mse"],
                "h8_object_x_delta_rmse": summary["8"]["predicted_minus_true_rmse_by_state"][2],
                "h8_object_angle_delta_rmse": summary["8"]["predicted_minus_true_rmse_by_state"][4],
            }
        )
    report = {
        "artifact": "external_frozen_lewm_physical_observability_normalization_robustness",
        "status": "complete",
        "protocol": {
            "reports": [str(path) for path in paths],
            "comparison": "same episodes, windows, held-out probes, state scaling, ridge, and bootstrap protocol; action normalization varies",
        },
        "rows": rows,
        "claim_boundary": [
            "The aggregate predicted-minus-true physical-state decode contrast at h=6 and h=8 is positive under every tested action normalization.",
            "This is robust frozen observability evidence, not physical counterfactual prediction, causal control, online planning, or task success.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Robustness of external physical-state observability",
        "",
        "Only action normalization varies; the public episodes, windows, held-out probe, state scaling, ridge, and bootstrap protocol are fixed.",
        "",
        "| normalization | h=6 predicted − true MSE | h=8 predicted − true MSE | h=8 object-x delta RMSE | h=8 object-angle delta RMSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        fmt = lambda value: f"{value['mean']:.6g} [{value['ci95'][0]:.6g}, {value['ci95'][1]:.6g}]"
        lines.append(
            f"| {row['normalization']} | {fmt(row['h6_delta_mse'])} | {fmt(row['h8_delta_mse'])} | "
            f"{fmt(row['h8_object_x_delta_rmse'])} | {fmt(row['h8_object_angle_delta_rmse'])} |"
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
