"""Summarize the external physical-displacement probe across action scales."""

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


def _metric(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean": float(value["mean"]),
        "ci95": [float(item) for item in value["ci95"]],
        "n_windows": int(value["n_windows"]),
        "clusters": int(value["clusters"]),
    }


def run_report(paths: list[Path], output: Path) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two normalization reports are required")
    reports = [json.loads(path.read_text()) for path in paths]
    reference = reports[0]
    reference_protocol = reference["protocol"]
    reference_source = reference["source"]
    selected_keys = (
        "dynamics_basis/top8",
        "residual_covariance_basis/middle32_128",
        "random_basis/middle32_128",
    )
    horizons = (1, 4, 8)
    rows: list[dict[str, Any]] = []
    for path, report in zip(paths, reports):
        source = report["source"]
        protocol = report["protocol"]
        if int(source["episodes"]) != int(reference_source["episodes"]):
            raise ValueError("normalization reports use different episode counts")
        if int(source["windows"]) != int(reference_source["windows"]):
            raise ValueError("normalization reports use different window counts")
        if protocol["horizons"] != reference_protocol["horizons"]:
            raise ValueError("normalization reports use different horizons")
        summary = report["crossfit"]["summary"]["by_basis_band_horizon"]
        normalization = protocol["action_normalization"]["mode"]
        for key_prefix in selected_keys:
            for horizon in horizons:
                value = summary[f"{key_prefix}/h{horizon}"]
                rows.append(
                    {
                        "report": str(path),
                        "normalization": normalization,
                        "basis_band": key_prefix,
                        "horizon": horizon,
                        "action_only_r2": _metric(value["action_only_r2"]),
                        "true_increment_r2": _metric(value["true_increment_r2"]),
                        "predicted_increment_r2": _metric(value["predicted_increment_r2"]),
                        "predicted_incremental_r2_over_action": _metric(
                            value["predicted_incremental_r2"]
                        ),
                    }
                )
    report = {
        "artifact": "external_frozen_lewm_physical_displacement_normalization_robustness",
        "status": "complete",
        "protocol": {
            "reports": [str(path) for path in paths],
            "comparison": "same external episodes, windows, transferred bases, held-out probes, target scaling, ridge, and episode bootstrap; only action normalization varies",
            "selected_basis_bands": list(selected_keys),
            "horizons": list(horizons),
        },
        "source": {
            "episodes": int(reference_source["episodes"]),
            "windows": int(reference_source["windows"]),
        },
        "rows": rows,
        "claim_boundary": [
            "The external physical-displacement readout is a frozen predictive diagnostic, not a causal action or physical-control test.",
            "The action-only baseline uses observed future-action summaries; incremental R2 therefore measures added predictive value, not an intervention effect.",
            "Dimension-matched random bands are retained for band comparisons, and no cross-band ranking is claimed from different feature dimensions.",
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
        "# Robustness of external physical-displacement dynamics probes",
        "",
        "Only action normalization varies; episodes, windows, transferred bases, held-out probes, target scaling, ridge, and episode-cluster bootstrap are fixed.",
        "",
        "| normalization | basis/band | h | action-only R2 | true increment R2 | predicted increment R2 | predicted incremental R2 over action |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    fmt = lambda value: f"{value['mean']:.6g} [{value['ci95'][0]:.6g}, {value['ci95'][1]:.6g}]"
    for row in report["rows"]:
        lines.append(
            f"| {row['normalization']} | {row['basis_band']} | {row['horizon']} | "
            f"{fmt(row['action_only_r2'])} | {fmt(row['true_increment_r2'])} | "
            f"{fmt(row['predicted_increment_r2'])} | {fmt(row['predicted_incremental_r2_over_action'])} |"
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
