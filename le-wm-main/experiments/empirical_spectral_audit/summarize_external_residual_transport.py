"""Aggregate external residual-error transport across basis splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary values must be finite and non-empty")
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "values": [float(value) for value in array],
    }


def run_report(paths: list[Path], output: Path) -> dict[str, Any]:
    if len(paths) < 3:
        raise ValueError("at least three external transport reports are required")
    reports = [json.loads(path.read_text()) for path in paths]
    splits = [path.stem.replace("external_residual_transport_", "") for path in paths]
    bases = (
        "covariance_basis",
        "dynamics_basis",
        "residual_basis",
        "conditional_dynamics_basis",
        "residual_covariance_basis",
        "random_basis",
    )
    bands = ("top8", "middle8_32", "tail128_192")
    horizons = (1, 2, 4, 8)
    share: dict[str, Any] = {}
    for basis in bases:
        for band in bands:
            for horizon in horizons:
                key = f"{basis}/{band}/h{horizon}"
                values = [float(report["summary"]["energy_share"][key]["mean"]) for report in reports]
                value = {"estimate": _summary(values)}
                if basis != "covariance_basis":
                    deltas = [
                        float(report["summary"]["paired_delta_vs_covariance"][key]["mean"])
                        for report in reports
                    ]
                    lower = [
                        float(report["summary"]["paired_delta_vs_covariance"][key]["ci95"][0])
                        for report in reports
                    ]
                    upper = [
                        float(report["summary"]["paired_delta_vs_covariance"][key]["ci95"][1])
                        for report in reports
                    ]
                    value["delta_vs_covariance"] = _summary(deltas)
                    value["original_ci_lower"] = _summary(lower)
                    value["original_ci_upper"] = _summary(upper)
                    value["positive_lower_ci_all_splits"] = all(item > 0.0 for item in lower)
                    value["negative_upper_ci_all_splits"] = all(item < 0.0 for item in upper)
                share[key] = value

    record_counts = [len(report["records"]) for report in reports]
    output_report = {
        "artifact": "external_lewm_residual_error_transport_split_robustness",
        "status": "complete_external_residual_transport_summary",
        "splits": splits,
        "split_count": len(paths),
        "energy_share": share,
        "protocol_consistency": {
            "same_record_count": len(set(record_counts)) == 1,
            "expected_record_count": 3024,
            "record_counts": record_counts,
            "same_horizons": len({tuple(report["protocol"]["horizons"]) for report in reports}) == 1,
            "same_random_seed": len({report["protocol"]["random_basis_seed"] for report in reports}) == 1,
        },
        "claim_boundary": [
            "The residual and dynamics bases are transported without refitting from a separate frozen latent export.",
            "A source-domain residual readout advantage does not imply cross-source error-energy concentration; the external transport result is reported separately.",
            "This is unmodified rollout error geometry, not causal error relocation, physical control, or planning success.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_report, indent=2, sort_keys=True) + "\n")
    _write_markdown(output_report, output.with_suffix(".md"))
    output.with_name("notes.txt").write_text(_notes(output_report))
    return output_report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Split-robust external residual error transport",
        "",
        "Residual bases fitted on three independent frozen-latent splits are transported to the same public PushT windows. Values are split estimates; no external rows are pooled.",
        "",
        "## Top-8 error-energy share",
        "",
        "| basis | h | mean share | range | delta vs covariance | original CI lower range | original CI upper range | stable sign |",
        "| --- | ---: | ---: | --- | ---: | --- | --- | --- |",
    ]
    for basis in (
        "covariance_basis",
        "dynamics_basis",
        "residual_basis",
        "conditional_dynamics_basis",
        "residual_covariance_basis",
        "random_basis",
    ):
        for horizon in (1, 2, 4, 8):
            key = f"{basis}/top8/h{horizon}"
            value = report["energy_share"][key]
            if basis == "covariance_basis":
                lines.append(
                    f"| {basis} | {horizon} | {value['estimate']['mean']:.6f} | "
                    f"[{value['estimate']['min']:.6f}, {value['estimate']['max']:.6f}] | -- | -- | -- | -- |"
                )
            else:
                positive = value["positive_lower_ci_all_splits"]
                negative = value["negative_upper_ci_all_splits"]
                sign = "positive" if positive else "negative" if negative else "crosses zero"
                lines.append(
                    f"| {basis} | {horizon} | {value['estimate']['mean']:.6f} | "
                    f"[{value['estimate']['min']:.6f}, {value['estimate']['max']:.6f}] | "
                    f"{value['delta_vs_covariance']['mean']:.6f} | "
                    f"[{value['original_ci_lower']['min']:.6f}, {value['original_ci_lower']['max']:.6f}] | "
                    f"[{value['original_ci_upper']['min']:.6f}, {value['original_ci_upper']['max']:.6f}] | {sign} |"
                )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def _notes(report: dict[str, Any]) -> str:
    residual = report["energy_share"]["residual_basis/top8/h8"]
    residual_cov = report["energy_share"]["residual_covariance_basis/top8/h8"]
    dynamics = report["energy_share"]["dynamics_basis/top8/h8"]
    return (
        "2026-10-03: Aggregated external residual-error transport across three basis splits.\n"
        f"At h=8, residual cross-covariance basis delta vs covariance={residual['delta_vs_covariance']['mean']:.6f}; stable negative CI sign={residual['negative_upper_ci_all_splits']}.\n"
        f"At h=8, residual covariance basis delta vs covariance={residual_cov['delta_vs_covariance']['mean']:.6f}; stable positive CI sign={residual_cov['positive_lower_ci_all_splits']}.\n"
        f"At h=8, dynamics basis delta vs covariance={dynamics['delta_vs_covariance']['mean']:.6f}; stable sign is not claimed when the CI crosses zero.\n"
        "Boundary: external error geometry only, not causal error relocation or control.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_report(args.input, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
