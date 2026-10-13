"""Aggregate external residual-basis intervention results across splits."""

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
        raise ValueError("at least three intervention reports are required")
    reports = [json.loads(path.read_text()) for path in paths]
    splits = [path.stem.replace("external_residual_basis_intervention_", "") for path in paths]
    interventions = (
        "residual_top8",
        "residual_covariance_top8",
        "residual_top8_energy_matched_to_dynamics",
        "residual_covariance_top8_energy_matched_to_dynamics",
    )
    summary: dict[str, Any] = {}
    for intervention in interventions:
        for horizon in (1, 2, 4, 8):
            key = f"{intervention}/h{horizon}"
            delta = [
                float(report["summary"]["paired_delta_vs_dynamics"][key]["mean"])
                for report in reports
            ]
            lower = [
                float(report["summary"]["paired_delta_vs_dynamics"][key]["ci95"][0])
                for report in reports
            ]
            upper = [
                float(report["summary"]["paired_delta_vs_dynamics"][key]["ci95"][1])
                for report in reports
            ]
            energy = [
                float(report["summary"]["energy_matching"][key]["applied_energy"]["mean"])
                / max(float(report["summary"]["energy_matching"][key]["reference_energy"]["mean"]), 1e-12)
                for report in reports
            ]
            summary[key] = {
                "delta_vs_dynamics": _summary(delta),
                "original_ci_lower": _summary(lower),
                "original_ci_upper": _summary(upper),
                "positive_lower_ci_all_splits": all(item > 0.0 for item in lower),
                "negative_upper_ci_all_splits": all(item < 0.0 for item in upper),
                "energy_ratio": _summary(energy),
            }

    record_counts = [len(report["records"]) for report in reports]
    result = {
        "artifact": "external_lewm_residual_basis_intervention_split_robustness",
        "status": "complete_external_residual_basis_intervention_summary",
        "splits": splits,
        "split_count": len(paths),
        "summary": summary,
        "protocol_consistency": {
            "same_record_count": len(set(record_counts)) == 1,
            "expected_record_count": 1344,
            "record_counts": record_counts,
            "energy_ratio_max_abs_error": max(
                abs(item - 1.0)
                for value in summary.values()
                for item in value["energy_ratio"]["values"]
            ),
        },
        "claim_boundary": [
            "Residual-basis interventions are transferred from separate frozen latent fits and evaluated on the same external windows.",
            "Positive or negative paired contrasts describe frozen rollout sensitivity, not causal physical control or planning success.",
            "Energy-matched arms remove perturbation magnitude as a direct confound but do not establish training improvement.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_markdown(result, output.with_suffix(".md"))
    output.with_name("notes.txt").write_text(_notes(result))
    return result


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Split-robust external residual-basis intervention",
        "",
        "Values are paired MSE contrasts against dynamics top-8 removal on the same external windows. Positive values mean the residual-basis intervention is more damaging.",
        "",
        "| intervention | h | mean delta vs dynamics | range | CI lower range | CI upper range | stable sign | energy ratio range |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for key, value in report["summary"].items():
        positive = value["positive_lower_ci_all_splits"]
        negative = value["negative_upper_ci_all_splits"]
        sign = "positive" if positive else "negative" if negative else "crosses zero"
        intervention, horizon = key.split("/h")
        lines.append(
            f"| {intervention} | {horizon} | {value['delta_vs_dynamics']['mean']:.6f} | "
            f"[{value['delta_vs_dynamics']['min']:.6f}, {value['delta_vs_dynamics']['max']:.6f}] | "
            f"[{value['original_ci_lower']['min']:.6f}, {value['original_ci_lower']['max']:.6f}] | "
            f"[{value['original_ci_upper']['min']:.6f}, {value['original_ci_upper']['max']:.6f}] | {sign} | "
            f"[{value['energy_ratio']['min']:.6f}, {value['energy_ratio']['max']:.6f}] |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def _notes(report: dict[str, Any]) -> str:
    residual = report["summary"]["residual_top8/h4"]
    residual_cov = report["summary"]["residual_covariance_top8/h8"]
    matched = report["summary"]["residual_top8_energy_matched_to_dynamics/h4"]
    return (
        "2026-10-03: Aggregated residual-basis external functional interventions across three splits.\n"
        f"Raw residual top-8 vs dynamics at h=4: delta={residual['delta_vs_dynamics']['mean']:.6f}, positive CI in all splits={residual['positive_lower_ci_all_splits']}.\n"
        f"Raw residual-covariance top-8 vs dynamics at h=8: delta={residual_cov['delta_vs_dynamics']['mean']:.6f}, negative CI in all splits={residual_cov['negative_upper_ci_all_splits']}.\n"
        f"Energy-matched residual top-8 vs dynamics at h=4: delta={matched['delta_vs_dynamics']['mean']:.6f}, positive CI in all splits={matched['positive_lower_ci_all_splits']}.\n"
        "Boundary: frozen latent rollout sensitivity only.\n"
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
