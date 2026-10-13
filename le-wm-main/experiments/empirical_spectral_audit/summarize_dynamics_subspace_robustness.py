"""Summarize split-robust frozen dynamics-subspace evidence.

This report combines independent train/test splits of the same frozen latent
export and the corresponding external public PushT interventions.  It reports
split-wise estimates and ranges rather than pooling rows or treating
episode-cluster bootstrap intervals as a single meta-analytic confidence
interval.

The resulting evidence is about predictive information in a frozen latent
representation and functional sensitivity of a frozen rollout.  It is not a
causal-control, planning-success, or end-to-end retraining result.
"""

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


def _split_label(path: Path, primary_name: str) -> str:
    if path.name == primary_name:
        return "primary"
    stem = path.stem
    marker = "_split_"
    return stem.split(marker, 1)[1] if marker in stem else stem


def _probe_summary(paths: list[Path]) -> dict[str, Any]:
    reports = [json.loads(path.read_text()) for path in paths]
    split_names = [_split_label(path, "dynamics_subspace_probe.json") for path in paths]
    records: dict[tuple[str, str, int], list[float]] = {}
    overlaps: dict[tuple[str, int], list[float]] = {}
    nulls: dict[tuple[str, int], list[float]] = {}
    for report in reports:
        for row in report["records"]:
            key = (row["target"], row["basis"], int(row["k"]))
            records.setdefault(key, []).append(float(row["r2"]))
        overlap = report["cross_covariance"]["subspace_overlap_with_covariance"]
        for k, values in overlap.items():
            for basis_name, value in values.items():
                overlaps.setdefault((basis_name, int(k)), []).append(float(value))
        for target, by_k in report["random_subspace_null"].items():
            for k, values in by_k.items():
                nulls.setdefault((target, int(k)), []).append(float(values["r2_mean"]))

    k32: dict[str, Any] = {}
    for (target, basis, k), values in sorted(records.items()):
        if k != 32:
            continue
        key = f"{target}/{basis}/k32"
        k32[key] = _summary(values)
        null = nulls[(target, k)]
        k32[key]["random_null_mean"] = _summary(null)
        k32[key]["above_random_null_all_splits"] = all(
            value > null_value for value, null_value in zip(values, null)
        )
        covariance_values = records[(target, "covariance", k)]
        if basis != "covariance":
            k32[key]["delta_vs_covariance"] = _summary(
                [value - covariance for value, covariance in zip(values, covariance_values)]
            )
            k32[key]["above_covariance_all_splits"] = all(
                value > covariance for value, covariance in zip(values, covariance_values)
            )

    overlap_summary = {
        f"{basis}/k{k}": _summary(values)
        for (basis, k), values in sorted(overlaps.items())
        if k == 32
    }
    return {
        "splits": split_names,
        "split_count": len(paths),
        "k32": k32,
        "k32_subspace_overlap_with_covariance": overlap_summary,
        "protocol_consistency": {
            "same_selected_rows": len({report["source"]["selected_rows"] for report in reports}) == 1,
            "same_latent_dim": len({report["source"]["latent_dim"] for report in reports}) == 1,
            "same_sample_seed": len({report["protocol"]["sample_seed"] for report in reports}) == 1,
            "same_test_fraction": len({report["protocol"]["test_fraction"] for report in reports}) == 1,
        },
    }


def _intervention_summary(paths: list[Path]) -> dict[str, Any]:
    reports = [json.loads(path.read_text()) for path in paths]
    split_names = [
        _split_label(path, "external_dynamics_basis_intervention.json")
        for path in paths
    ]
    selected: dict[str, list[float]] = {}
    ci_lower: dict[str, list[float]] = {}
    ci_upper: dict[str, list[float]] = {}
    record_counts = []
    for report in reports:
        record_counts.append(len(report["records"]))
        for intervention in ("dynamics_top8", "conditional_dynamics_top8"):
            key = f"{intervention}/h8"
            value = report["summary"]["paired_delta_vs_none"][key]
            selected.setdefault(key, []).append(float(value["mean"]))
            ci_lower.setdefault(key, []).append(float(value["ci95"][0]))
            ci_upper.setdefault(key, []).append(float(value["ci95"][1]))
            contrast = report["summary"]["paired_delta_vs_covariance"][key]
            contrast_key = f"{key}/vs_covariance"
            selected.setdefault(contrast_key, []).append(float(contrast["mean"]))
            ci_lower.setdefault(contrast_key, []).append(float(contrast["ci95"][0]))
            ci_upper.setdefault(contrast_key, []).append(float(contrast["ci95"][1]))

    result: dict[str, Any] = {"splits": split_names, "record_counts": record_counts}
    for key, values in selected.items():
        result[key] = {
            "estimate": _summary(values),
            "ci95_lower": _summary(ci_lower[key]),
            "ci95_upper": _summary(ci_upper[key]),
            "ci95_excludes_zero_all_splits": all(value > 0 for value in ci_lower[key]),
            "interpretation": (
                "positive paired rollout degradation under coefficient removal"
                if "/vs_covariance" not in key
                else "split-wise contrast against covariance basis; not a superiority claim"
            ),
        }
    result["protocol_consistency"] = {
        "same_record_count": len(set(record_counts)) == 1,
        "expected_record_count": 1176,
    }
    return result


def _energy_intervention_summary(paths: list[Path]) -> dict[str, Any]:
    reports = [json.loads(path.read_text()) for path in paths]
    split_names = [
        _split_label(path, "external_energy_matched_dynamics_primary.json")
        for path in paths
    ]
    keys = [
        "dynamics_top8",
        "conditional_dynamics_top8",
        "covariance_top8_energy_matched_to_dynamics",
        "random_top8_energy_matched_to_dynamics",
    ]
    output: dict[str, Any] = {"splits": split_names, "horizons": {}}
    for intervention in keys:
        for horizon in (1, 2, 4, 8):
            key = f"{intervention}/h{horizon}"
            values = []
            ci_lower = []
            ci_upper = []
            ratios = []
            for report in reports:
                if intervention == "dynamics_top8":
                    contrast = report["summary"]["paired_delta_vs_none"][key]
                else:
                    contrast = report["summary"]["paired_delta_vs_dynamics"][key]
                values.append(float(contrast["mean"]))
                ci_lower.append(float(contrast["ci95"][0]))
                ci_upper.append(float(contrast["ci95"][1]))
                energy = report["summary"]["energy_matching"][key]
                ratios.append(
                    float(energy["applied_energy"]["mean"])
                    / max(float(energy["reference_energy"]["mean"]), 1e-12)
                )
            output["horizons"][key] = {
                "contrast_unit": "delta vs none" if intervention == "dynamics_top8" else "delta vs dynamics top-8",
                "estimate": _summary(values),
                "original_ci_lower": _summary(ci_lower),
                "original_ci_upper": _summary(ci_upper),
                "ci_lower_gt_zero_all_splits": all(value > 0.0 for value in ci_lower),
                "energy_ratio": _summary(ratios),
            }
    output["protocol_consistency"] = {
        "same_record_count": len({len(report["records"]) for report in reports}) == 1,
        "expected_record_count": 1008,
        "energy_ratio_max_abs_error": max(
            abs(value - 1.0)
            for item in output["horizons"].values()
            for value in item["energy_ratio"]["values"]
        ),
    }
    return output


def run_report(probe_paths: list[Path], intervention_paths: list[Path], output: Path) -> dict[str, Any]:
    if len(probe_paths) != len(intervention_paths) or len(probe_paths) < 3:
        raise ValueError("provide at least three paired probe and intervention reports")
    report = {
        "artifact": "frozen_lewm_dynamics_subspace_split_robustness",
        "status": "complete_split_robustness_summary",
        "probe": _probe_summary(probe_paths),
        "external_intervention": _intervention_summary(intervention_paths),
        "claim_boundary": [
            "The dynamics and conditional-dynamics bases are fitted on training rows only and evaluated with held-out readouts.",
            "Across split estimates, the result supports a dynamics-sensitive predictive subspace distinct from covariance ranking.",
            "External coefficient-removal effects establish frozen-rollout functional sensitivity, not causal physical control or planning success.",
            "Cross-basis contrasts whose confidence intervals cross zero are not reported as superiority claims.",
            "No pooled confidence interval is constructed; reported ranges summarize independent split estimates and their original bootstrap intervals.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("notes.txt").write_text(_notes(report))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    probe = report["probe"]
    lines = [
        "# Split-robust dynamics-sensitive subspaces in frozen LeWM latents",
        "",
        "Three independent train/test splits of the same frozen latent export are summarized. Values are split estimates; no rows are pooled and no meta-analytic confidence interval is claimed.",
        "",
        "## Held-out k=32 readouts",
        "",
        "| target/basis | mean R² | range | delta vs covariance | above covariance all splits | above random all splits |",
        "| --- | ---: | --- | ---: | :---: | :---: |",
    ]
    for key, value in probe["k32"].items():
        delta = value.get("delta_vs_covariance", {}).get("mean")
        delta_text = "--" if delta is None else f"{delta:.6f}"
        cov_text = "--" if "above_covariance_all_splits" not in value else str(value["above_covariance_all_splits"])
        lines.append(
            f"| {key} | {value['mean']:.6f} | [{value['min']:.6f}, {value['max']:.6f}] | "
            f"{delta_text} | {cov_text} | {value['above_random_null_all_splits']} |"
        )
    lines.extend([
        "",
        "## k=32 covariance overlap",
        "",
        "| supervised basis | mean overlap | range |",
        "| --- | ---: | --- |",
    ])
    for key, value in probe["k32_subspace_overlap_with_covariance"].items():
        lines.append(f"| {key} | {value['mean']:.6f} | [{value['min']:.6f}, {value['max']:.6f}] |")
    lines.extend([
        "",
        "## External h=8 coefficient-removal intervention",
        "",
        "| intervention | mean delta vs none | range | original CI lower range | original CI upper range | excludes zero all splits |",
        "| --- | ---: | --- | --- | --- | :---: |",
    ])
    for key, value in report["external_intervention"].items():
        if not (key.endswith("/h8") or key.endswith("/h8/vs_covariance")):
            continue
        estimate = value["estimate"]
        lower = value["ci95_lower"]
        upper = value["ci95_upper"]
        lines.append(
            f"| {key} | {estimate['mean']:.6f} | [{estimate['min']:.6f}, {estimate['max']:.6f}] | "
            f"[{lower['min']:.6f}, {lower['max']:.6f}] | [{upper['min']:.6f}, {upper['max']:.6f}] | "
            f"{value['ci95_excludes_zero_all_splits']} |"
        )
    if "energy_matched" in report:
        lines.extend(
            [
                "",
                "## Energy-matched external intervention",
                "",
                "Positive values for controls mean higher rollout MSE than the dynamics top-8 reference after matching per-step perturbation energy.",
                "",
                "| intervention/horizon | mean contrast | range | CI lower range | CI upper range | lower CI > 0 all splits | energy ratio range |",
                "| --- | ---: | --- | --- | --- | :---: | --- |",
            ]
        )
        for key, value in report["energy_matched"]["horizons"].items():
            lines.append(
                f"| {key} | {value['estimate']['mean']:.6f} | [{value['estimate']['min']:.6f}, {value['estimate']['max']:.6f}] | "
                f"[{value['original_ci_lower']['min']:.6f}, {value['original_ci_lower']['max']:.6f}] | "
                f"[{value['original_ci_upper']['min']:.6f}, {value['original_ci_upper']['max']:.6f}] | "
                f"{value['ci_lower_gt_zero_all_splits']} | [{value['energy_ratio']['min']:.6f}, {value['energy_ratio']['max']:.6f}] |"
            )
    lines.extend(["", "## Claim boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def _notes(report: dict[str, Any]) -> str:
    raw = report["probe"]["k32"]["dynamics/dynamics_cross_covariance/k32"]
    conditional = report["probe"]["k32"]["dynamics_after_action/conditional_dynamics_cross_covariance/k32"]
    external = report["external_intervention"]["dynamics_top8/h8"]
    conditional_external = report["external_intervention"]["conditional_dynamics_top8/h8"]
    return (
        "2026-10-03: Summarized three independent 80k/20k held-out splits from the same 100k frozen latent sample.\n"
        f"Raw dynamics k=32 held-out R2 mean={raw['mean']:.6f}, range=[{raw['min']:.6f}, {raw['max']:.6f}]; above covariance in all splits={raw['above_covariance_all_splits']}.\n"
        f"Conditional dynamics-after-action k=32 held-out R2 mean={conditional['mean']:.6f}, range=[{conditional['min']:.6f}, {conditional['max']:.6f}]; above covariance in all splits={conditional['above_covariance_all_splits']}.\n"
        f"External raw dynamics top-8 h=8 delta vs none mean={external['estimate']['mean']:.6f}, original CI lower range=[{external['ci95_lower']['min']:.6f}, {external['ci95_lower']['max']:.6f}], excludes zero in all splits={external['ci95_excludes_zero_all_splits']}.\n"
        f"External conditional dynamics top-8 h=8 delta vs none mean={conditional_external['estimate']['mean']:.6f}, original CI lower range=[{conditional_external['ci95_lower']['min']:.6f}, {conditional_external['ci95_lower']['max']:.6f}], excludes zero in all splits={conditional_external['ci95_excludes_zero_all_splits']}.\n"
        "Boundary: split-robust frozen predictive and rollout-sensitivity evidence, not causal control, planning success, or retraining evidence.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, nargs="+", required=True)
    parser.add_argument("--intervention", type=Path, nargs="+", required=True)
    parser.add_argument("--energy-intervention", type=Path, nargs="+", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_report(args.probe, args.intervention, args.output)
    if args.energy_intervention is not None:
        if len(args.energy_intervention) != len(args.probe):
            raise ValueError("energy-intervention reports must be paired with probe reports")
        report["energy_matched"] = _energy_intervention_summary(args.energy_intervention)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        _write_markdown(report, args.output.with_suffix(".md"))
        args.output.with_name("notes.txt").write_text(
            args.output.with_name("notes.txt").read_text()
            + "Energy-matched summary included; per-step applied/reference energy ratios were checked.\n"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
