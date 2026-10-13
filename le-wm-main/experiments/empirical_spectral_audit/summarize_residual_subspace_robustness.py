"""Aggregate split-robust residual-subspace diagnostics."""

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
        raise ValueError("at least three residual probe reports are required")
    reports = [json.loads(path.read_text()) for path in paths]
    splits = [path.parent.name for path in paths]
    k32: dict[str, Any] = {}
    targets = ("residual", "dynamics", "predicted_dynamics", "dynamics_after_action")
    bases = (
        "covariance",
        "dynamics_cross_covariance",
        "residual_cross_covariance",
        "conditional_dynamics_cross_covariance",
    )
    for target in targets:
        for basis in bases:
            values = [
                float(row["r2"])
                for report in reports
                for row in report["records"]
                if row["target"] == target and row["basis"] == basis and row["k"] == 32
            ]
            if len(values) != len(reports):
                raise ValueError(f"missing k=32 record for {target}/{basis}")
            value = _summary(values)
            null = _summary(
                [
                    float(report["random_subspace_null"][target]["32"]["r2_mean"])
                    for report in reports
                ]
            )
            value["random_null_mean"] = null
            covariance_values = [
                float(row["r2"])
                for report in reports
                for row in report["records"]
                if row["target"] == target and row["basis"] == "covariance" and row["k"] == 32
            ]
            if basis != "covariance":
                value["delta_vs_covariance"] = _summary(
                    [a - b for a, b in zip(values, covariance_values)]
                )
                value["above_covariance_all_splits"] = all(
                    a > b for a, b in zip(values, covariance_values)
                )
            value["above_random_all_splits"] = all(
                a > b for a, b in zip(values, null["values"])
            )
            k32[f"{target}/{basis}"] = value

    overlap_keys = (
        "residual_cross_covariance",
        "residual_covariance",
    )
    overlap_with_covariance = {}
    overlap_with_dynamics = {}
    for key in overlap_keys:
        overlap_with_covariance[key] = _summary(
            [
                float(report["cross_covariance"]["subspace_overlap_with_covariance"]["32"][key])
                for report in reports
            ]
        )
        overlap_with_dynamics[key] = _summary(
            [
                float(report["cross_covariance"]["subspace_overlap_with_dynamics"]["32"][key])
                for report in reports
            ]
        )

    profile = {}
    profile_keys = (
        "covariance_basis",
        "dynamics_basis",
        "residual_cross_covariance_basis",
        "residual_covariance_basis",
    )
    for basis in profile_keys:
        profile[basis] = {
            band: _summary(
                [
                    float(report["heldout_residual_energy_profile"][basis][band])
                    for report in reports
                ]
            )
            for band in ("top", "middle", "tail")
        }

    report = {
        "artifact": "frozen_lewm_residual_subspace_split_robustness",
        "status": "complete_residual_subspace_split_summary",
        "splits": splits,
        "split_count": len(paths),
        "k32": k32,
        "k32_overlap_with_covariance": overlap_with_covariance,
        "k32_overlap_with_dynamics": overlap_with_dynamics,
        "heldout_residual_energy_profile": profile,
        "protocol_consistency": {
            "same_selected_rows": len({report["source"]["selected_rows"] for report in reports}) == 1,
            "same_latent_dim": len({report["source"]["latent_dim"] for report in reports}) == 1,
            "same_sample_seed": len({report["protocol"]["sample_seed"] for report in reports}) == 1,
            "same_test_fraction": len({report["protocol"]["test_fraction"] for report in reports}) == 1,
        },
        "claim_boundary": [
            "Residual-sensitive bases are fitted on training rows and evaluated on held-out frozen latent rows.",
            "The result supports a residual-sensitive subspace distinct from covariance top-band ranking, not an intentional error-relocation mechanism.",
            "No causal control, online planning, or training-improvement claim is made from this frozen artifact.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("notes.txt").write_text(_notes(report))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Split-robust residual-sensitive subspaces in frozen LeWM latents",
        "",
        "Three independent train/test splits are summarized. Values are split estimates; no rows are pooled.",
        "",
        "## Held-out k=32 residual readout",
        "",
        "| basis | mean residual R² | range | random-null mean | delta vs covariance | above covariance all splits | above random all splits |",
        "| --- | ---: | --- | ---: | ---: | :---: | :---: |",
    ]
    for basis in (
        "covariance",
        "dynamics_cross_covariance",
        "residual_cross_covariance",
        "conditional_dynamics_cross_covariance",
    ):
        value = report["k32"][f"residual/{basis}"]
        delta = value.get("delta_vs_covariance", {}).get("mean")
        lines.append(
            f"| {basis} | {value['mean']:.6f} | [{value['min']:.6f}, {value['max']:.6f}] | "
            f"{value['random_null_mean']['mean']:.6f} | {'--' if delta is None else f'{delta:.6f}'} | "
            f"{value.get('above_covariance_all_splits', '--')} | {value['above_random_all_splits']} |"
        )
    lines.extend(
        [
            "",
            "## k=32 subspace overlap",
            "",
            "| residual geometry | overlap with covariance | overlap with dynamics |",
            "| --- | ---: | ---: |",
        ]
    )
    for basis in ("residual_cross_covariance", "residual_covariance"):
        lines.append(
            f"| {basis} | {report['k32_overlap_with_covariance'][basis]['mean']:.6f} | "
            f"{report['k32_overlap_with_dynamics'][basis]['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Held-out residual energy profile",
            "",
            "| basis | top 8 | middle 8--31 | tail 128--191 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for basis, values in report["heldout_residual_energy_profile"].items():
        lines.append(
            f"| {basis} | {values['top']['mean']:.6f} | {values['middle']['mean']:.6f} | {values['tail']['mean']:.6f} |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def _notes(report: dict[str, Any]) -> str:
    residual = report["k32"]["residual/residual_cross_covariance"]
    covariance = report["k32"]["residual/covariance"]
    profile = report["heldout_residual_energy_profile"]["covariance_basis"]
    return (
        "2026-10-03: Summarized three independent residual-subspace probe splits.\n"
        f"Residual cross-covariance k=32 held-out R2 mean={residual['mean']:.6f}, range=[{residual['min']:.6f}, {residual['max']:.6f}], versus covariance mean={covariance['mean']:.6f}.\n"
        f"Held-out residual energy in covariance top-8 mean={profile['top']['mean']:.6f}, middle-8:32 mean={profile['middle']['mean']:.6f}, tail-128:192 mean={profile['tail']['mean']:.6f}.\n"
        "Boundary: frozen residual representation diagnostic, not an intentional error-relocation or control claim.\n"
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
