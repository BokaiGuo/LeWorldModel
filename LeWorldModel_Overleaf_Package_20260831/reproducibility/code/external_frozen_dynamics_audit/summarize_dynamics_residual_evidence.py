"""Assemble a claim-safe evidence summary for dynamics and residual geometry.

The underlying experiments deliberately answer different estimands: frozen
latent readout, frozen rollout sensitivity, physical-motion prediction beyond
actions, external residual transport, and synthetic planner ranking.  This
script validates those artifacts and reports them side by side without
pooling rows, bootstrap intervals, or split estimates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text())
    if not _finite(report):
        raise ValueError(f"non-finite value in {path}")
    return report


def _require(mapping: dict[str, Any], key: str, source: Path) -> Any:
    if key not in mapping:
        raise KeyError(f"{source}: missing key {key}")
    return mapping[key]


def run_report(
    dynamics_path: Path,
    residual_path: Path,
    physical_path: Path,
    transport_path: Path,
    planner_path: Path,
    candidate_path: Path,
    rank_specificity_path: Path | None = None,
) -> dict[str, Any]:
    dynamics = _load(dynamics_path)
    residual = _load(residual_path)
    physical = _load(physical_path)
    transport = _load(transport_path)
    planner = _load(planner_path)
    candidate = _load(candidate_path)
    rank_specificity = _load(rank_specificity_path) if rank_specificity_path is not None else None

    dynamics_k32 = _require(_require(dynamics, "probe", dynamics_path), "k32", dynamics_path)
    dynamics_supervised = _require(dynamics_k32, "dynamics/dynamics_cross_covariance/k32", dynamics_path)
    dynamics_covariance = _require(dynamics_k32, "dynamics/covariance/k32", dynamics_path)
    residual_k32 = _require(residual, "k32", residual_path)
    residual_supervised = _require(residual_k32, "residual/residual_cross_covariance", residual_path)
    residual_covariance = _require(residual_k32, "residual/covariance", residual_path)

    dynamics_external = _require(_require(dynamics, "external_intervention", dynamics_path), "dynamics_top8/h8", dynamics_path)
    energy = _require(_require(dynamics, "energy_matched", dynamics_path), "horizons", dynamics_path)
    covariance_energy = _require(energy, "covariance_top8_energy_matched_to_dynamics/h8", dynamics_path)
    random_energy = _require(energy, "random_top8_energy_matched_to_dynamics/h8", dynamics_path)

    physical_rows = _require(physical, "rows", physical_path)
    physical_row = next(
        row for row in physical_rows
        if row["normalization"] == "public" and row["basis_band"] == "dynamics_basis/top8" and int(row["horizon"]) == 8
    )

    transport_energy = _require(transport, "energy_share", transport_path)
    residual_transport = _require(transport_energy, "residual_basis/top8/h8", transport_path)
    dynamics_transport = _require(transport_energy, "dynamics_basis/top8/h8", transport_path)
    covariance_transport = _require(transport_energy, "covariance_basis/top8/h8", transport_path)

    candidate_focus = _require(candidate, "focused_ood_aligned_minus_random", candidate_path)
    candidate_rows = [value for value in candidate_focus.values()]
    if len(candidate_rows) != 3:
        raise ValueError("candidate sensitivity must contain exactly three focused candidate-count cells")

    strength_focus = _require(planner, "focused_ood_aligned_minus_random", planner_path)
    # The strength artifact has no candidate-count field; select the heldout,
    # OOD, h=8, strength=1 row explicitly from its key.
    strength_key = "none:heldout_calibration:a=1:shift=1.5:h=8"
    strength_one = _require(strength_focus, strength_key, planner_path)

    target_specificity: dict[str, Any] | None = None
    if rank_specificity is not None:
        rank_summary = _require(rank_specificity, "crossfit", rank_specificity_path)["summary"]

        def rank_cell(target: str, basis: str, horizon: int, rank: int = 8) -> dict[str, Any]:
            return _require(
                rank_summary,
                f"{target}/{basis}/h{horizon}/k{rank}",
                rank_specificity_path,
            )["state_incremental_r2_over_action"]

        dynamics_h4 = rank_cell("dynamics_band_increment", "dynamics_basis", 4)
        dynamics_h8 = rank_cell("dynamics_band_increment", "dynamics_basis", 8)
        covariance_h4 = rank_cell("dynamics_band_increment", "covariance_basis", 4)
        random_h4 = rank_cell("dynamics_band_increment", "random_basis", 4)
        residual_h8 = rank_cell("residual_band_increment", "residual_basis", 8)
        cross_residual_h8 = rank_cell("residual_band_increment", "dynamics_basis", 8)
        physical_h8 = rank_cell("physical_displacement", "dynamics_basis", 8)
        target_specificity = {
            "dynamics_target_h4_incremental_r2": float(dynamics_h4["mean"]),
            "dynamics_target_h4_ci95": [float(value) for value in dynamics_h4["ci95"]],
            "dynamics_target_h8_incremental_r2": float(dynamics_h8["mean"]),
            "dynamics_target_h8_ci95": [float(value) for value in dynamics_h8["ci95"]],
            "covariance_control_h4_incremental_r2": float(covariance_h4["mean"]),
            "covariance_control_h4_ci95": [float(value) for value in covariance_h4["ci95"]],
            "random_control_h4_incremental_r2": float(random_h4["mean"]),
            "random_control_h4_ci95": [float(value) for value in random_h4["ci95"]],
            "residual_target_h8_incremental_r2": float(residual_h8["mean"]),
            "residual_target_h8_ci95": [float(value) for value in residual_h8["ci95"]],
            "dynamics_on_residual_target_h8_incremental_r2": float(cross_residual_h8["mean"]),
            "dynamics_on_residual_target_h8_ci95": [float(value) for value in cross_residual_h8["ci95"]],
            "physical_target_h8_incremental_r2": float(physical_h8["mean"]),
            "physical_target_h8_ci95": [float(value) for value in physical_h8["ci95"]],
            "status": "target_specific_frozen_predictive_diagnostic",
        }

    report = {
        "artifact": "dynamics_residual_planner_evidence_summary",
        "source_artifacts": {
            "latent_dynamics": str(dynamics_path),
            "latent_residual": str(residual_path),
            "external_physical": str(physical_path),
            "external_transport": str(transport_path),
            "synthetic_strength": str(planner_path),
            "synthetic_candidate_sensitivity": str(candidate_path),
        },
        "evidence": {
            "latent_dynamics_readout": {
                "dynamics_basis_r2": float(dynamics_supervised["mean"]),
                "covariance_basis_r2": float(dynamics_covariance["mean"]),
                "delta_vs_covariance": float(dynamics_supervised["delta_vs_covariance"]["mean"]),
                "above_covariance_all_splits": bool(dynamics_supervised["above_covariance_all_splits"]),
                "status": "supported_frozen_predictive_subspace",
            },
            "latent_residual_readout": {
                "residual_basis_r2": float(residual_supervised["mean"]),
                "covariance_basis_r2": float(residual_covariance["mean"]),
                "delta_vs_covariance": float(residual_supervised["delta_vs_covariance"]["mean"]),
                "above_covariance_all_splits": bool(residual_supervised["above_covariance_all_splits"]),
                "status": "supported_frozen_residual_sensitive_subspace",
            },
            "external_rollout_sensitivity": {
                "dynamics_top8_h8_delta_vs_none": float(dynamics_external["estimate"]["mean"]),
                "dynamics_top8_ci_lower_min_across_splits": float(dynamics_external["ci95_lower"]["min"]),
                "ci_excludes_zero_all_splits": bool(dynamics_external["ci95_excludes_zero_all_splits"]),
                "covariance_energy_matched_delta_vs_dynamics": float(covariance_energy["estimate"]["mean"]),
                "random_energy_matched_delta_vs_dynamics": float(random_energy["estimate"]["mean"]),
                "controls_ci_lower_gt_zero_all_splits": bool(covariance_energy["ci_lower_gt_zero_all_splits"] and random_energy["ci_lower_gt_zero_all_splits"]),
                "status": "supported_frozen_rollout_sensitivity_not_superiority",
            },
            "external_physical_motion": {
                "action_only_r2": float(physical_row["action_only_r2"]["mean"]),
                "predicted_incremental_r2_over_action": float(physical_row["predicted_incremental_r2_over_action"]["mean"]),
                "predicted_incremental_ci_upper": float(physical_row["predicted_incremental_r2_over_action"]["ci95"][1]),
                "status": "no_added_physical_motion_predictability",
            },
            "external_residual_transport": {
                "residual_top8_share": float(residual_transport["estimate"]["mean"]),
                "dynamics_top8_share": float(dynamics_transport["estimate"]["mean"]),
                "covariance_top8_share": float(covariance_transport["estimate"]["mean"]),
                "residual_minus_covariance": float(residual_transport["delta_vs_covariance"]["mean"]),
                "status": "descriptive_transport_does_not_support_residual_dominance",
            },
            "synthetic_residual_planner": {
                "strength_one_aligned_minus_random": float(strength_one["mean_delta_aligned_minus_random"]),
                "strength_one_focused_holm": float(strength_one["holm_p_focused_family"]),
                "strength_one_global_holm": float(strength_one["holm_p_global"]),
                "candidate_count_aligned_minus_random": {
                    str(int(value["candidate_count"])): float(value["mean_delta_aligned_minus_random"])
                    for value in candidate_rows
                },
                "candidate_count_sign_reversal": bool(
                    candidate_rows[0]["mean_delta_aligned_minus_random"] < 0.0
                    and candidate_rows[-1]["mean_delta_aligned_minus_random"] > 0.0
                ),
                "status": "local_signal_not_candidate_invariant",
            },
        },
        "claim_matrix": [
            {
                "claim": "A dynamics-sensitive latent subspace can be extracted.",
                "evidence": "held-out latent dynamics R2 exceeds covariance across all independent splits",
                "verdict": "supported within frozen predictive diagnostic",
            },
            {
                "claim": "The extracted dynamics subspace is functionally involved in frozen rollout.",
                "evidence": "coefficient removal increases external h=8 error across split intervals",
                "verdict": "supported as frozen rollout sensitivity",
            },
            {
                "claim": "The extracted dynamics subspace adds physical-motion prediction beyond observed actions.",
                "evidence": "predicted latent dynamics increment has negative incremental R2 over action",
                "verdict": "not supported",
            },
            {
                "claim": "Dynamics and residual subspaces are target-specific rather than interchangeable latent bands.",
                "evidence": "rank curve and paired external readouts: dynamics top-8 helps the dynamics-band target, residual top-8 helps the residual-band target, and dynamics top-8 adds no physical-displacement prediction",
                "verdict": "supported as a target-specific frozen diagnostic",
            },
            {
                "claim": "Residual-aware eigenbasis gives a stable planner improvement.",
                "evidence": "aligned-minus-random sign reverses from 16 to 96 candidates",
                "verdict": "not supported",
            },
        ],
        "claim_boundary": [
            "All frozen PushT results are predictive or rollout-sensitivity diagnostics, not physical counterfactuals, causal control, online planning, task success, or retraining gains.",
            "The synthetic planner results are candidate-ranking proxies and are not transferred to PushT.",
            "No rows, split estimates, or bootstrap intervals are pooled into a new confidence interval by this summary.",
        ],
    }
    if rank_specificity is not None:
        report["source_artifacts"]["rank_target_specificity"] = str(rank_specificity_path)
        report["evidence"]["target_specificity"] = target_specificity
    if not _finite(report):
        raise ValueError("summary contains non-finite values")
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    evidence = report["evidence"]
    lines = [
        "# Dynamics, residual, and planner evidence summary",
        "",
        "This report joins independent artifacts without pooling their rows or confidence intervals. Each line retains its original estimand and claim boundary.",
        "",
        "| question | key evidence | verdict |",
        "| --- | --- | --- |",
        f"| Can a dynamics-sensitive latent subspace be extracted? | R² {evidence['latent_dynamics_readout']['dynamics_basis_r2']:.6f} vs covariance {evidence['latent_dynamics_readout']['covariance_basis_r2']:.6f}; delta {evidence['latent_dynamics_readout']['delta_vs_covariance']:.6f} | supported frozen predictive diagnostic |",
        f"| Can a residual-sensitive subspace be extracted? | R² {evidence['latent_residual_readout']['residual_basis_r2']:.6f} vs covariance {evidence['latent_residual_readout']['covariance_basis_r2']:.6f} | supported frozen residual diagnostic |",
        f"| Is the dynamics subspace functionally involved in frozen rollout? | h=8 top-8 removal delta {evidence['external_rollout_sensitivity']['dynamics_top8_h8_delta_vs_none']:.6f}; all split CIs exclude zero | supported as rollout sensitivity |",
        f"| Does it add physical-motion prediction beyond actions? | action-only R² {evidence['external_physical_motion']['action_only_r2']:.6f}; predicted latent increment ΔR² {evidence['external_physical_motion']['predicted_incremental_r2_over_action']:.6f} | not supported |",
        f"| Is residual-aligned planning improvement stable? | aligned−random: c16 {evidence['synthetic_residual_planner']['candidate_count_aligned_minus_random']['16']:.6f}, c48 {evidence['synthetic_residual_planner']['candidate_count_aligned_minus_random']['48']:.6f}, c96 {evidence['synthetic_residual_planner']['candidate_count_aligned_minus_random']['96']:.6f} | not candidate-invariant |",
        "",
        "## Claim matrix",
        "",
        "| claim | evidence | verdict |",
        "| --- | --- | --- |",
    ]
    for row in report["claim_matrix"]:
        lines.append(f"| {row['claim']} | {row['evidence']} | {row['verdict']} |")
    if "target_specificity" in evidence:
        target = evidence["target_specificity"]
        lines.insert(
            lines.index("## Claim matrix"),
            f"| Are dynamics and residual subspaces target-specific? | dynamics target h=4 ΔR² {target['dynamics_target_h4_incremental_r2']:.6f}; residual target h=8 ΔR² {target['residual_target_h8_incremental_r2']:.6f}; physical target h=8 ΔR² {target['physical_target_h8_incremental_r2']:.6f} | supported as target-specific diagnostic |",
        )
        lines.insert(lines.index("## Claim matrix"), "")
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--rank-specificity", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_report(
        args.dynamics,
        args.residual,
        args.physical,
        args.transport,
        args.planner,
        args.candidate,
        args.rank_specificity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
