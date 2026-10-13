"""Create a reader-facing cross-environment audit of enriched bridge results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads((path / "enriched_action_task_bridge.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pusht", type=Path, required=True)
    parser.add_argument("--pointmaze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {"PushT": _load(args.pusht), "PointMaze": _load(args.pointmaze)}
    rows = []
    for environment, report in reports.items():
        summary = report["summary"]
        fixed = summary["fixed_spectrum_bridge"]
        static_rank = [
            split["matched_rank_correlation"].get("observed_zero_action_effect_pullback_alignment")
            for split in report["splits"]
        ]
        joint_rank = [
            split["action_conditioned"]["matched_rank_correlation"].get(
                "joint_observed_zero_action_effect_pullback_transient_amplification"
            )
            for split in report["splits"]
        ]
        static_rank = [value for value in static_rank if value is not None]
        joint_rank = [value for value in joint_rank if value is not None]
        rows.append({
            "environment": environment,
            "n_observed_control_matches": summary.get("zero", {}).get("matched_episodes"),
            "observed_mse": summary.get("zero", {}).get("observed_prediction_mse"),
            "zero_control_mse": summary.get("zero", {}).get("latent_prediction_mse"),
            "observed_zero_gap": summary.get("zero", {}).get("latent_action_gap_rmse"),
            "paired_effect_samples": summary.get("observed_zero_action_effect", {}).get("matched_samples"),
            "source_latent_shape_mismatches": summary.get("observed_zero_action_effect", {}).get("source_latent_shape_mismatches"),
            "shuffle_control_mse": summary.get("shuffle", {}).get("latent_prediction_mse"),
            "matched_candidate_min": min(s["validation_feasible_count"] for s in report["splits"]),
            "matched_candidate_mean": sum(s["validation_feasible_count"] for s in report["splits"]) / len(report["splits"]),
            "static_validation_regret": fixed["mean_regret"]["validation_mse"],
            "static_matched_action_metric_regret": fixed["mean_regret"]["matched_action_interaction_pullback_alignment"],
            "static_matched_observed_zero_effect_regret": fixed["mean_regret"].get("matched_observed_zero_action_effect_pullback_alignment"),
            "static_physical_effect_regret": fixed.get("mean_physical_regret", {}).get("matched_observed_zero_action_effect_pullback_alignment"),
            "residual_action_conditioned_validation_regret": fixed["action_conditioned_mean_regret"]["action_conditioned_validation_mse"],
            "residual_action_conditioned_matched_metric_regret": fixed["action_conditioned_mean_regret"]["action_conditioned_matched_action_interaction_pullback_alignment"],
            "augmented_joint_task_regret": fixed["action_conditioned_mean_regret"].get("action_conditioned_matched_joint_task_pullback_alignment"),
            "augmented_joint_effect_transient_regret": fixed["action_conditioned_mean_regret"].get("action_conditioned_matched_joint_observed_zero_action_effect_pullback_transient_amplification"),
            "augmented_joint_physical_effect_transient_regret": fixed.get("action_conditioned_mean_physical_regret", {}).get("action_conditioned_matched_joint_observed_zero_action_effect_pullback_transient_amplification"),
            "effect_alignment_mean_rank_correlation": sum(static_rank) / len(static_rank) if static_rank else None,
            "joint_effect_transient_mean_rank_correlation": sum(joint_rank) / len(joint_rank) if joint_rank else None,
            "physical_effect_alignment_mean_rank_correlation": sum(
                value for value in (
                    split["matched_physical_rank_correlation"].get("observed_zero_action_effect_pullback_alignment")
                    for split in report["splits"]
                ) if value is not None
            ) / max(1, sum(
                value is not None for value in (
                    split["matched_physical_rank_correlation"].get("observed_zero_action_effect_pullback_alignment")
                    for split in report["splits"]
                )
            )),
            "joint_physical_effect_transient_mean_rank_correlation": sum(
                value for value in (
                    split["action_conditioned"]["matched_physical_rank_correlation"].get("joint_observed_zero_action_effect_pullback_transient_amplification")
                    for split in report["splits"]
                ) if value is not None
            ) / max(1, sum(
                value is not None for value in (
                    split["action_conditioned"]["matched_physical_rank_correlation"].get("joint_observed_zero_action_effect_pullback_transient_amplification")
                    for split in report["splits"]
                )
            )),
            "latent_only_r2": fixed["mean_physical_readout_r2"]["latent_only"],
            "joint_r2": fixed["mean_physical_readout_r2"]["joint"],
        })
    output = {
        "protocol": {
            "environments": ["PushT", "PointMaze"],
            "architectures": ["JEPA-WM"],
            "architecture_count": 1,
            "control": "matched observed versus zero action; within-episode shuffle when available",
            "metric": "train-only conditional external pullback with fixed 0.05 floor",
            "selector": "fixed-spectrum probes; episode-disjoint train/validation/test split",
            "negative_result_policy": "all prior pooled-latent results remain separate and unchanged",
        },
        "rows": rows,
        "root_cause_findings": [
            "The old pooled export omitted action blocks, raw state transitions, and source proprio, so cross-environment external metrics could not be identified.",
            "The released JEPA-WM interface is multimodal; visual-only physical readout underestimates observability when proprio is omitted.",
            "A static A_x probe marginalizes the action-conditioned B_u interface; residual action fitting is numerically more stable than joint A_x/B_u fitting under weak action signal.",
            "The initial random-conjugate five-percent matching gate was anchor-only; a deterministic local spectrum-preserving family now supplies genuine matched candidates and records the feasible-pool size.",
            "The augmented Schur/transient diagnostics evaluate [[A_x^T, B_u^T], [0, 0]] under the external metric, reducing interface mismatch but not establishing a universal selector.",
            "The rerun reports environment-specific effect geometry rather than pooling it: "
            + "; ".join(
                f"{row['environment']} effect-alignment rho={row['effect_alignment_mean_rank_correlation']:.6g}, "
                f"joint physical-transient rho={row['joint_physical_effect_transient_mean_rank_correlation']:.6g}"
                for row in rows
            ) + ".",
            "PointMaze remains weakly action-sensitive under this checkpoint/data protocol, so metric repair alone cannot manufacture transferable action information.",
        ],
        "claim_boundary": "The enriched interface and residual action repair address measurement and conditioning failures, but this is a two-environment, one-architecture JEPA-WM pilot. It does not establish a universal model-selection criterion or world-model transfer theorem.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "enriched_cross_environment_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# Enriched cross-environment action/task bridge audit", "",
        "| Environment | observed MSE | zero MSE | observed-zero gap | matched candidates (min/mean) | static matched metric | effect metric | physical effect metric | residual action-conditioned | augmented joint task | augmented joint effect transient | physical joint transient | effect rank rho | physical rank rho | latent R2 | joint R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['environment']} | {row['observed_mse']:.6g} | {row['zero_control_mse']:.6g} | "
            f"{row['observed_zero_gap']:.6g} | {row['matched_candidate_min']:.0f}/{row['matched_candidate_mean']:.1f} | "
            f"{row['static_matched_action_metric_regret']:.6g} | {row['static_matched_observed_zero_effect_regret']:.6g} | "
            f"{row['static_physical_effect_regret']:.6g} | {row['residual_action_conditioned_matched_metric_regret']:.6g} | "
            f"{row['augmented_joint_task_regret']:.6g} | {row['augmented_joint_effect_transient_regret']:.6g} | "
            f"{row['augmented_joint_physical_effect_transient_regret']:.6g} | {row['effect_alignment_mean_rank_correlation']:.6g} | "
            f"{row['physical_effect_alignment_mean_rank_correlation']:.6g} | {row['latent_only_r2']:.6g} | {row['joint_r2']:.6g} |"
        )
    lines += ["", "## Root-cause findings", ""] + [f"- {item}" for item in output["root_cause_findings"]]
    lines += ["", "## Claim boundary", "", output["claim_boundary"], ""]
    (args.output / "enriched_cross_environment_audit.md").write_text("\n".join(lines))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
