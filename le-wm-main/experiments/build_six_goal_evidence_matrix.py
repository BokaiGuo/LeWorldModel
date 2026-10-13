"""Build an evidence-bounded acceptance matrix for the six research goals.

This is an audit surface, not a meta-analysis.  It joins independently produced
artifacts without pooling their observations and keeps simulator intervention,
frozen-latent diagnostics, privileged planning, and visual planning separate.
The matrix is intentionally fail-closed: a missing control, environment, or
runtime restriction prevents promotion to the stronger claim tier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": str(path)}
    return json.loads(path.read_text())


def metric(report: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = report.get("summary", {}).get(key)
    return value if isinstance(value, dict) else None


def sweep_summary(paths: list[Path]) -> dict[str, Any]:
    rows: dict[str, list[float]] = {}
    for path in paths:
        report = load(path)
        for row in report.get("results", []):
            horizon = str(row.get("horizon"))
            for key in (
                "latent_endpoint_prediction_error",
                "latent_counterfactual_separation",
                "physical_counterfactual_xy_separation",
                "decoded_position_error_plus",
                "decoded_position_error_minus",
            ):
                value = row.get("metrics", {}).get(key, {}).get("estimate")
                if value is not None:
                    rows.setdefault(f"h{horizon}.{key}", []).append(float(value))
    return {
        key: {"seed_count": len(values), "mean": mean(values), "values": values}
        for key, values in sorted(rows.items())
    }


def build_matrix(
    repo: Path,
    *,
    temporal_summary_path: Path | None = None,
    temporal_action_history_path: Path | None = None,
    temporal_fourier_path: Path | None = None,
) -> dict[str, Any]:
    results = repo / "results"
    benchmark = load(results / "covariance_predictor_diagnostic_benchmark_20261008" / "diagnostic.json")
    learned = load(results / "covariance_predictor_learned_checkpoint_benchmark_20261008" / "learned_checkpoint_diagnostic.json")
    physical = load(results / "pusht" / "lewm" / "physical_intervention_20261004" / "physical_intervention.json")
    closure = load(results / "pusht" / "lewm" / "learned_adapter_closure_contactintervention4096_20261006" / "learned_adapter_closure_aggregate.json")
    pure_phase = load(results / "pusht" / "lewm" / "pure_learned_phase_heads_audit_20261009" / "pure_learned_closure_audit.json")
    pure_dagger = load(results / "pusht" / "lewm" / "pure_learned_dagger_audit_20261009" / "pure_learned_closure_audit.json")
    pure_blockframe = load(results / "pusht" / "lewm" / "pure_learned_blockframe_audit_20261009" / "pure_learned_closure_audit.json")
    visual = load(results / "pusht" / "lewm" / "temporal_visual_adapter_20261004" / "temporal_visual_adapter.json")
    sequence = load(results / "pusht" / "lewm" / "sequence_closure_evidence_20261008" / "sequence_closure_evidence.json")
    visual_online_paths = sorted(results.glob("pusht/lewm/visual_only_online_planning_20261008_seed*/online_planning.json"))
    visual_online_reports = [load(path) for path in visual_online_paths]
    visual_online_summary: dict[str, Any] = {}
    for policy in ("zero", "random", "vanilla"):
        rows = [
            row.get("summary", {})
            for report in visual_online_reports
            for row in report.get("reports", [])
            if row.get("policy") == policy
        ]
        if rows:
            visual_online_summary[policy] = {
                "seed_count": len(rows),
                "episodes": sum(int(row.get("n", 0)) for row in rows),
                "successes": sum(float(row.get("success_rate", 0.0)) * int(row.get("n", 0)) for row in rows),
                "mean_block_xy_distance": mean(float(row["mean_block_xy_distance"]) for row in rows),
                "mean_final_distance": mean(float(row["mean_final_distance"]) for row in rows),
            }
    geometry = load(results / "theory_operator_alignment_iclr_oral_20260829" / "jepa_wm_temporal_gate" / "cross_model_temporal_geometry_audit.json")
    uncertainty = load(results / "pusht" / "lewm" / "uncertainty_subspace_conflict_20261003.json")
    gauge = load(results / "transfer_certificate_20261008" / "gauge_nonnormal_audit.json")
    temporal = load(
        temporal_summary_path
        if temporal_summary_path is not None
        else results / "protocol_corrected_temporal_export_20261012"
        / "jepa_pusht_windows4_composite_multiseed_summary.json"
    )
    temporal_action_history = load(
        temporal_action_history_path
        if temporal_action_history_path is not None
        else results / "protocol_corrected_temporal_export_20261012"
        / "jepa_pusht_windows4_action_history_multiseed_summary.json"
    )
    temporal_fourier = load(
        temporal_fourier_path
        if temporal_fourier_path is not None
        else results / "protocol_corrected_temporal_export_20261012"
        / "jepa_pusht_windows4_fourier_state_multiseed_summary.json"
    )

    sweeps = sorted(
        results.glob("pusht/lewm/counterfactual_horizon_sweep_*_20261008/counterfactual_horizon_sweep.json")
    )
    cf = sweep_summary(sweeps)

    goals = [
        {
            "id": 1,
            "name": "alignment predicts held-out transfer/planning",
            "status": "partial",
            "evidence_bucket": "partial",
            "establishes": "Alignment beats spectrum-only tie breaking in a controlled linear family; learned-checkpoint evidence is heterogeneous and does not yet predict planning success.",
            "key_metrics": {
                "controlled_selection_win_rate": benchmark.get("summary", {}).get("alignment_selection_win_rate"),
                "controlled_rank_spearman": benchmark.get("summary", {}).get("alignment_transfer_rank_spearman"),
                "learned_checkpoint_error_constrained_regret": learned.get("summary", {}).get("error_constrained_alignment_selection_regret"),
            },
            "missing_acceptance": "Independent learned checkpoints/environments with matched planning outcomes and an intervention that changes alignment while holding spectrum and prediction error fixed.",
            "artifacts": [
                "results/covariance_predictor_diagnostic_benchmark_20261008/diagnostic.json",
                "results/covariance_predictor_learned_checkpoint_benchmark_20261008/learned_checkpoint_diagnostic.json",
            ],
        },
        {
            "id": 2,
            "name": "learned adapter stably closes orientation",
            "status": "partial_negative",
            "evidence_bucket": "negative",
            "establishes": "A privileged learned proposal plus simulator full-fan verification can close pose in some matched cases, but the protocol explicitly reports pure_learned_rollout=false; therefore stable learned closure is not established.",
            "key_metrics": {
                "privileged_pose_closure": closure.get("aggregate", {}).get("metrics", {}).get("block_pose_closure"),
                "privileged_success": closure.get("aggregate", {}).get("metrics", {}).get("success"),
                "sequence_strict_success": sequence.get("aggregate_closure", {}).get("sequence_continuous_rank", {}).get("strict_success"),
                "fresh_pure_learned_phase_heads": {
                    "strict_success": pure_phase.get("success_count"),
                    "denominator": pure_phase.get("success_denominator"),
                    "mean_block_xy": pure_phase.get("metrics", {}).get("block_xy_distance"),
                    "mean_angle": pure_phase.get("metrics", {}).get("angle_error"),
                },
                "fresh_pure_learned_dagger": {
                    "strict_success": pure_dagger.get("success_count"),
                    "denominator": pure_dagger.get("success_denominator"),
                    "mean_block_xy": pure_dagger.get("metrics", {}).get("block_xy_distance"),
                    "mean_angle": pure_dagger.get("metrics", {}).get("angle_error"),
                },
                "fresh_pure_learned_blockframe": {
                    "strict_success": pure_blockframe.get("success_count"),
                    "denominator": pure_blockframe.get("success_denominator"),
                    "mean_block_xy": pure_blockframe.get("metrics", {}).get("block_xy_distance"),
                    "mean_angle": pure_blockframe.get("metrics", {}).get("angle_error"),
                },
            },
            "missing_acceptance": "The fresh phase-head, block-frame, and DAgger probes remain 1/16 or 2/16 strict success across two seed clusters; stable closure still requires a prespecified success threshold met across fresh seeds and unseen initial states without simulator candidate verification.",
            "artifacts": [
                "results/pusht/lewm/learned_adapter_closure_contactintervention4096_20261006/learned_adapter_closure_aggregate.json",
                "results/pusht/lewm/sequence_closure_evidence_20261008/sequence_closure_evidence.json",
                "results/pusht/lewm/pure_learned_phase_heads_audit_20261009/pure_learned_closure_audit.json",
                "results/pusht/lewm/pure_learned_dagger_audit_20261009/pure_learned_closure_audit.json",
                "results/pusht/lewm/pure_learned_blockframe_audit_20261009/pure_learned_closure_audit.json",
            ],
        },
        {
            "id": 3,
            "name": "frozen latent has bounded physical action-effect dynamics",
            "status": "partial",
            "evidence_bucket": "partial",
            "establishes": "The simulator has a paired action effect, and the frozen latent responds to intervention sequences; held-out physical decoding remains poor, so only a bounded latent action-associated diagnostic is supported.",
            "key_metrics": {
                "simulator_block_pose_effect": physical.get("metrics", {}).get("block_pose_effect"),
                "new_frozen_counterfactual_sweep": cf,
                "protocol_corrected_temporal_transfer": {
                    "n_splits": temporal.get("n_splits"),
                    "wins": temporal.get("wins"),
                    "aggregates": temporal.get("aggregates"),
                },
                "action_history_memory_repair": {
                    "n_splits": temporal_action_history.get("n_splits"),
                    "paired_memory4_wins": temporal_action_history.get("wins", {}).get("paired_memory4"),
                    "aggregates": temporal_action_history.get("aggregates"),
                },
                "fourier_state_memory": {
                    "n_splits": temporal_fourier.get("n_splits"),
                    "paired_memory4_wins": temporal_fourier.get("wins", {}).get("paired_memory4"),
                    "aggregates": temporal_fourier.get("aggregates"),
                    "claim_boundary": "state-history reparameterization only; not evidence of recovered contact-mode observability",
                },
            },
            "missing_acceptance": "Cross-fitted latent-to-physical effect prediction with uncertainty calibration, held-out action ranking, and multi-step error that beats no-action and observed-state controls across seeds; the corrected temporal audit remains transfer evidence rather than physical causality.",
            "artifacts": [
                "results/pusht/lewm/physical_intervention_20261004/physical_intervention.json",
                *[str(path.relative_to(results)) for path in sweeps],
                "results/pusht/lewm/temporal_visual_adapter_20261004/temporal_visual_adapter.json",
            ],
        },
        {
            "id": 4,
            "name": "frozen LeWM supports visual online planning",
            "status": "blocked",
            "evidence_bucket": "negative",
            "establishes": "The existing temporal visual readout tests observability, while the current successful planning artifacts are privileged or use successor verification; a pure visual closed-loop result is absent.",
            "key_metrics": {
                "visual_observability_rows": visual.get("sanity_checks", {}).get("n_test_rows"),
                "visual_only_online": visual_online_summary,
                "sequence_strict_success": sequence.get("aggregate_closure", {}).get("sequence_continuous_rank", {}).get("strict_success"),
            },
            "missing_acceptance": "Runtime-only image/action interface on PushT and a second environment, with simulator state, true contact, successor verification, and post-hoc correction disabled.",
            "artifacts": [
                "results/pusht/lewm/temporal_visual_adapter_20261004/temporal_visual_adapter.json",
                *[str(path.relative_to(results)) for path in visual_online_paths],
                "results/pusht/lewm/sequence_closure_evidence_20261008/sequence_closure_evidence.json",
            ],
        },
        {
            "id": 5,
            "name": "cross-environment transferable latent world model",
            "status": "partial_proxy",
            "evidence_bucket": "proxy",
            "establishes": "Cross-model temporal geometry contains multiple diagnostic quantities and prior external latent-trajectory transport exists, but zero-shot task transfer and visual task success are not established.",
            "key_metrics": {
                "geometry_record_count": geometry.get("records", 0),
                "uncertainty_summary_present": bool(uncertainty.get("summary")),
            },
            "missing_acceptance": "At least two environments with different action interfaces and held-out geometry/tasks, zero-shot and few-shot transfer, OOD rollout, task success, and calibration.",
            "artifacts": [
                "results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/cross_model_temporal_geometry_audit.json",
                "results/pusht/lewm/uncertainty_subspace_conflict_20261003.json",
            ],
        },
        {
            "id": 6,
            "name": "joint transfer diagnostic/certificate",
            "status": "partial_diagnostic",
            "evidence_bucket": "partial",
            "establishes": "The joint descriptor is now operationally assembled from spectrum, projector/subspace overlap, commutator, non-normality, held-out residual, and uncertainty artifacts; metric-aware gauge and Schur controls pass.",
            "key_metrics": {
                "gauge_audit": gauge.get("gauge", {}),
                "geometry_records": geometry.get("records", 0),
                "uncertainty_summary_present": bool(uncertainty.get("summary")),
                "unified_temporal_audit": {
                    "n_splits": temporal.get("n_splits"),
                    "wins": temporal.get("wins"),
                    "claim_boundary": temporal.get("claim_boundary"),
                "fourier_state_memory": {
                    "n_splits": temporal_fourier.get("n_splits"),
                    "wins": temporal_fourier.get("wins"),
                    "claim_boundary": temporal_fourier.get("claim_boundary"),
                },
                },
            },
            "missing_acceptance": "A prespecified bound or calibrated prediction interval for held-out transfer error, tested on independent learned checkpoints/environments; current evidence is a diagnostic vector, not a universal certificate.",
            "artifacts": [
                "results/transfer_certificate_20261008/gauge_nonnormal_audit.json",
                "results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/cross_model_temporal_geometry_audit.json",
                "results/pusht/lewm/uncertainty_subspace_conflict_20261003.json",
                "results/protocol_corrected_temporal_export_20261012/jepa_pusht_windows4_composite_multiseed_summary.json",
                "results/protocol_corrected_temporal_export_20261012/jepa_pusht_windows4_action_history_multiseed_summary.json",
            ],
        },
    ]

    return {
        "artifact": "six_goal_evidence_matrix",
        "date": "2026-10-12",
        "claim_boundary": [
            "This matrix joins artifacts without pooling their observations or treating row count as independent replication.",
            "Privileged simulator interventions and simulator-verified planning are kept separate from frozen-latent causality and visual online planning.",
            "No goal is promoted beyond the strongest protocol actually run.",
        ],
        "goals": goals,
        "protocol": {
            "fresh_counterfactual_sweep_artifacts": [str(path.relative_to(results)) for path in sweeps],
            "frozen_latent_checkpoint": "pusht/lewm",
            "matrix_is_meta_analysis": False,
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Six-goal evidence matrix",
        "",
        "This is an evidence-bounded audit, not a pooled meta-analysis.  A stronger status requires the stronger protocol; compile-clean or schema-complete artifacts do not promote a claim.",
        "",
        "| goal | status | evidence bucket | what is established | remaining acceptance test |",
        "|---|---|---|---|---|",
    ]
    for goal in report["goals"]:
        lines.append(
            f"| {goal['id']}. {goal['name']} | {goal['status']} | {goal['evidence_bucket']} | {goal['establishes']} | {goal['missing_acceptance']} |"
        )
    lines.extend(["", "## Reader-facing conclusion", ""])
    lines.append(
        "The present evidence supports a covariance--predictor diagnostic program with bounded latent/action-associated structure, but it does not yet support a stable learned pose-closure claim, pure visual online planning, or a cross-environment world-model claim. The decisive remaining gap is not another PushT endpoint ablation: it is an independently held-out visual/action protocol with pure learned successor prediction and matched cross-environment transfer."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporal-summary", type=Path, default=None)
    parser.add_argument("--temporal-action-history-summary", type=Path, default=None)
    parser.add_argument("--temporal-fourier-summary", type=Path, default=None)
    args = parser.parse_args()
    report = build_matrix(
        ROOT,
        temporal_summary_path=args.temporal_summary,
        temporal_action_history_path=args.temporal_action_history_summary,
        temporal_fourier_path=args.temporal_fourier_summary,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "six_goal_evidence_matrix.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "six_goal_evidence_matrix.md").write_text(markdown(report))
    print(json.dumps({"output": str(args.output), "statuses": {str(g["id"]): g["status"] for g in report["goals"]}}, indent=2))


if __name__ == "__main__":
    main()
