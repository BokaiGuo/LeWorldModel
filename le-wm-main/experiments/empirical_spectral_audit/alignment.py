"""Coverage manifest mapping the empirical proposal to code artifacts."""

EMPIRICAL_PROPOSAL_ALIGNMENT = {
    "conditional_latent_spectra": {
        "proposal": "state/action/goal/episode/phase/latent conditional eigenspace audit",
        "apis": ["EmpiricalAuditConfig", "build_condition_labels", "run_empirical_audit"],
        "report_fields": ["condition", "cells", "weighted_mean"],
    },
    "tangent_families": {
        "proposal": "dynamics, action, predicted, residual, physical, nuisance, counterfactual tangents",
        "apis": ["build_tangent_families", "latent_eigenspace_audit.py --rich-fields"],
        "report_fields": ["schema.available_tangents", "goals", "success", "episode_id", "timestep", "physical_*", "counterfactual_*"],
    },
    "core_metrics": {
        "proposal": "DTL, rank-wise fingerprint, PSE, PPTE, NCR",
        "apis": [
            "rank_wise_transfer_fingerprint",
            "prediction_spectral_error",
            "planning_projected_tangent_energy",
            "nuisance_contamination_ratio",
        ],
        "report_fields": ["lifts", "fingerprints", "prediction_spectral_error", "ppte", "ncr"],
    },
    "controls": {
        "proposal": "random basis, shuffle, norm-matched, Gaussian, random rotation, counterfactual controls",
        "apis": ["norm_matched_random_tangents", "isotropic_gaussian_tangents_like", "matched_eigenvalue_random_rotation"],
        "report_fields": ["label_controls", "controls"],
    },
    "multi_horizon": {
        "proposal": "horizon spectral collapse and multi-horizon residual spectra",
        "apis": ["multi_horizon_report", "uncertainty_residual_report"],
        "report_fields": ["multi_horizon", "uncertainty_residual"],
    },
    "controllability_observability": {
        "proposal": "controllability spectrum and observability/probe spectrum",
        "apis": ["controllability_spectrum_report", "observability_probe_report"],
        "report_fields": ["controllability", "observability"],
    },
    "eigenspace_transport": {
        "proposal": "success/failure or phase trajectory eigenspace transport",
        "apis": ["eigenspace_transport_report"],
        "report_fields": ["eigenspace_transport"],
    },
    "planner_interventions": {
        "proposal": "band dropout, band noise, factor suppression/amplification/rotation",
        "apis": ["run_intervention_audit"],
        "report_fields": ["planner_interventions"],
    },
    "spectral_planner_metric": {
        "proposal": "d_spec(z,g) spectral planner metric weighted by reachability/controllability/nuisance",
        "apis": ["spectral_planner_weights", "spectral_planner_distance", "spectral_planner_metric_report", "SpectralCostModel", "eval.py eval.spectral_planner.enabled=true"],
        "report_fields": ["spectral_planner_metric", "eval_metrics.json"],
    },
    "operator_alignment_bridge": {
        "proposal": "post-hoc empirical bridge from frozen PushT audit to operator-alignment evidence",
        "apis": ["empirical_operator_alignment_bridge_report", "run_operator_alignment_bridge.py"],
        "report_fields": ["primary_lifts", "control_lifts", "label_shuffle_lifts", "transport", "planner_weights"],
    },
    "operator_alignment_paper_outline": {
        "proposal": "standalone empirical paper spine for the frozen PushT audit",
        "apis": ["empirical_operator_alignment_paper_outline"],
        "report_fields": ["claim_ladder", "main_figure", "rivals", "falsifiers", "evidence_bridge"],
    },
    "operator_alignment_paper_spine": {
        "proposal": "manuscript-form empirical paper spine for the frozen PushT audit",
        "apis": ["empirical_operator_alignment_paper_spine"],
        "report_fields": ["abstract", "contribution", "main_figure", "rivals", "falsifiers", "evidence_bridge"],
    },
    "spectral_regularization": {
        "proposal": "action-align, reach-align, nuisance suppression, spectral prediction, residual/contact losses",
        "apis": ["directional_regularizer_losses", "action_controllability_alignment_loss", "teacher_anchor_loss", "train_lewm_directional.py"],
        "report_fields": ["training_helper"],
    },
    "conditional_operator_alignment": {
        "proposal": "opt-in global or action-conditional covariance--predictor alignment intervention with an isotropy boundary",
        "apis": ["operator_alignment_loss", "operator_alignment_diagnostic", "run_alignment_regularizer_sweep.py", "train_lewm_directional.py"],
        "report_fields": ["operator_alignment_loss", "alignment_regularizer_sweep.json", "alignment_regularizer_sweep.md", "protocol.learned_training"],
    },
    "action_composition": {
        "proposal": "action-conditioned operator composition as a bridge from local geometry to multi-step prediction",
        "apis": ["_action_composition_errors", "run_alignment_training_synthetic.py"],
        "report_fields": ["action_composition_error", "summary.*.action_composition_error_mean", "rollout_mse"],
        "boundary": "synthetic action-conditioned linear benchmark only; not evidence of nonlinear LeWM composition or planning",
    },
    "alignment_functional_consequences": {
        "proposal": "matched alignment intervention evaluated with arbitrary action-sequence rollout and candidate-set planning proxy",
        "apis": ["run_alignment_functional_synthetic.py", "_random_rollout_mse", "_planning_proxy"],
        "report_fields": ["random_rollout_mse", "action_composition_error", "planning", "diagnostics.conditional_commutator"],
        "boundary": "CPU-only synthetic candidate-set proxy; not evidence of nonlinear LeWM, PushT, online planning, or causal control",
    },
    "joint_functional_factorial": {
        "proposal": "matched factorial functional comparison of predictor equivariance and conditional alignment",
        "apis": ["run_joint_functional_synthetic.py", "summarize_joint_functional_inference.py"],
        "report_fields": ["id_rollout_mse", "ood_rollout_mse", "planning", "diagnostics", "interaction"],
        "boundary": "CPU-only synthetic factorial proxy; not evidence of nonlinear LeWM, PushT, online planning, or causal control",
    },
    "nonlinear_local_alignment_functional": {
        "proposal": "state-local covariance--Jacobian alignment in a nonlinear action-conditioned world",
        "apis": ["run_nonlinear_local_alignment_synthetic.py", "summarize_matched_functional_inference.py"],
        "report_fields": ["test_one_step_mse", "id_rollout_mse", "ood_rollout_mse", "composition_error", "planning", "diagnostics.conditional_local_commutator"],
        "boundary": "CPU-only nonlinear synthetic functional proxy; not evidence of nonlinear LeWM, PushT, online planning, or causal control",
    },
    "nonlinear_planning_horizon_sensitivity": {
        "proposal": "action-composition to long-horizon planning sensitivity under shared candidate sets",
        "apis": ["run_nonlinear_planning_horizon_synthetic.py"],
        "report_fields": ["summary", "paired_inference", "planning.trajectory_regret", "planning.trajectory_success_rate"],
        "boundary": "CPU-only candidate-set planning proxy; not online task success, causal control, nonlinear LeWM, or PushT evidence",
    },
    "checkpoint_matched_geometry_intervention": {
        "proposal": "checkpoint-level validation-MSE matching before downstream nonlinear geometry evaluation",
        "apis": ["run_checkpoint_matched_geometry_synthetic.py", "_select_baseline", "_select_nearest"],
        "report_fields": ["protocol.checkpoint_matching", "records.baseline_epoch", "records.intervention_epoch", "records.relative_validation_gap", "inference"],
        "boundary": "CPU-only checkpoint-matched nonlinear synthetic proxy; controls observed validation one-step error but is not causal evidence or LeWM/PushT performance",
    },
    "nonlinear_alignment_strength_phase": {
        "proposal": "strength-dependent trade-off between local covariance-Jacobian alignment and nonlinear functional readouts",
        "apis": ["run_nonlinear_alignment_phase_synthetic.py", "run_phase", "paired_inference"],
        "report_fields": ["absolute_summary", "paired_inference", "metrics.planning_trajectory_regret", "metrics.conditional_local_commutator"],
        "boundary": "CPU-only nonlinear synthetic phase proxy; does not establish monotone alignment benefit, causal control, nonlinear LeWM, or PushT performance",
    },
    "nonlinear_alignment_phase_matched": {
        "proposal": "checkpoint-level matched-prediction strength sweep for nonlinear local alignment",
        "apis": ["run_nonlinear_alignment_phase_matched_synthetic.py", "_select_baseline", "_select_nearest"],
        "report_fields": ["protocol.checkpoint_matching", "records.relative_validation_gap", "paired_inference", "matching_summary"],
        "boundary": "CPU-only nearest-checkpoint nonlinear proxy; strict matched interpretation is limited by the reported validation gaps and does not establish causal control or LeWM/PushT performance",
    },
    "pixel_world_training_intervention": {
        "proposal": "joint pixel-encoder/predictor alignment intervention with random-sequence rollout and action composition",
        "apis": ["run_pixel_world_model_synthetic.py", "PixelWorldModel", "operator_alignment_loss"],
        "report_fields": ["test_one_step_mse", "rollout_mse", "action_composition_error", "diagnostics.stable_rank", "best_epoch"],
        "boundary": "synthetic pixel-world benchmark only; not evidence of nonlinear LeWM, PushT, controllability, or planning",
    },
    "pixel_nuisance_ood": {
        "proposal": "learned pixel-world nuisance-amplitude OOD with action/nuisance tangent diagnostics",
        "apis": ["run_pixel_nuisance_ood_synthetic.py", "summarize_pixel_nuisance_inference.py"],
        "report_fields": ["ood_one_step_mse", "ood_rollout_mse", "ood_tangent_diagnostics", "diagnostics"],
        "boundary": "synthetic learned-pixel nuisance bridge only; not evidence of invariant LeWM or real-environment robustness",
    },
    "spectral_exploration": {
        "proposal": "spectrally guided exploration score",
        "apis": ["spectral_exploration_report"],
        "report_fields": ["spectral_exploration"],
    },
    "ood_shift": {
        "proposal": "OOD spectral shift for generalization/failure prediction",
        "apis": ["spectral_shift_report"],
        "report_fields": ["ood_spectral_shift"],
    },
    "warnings_schema": {
        "proposal": "explicit warnings for optional arrays and backwards-compatible schema",
        "apis": ["audit_warnings", "load_latent_arrays"],
        "report_fields": ["schema", "warnings"],
    },
}


def empirical_alignment_keys() -> set[str]:
    return set(EMPIRICAL_PROPOSAL_ALIGNMENT)
