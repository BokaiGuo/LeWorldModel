"""Coverage manifest mapping the theory proposal to code artifacts."""

THEORY_PROPOSAL_ALIGNMENT = {
    "oracle_transfer_operator": {
        "proposal": "C_theta^{1/2} G_E C_theta^{1/2}, power law, broken power law",
        "apis": ["TransferConfig", "diagonal_transfer_spectra", "power_law_spectrum", "broken_power_law_response"],
        "report_fields": ["spectra.latent_eigenvalues", "expected_oracle_alpha", "slopes"],
    },
    "encoder_predictor_joint_transfer": {
        "proposal": "encoder transfer, predictor transfer, prediction residual spectrum",
        "apis": ["diagonal_transfer_spectra", "spectral_transfer_report"],
        "report_fields": ["spectra.predictor_response", "spectra.predicted_eigenvalues", "spectra.residual_eigenvalues"],
    },
    "multi_horizon_residual_dynamics": {
        "proposal": "multi-horizon predictor composition and residual growth",
        "apis": ["multi_horizon_transfer_report", "spectral_transfer_report"],
        "report_fields": ["multi_horizon.horizons", "multi_horizon.reports", "multi_horizon.growth"],
    },
    "invariance_heat_filter": {
        "proposal": "invariance objective as heat filter and known-generator recovery",
        "apis": ["heat_filter", "heat_filter_recovery_report"],
        "report_fields": ["heat_filter_recovery"],
    },
    "oblique_stress": {
        "proposal": "oblique-frame finite-window stress tests",
        "apis": ["oblique_frame_stress_report"],
        "report_fields": ["oblique_frame_stress"],
    },
    "operator_alignment_noncommutativity": {
        "proposal": "same spectrum, different basis, and commutator-sensitive operator alignment",
        "apis": ["operator_alignment_report"],
        "report_fields": ["records", "random_null", "contrast"],
    },
    "operator_alignment_law_card": {
        "proposal": "formal law card for the same-spectrum operator-alignment theorem candidate",
        "apis": ["operator_alignment_law_card"],
        "report_fields": ["phenomenon", "candidate_law", "falsifiers"],
    },
    "operator_alignment_oral_object_card": {
        "proposal": "oral-level object card for same-spectrum operator alignment",
        "apis": ["operator_alignment_oral_object_card"],
        "report_fields": ["mother_object", "field_tension", "candidate_law", "comparison_battlefield", "falsifier"],
    },
    "operator_alignment_theorem": {
        "proposal": "exact single-plane commutator law and flat-spectrum falsifiers",
        "apis": ["operator_alignment_theorem_report"],
        "report_fields": ["theorem", "falsifiers", "claim_boundary"],
    },
    "operator_alignment_paper_spine": {
        "proposal": "manuscript-form paper spine for the same-spectrum operator-alignment theorem",
        "apis": ["operator_alignment_paper_spine"],
        "report_fields": ["abstract", "contribution", "main_figure", "rivals", "falsifiers", "evidence_bridge"],
    },
    "operator_alignment_paper_draft": {
        "proposal": "full manuscript draft for the same-spectrum operator-alignment theory paper",
        "apis": ["operator_alignment_paper_draft"],
        "report_fields": ["abstract", "introduction", "method", "results", "discussion", "claim_ledger"],
    },
    "operator_alignment_display_plan": {
        "proposal": "claim-owned display plan for main and supplementary figures in the theory paper",
        "apis": ["operator_alignment_display_plan.md"],
        "report_fields": ["main_figure", "supplement_s1", "supplement_s2", "supplement_s3", "supplement_s4"],
    },
    "operator_alignment_research_plan": {
        "proposal": "ICLR-oral research plan for strengthening the same-spectrum operator-alignment paper",
        "apis": ["operator_alignment_research_plan"],
        "report_fields": ["current_verdict", "must_run_experiments", "defer_to_supplement", "defer_to_future_work"],
    },
    "operator_alignment_null_sweep": {
        "proposal": "stronger falsifier sweep with permutation-preserving and spectrum-matched random-basis nulls",
        "apis": ["operator_alignment_null_sweep_report", "run_operator_alignment_null_sweep.py"],
        "report_fields": ["aligned", "oblique", "permutation_preserving_null", "spectrum_matched_random_basis_null"],
    },
    "operator_alignment_anisotropy_sweep": {
        "proposal": "continuous anisotropy boundary sweep from flat to strongly anisotropic spectra",
        "apis": ["operator_alignment_anisotropy_sweep_report", "run_operator_alignment_anisotropy_sweep.py"],
        "report_fields": ["rows", "claim_boundary"],
    },
    "operator_alignment_metric_robustness": {
        "proposal": "transfer-proxy and residual-proxy robustness under the same alignment law",
        "apis": ["operator_alignment_metric_robustness_report", "run_operator_alignment_metric_robustness.py"],
        "report_fields": ["rows", "claim_boundary"],
    },
    "operator_alignment_decoupling": {
        "proposal": "two-axis ablation separating same-spectrum alignment variation from fixed-alignment spectrum variation",
        "apis": ["operator_alignment_decoupling_report", "run_operator_alignment_decoupling.py"],
        "report_fields": ["fixed_spectrum_alignment_rows", "fixed_alignment_spectrum_rows", "summary"],
    },
    "operator_alignment_empirical_bridge": {
        "proposal": "PushT frozen-latent bridge from operator alignment to random-control gaps and transport",
        "apis": ["run_operator_alignment_bridge.py"],
        "report_fields": ["primary_lifts", "control_lifts", "transport", "planner_weights"],
    },
    "sigreg_local_anisotropy": {
        "proposal": "SIGReg can flatten global spectra while local anisotropy remains",
        "apis": ["sigreg_local_anisotropy_report"],
        "report_fields": ["sigreg_local_anisotropy"],
    },
    "identifiability_controllability": {
        "proposal": "planning needs controllability alignment, not only identifiability",
        "apis": ["identifiability_controllability_report"],
        "report_fields": ["identifiability_controllability"],
    },
    "controllability_alignment_sweep": {
        "proposal": "side branch: fixed spectrum with rotated controllable subspace",
        "apis": ["controllability_alignment_sweep_report"],
        "report_fields": ["records", "random_rotation_null", "contrast"],
    },
    "cross_model_temporal_geometry": {
        "proposal": "held-out covariance--operator geometry comparison across frozen JEPA-WM and DINO-WM checkpoints",
        "apis": ["run_cross_model_temporal_geometry_audit.py", "run_audit", "_fit_operator"],
        "report_fields": ["records", "summary", "cross_model_association", "source_protocols"],
        "boundary": "frozen temporal diagnostic only; cross-model association is descriptive and not online-control or universal model-quality evidence",
    },
    "residual_band_training_objective": {
        "proposal": "residual-band regularization primitive",
        "apis": ["residual_band_penalty"],
        "report_fields": ["band_energy"],
    },
}


def theory_alignment_keys() -> set[str]:
    return set(THEORY_PROPOSAL_ALIGNMENT)
