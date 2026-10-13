from experiments.empirical_spectral_audit import EMPIRICAL_PROPOSAL_ALIGNMENT
from experiments.theory_spectral_transfer import THEORY_PROPOSAL_ALIGNMENT


def test_theory_proposal_alignment_manifest_covers_required_topics():
    required = {
        "oracle_transfer_operator",
        "encoder_predictor_joint_transfer",
        "multi_horizon_residual_dynamics",
        "invariance_heat_filter",
        "oblique_stress",
        "operator_alignment_noncommutativity",
        "operator_alignment_law_card",
        "operator_alignment_oral_object_card",
        "operator_alignment_paper_spine",
        "operator_alignment_paper_draft",
        "operator_alignment_display_plan",
        "operator_alignment_decoupling",
        "sigreg_local_anisotropy",
        "identifiability_controllability",
        "cross_model_temporal_geometry",
        "residual_band_training_objective",
    }

    assert required <= set(THEORY_PROPOSAL_ALIGNMENT)
    assert all(item["apis"] and item["report_fields"] for item in THEORY_PROPOSAL_ALIGNMENT.values())


def test_empirical_proposal_alignment_manifest_covers_required_topics():
    required = {
        "conditional_latent_spectra",
        "tangent_families",
        "core_metrics",
        "controls",
        "multi_horizon",
        "controllability_observability",
        "eigenspace_transport",
        "planner_interventions",
        "spectral_planner_metric",
        "operator_alignment_bridge",
        "operator_alignment_paper_outline",
        "operator_alignment_paper_spine",
        "spectral_regularization",
        "conditional_operator_alignment",
        "checkpoint_matched_geometry_intervention",
        "nonlinear_alignment_strength_phase",
        "nonlinear_alignment_phase_matched",
        "spectral_exploration",
        "ood_shift",
        "warnings_schema",
    }

    assert required <= set(EMPIRICAL_PROPOSAL_ALIGNMENT)
    assert all(item["apis"] and item["report_fields"] for item in EMPIRICAL_PROPOSAL_ALIGNMENT.values())
