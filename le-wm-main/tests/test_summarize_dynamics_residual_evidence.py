from pathlib import Path

from experiments.empirical_spectral_audit.summarize_dynamics_residual_evidence import run_report


ROOT = Path(__file__).resolve().parents[1]


def test_current_dynamics_residual_evidence_summary_is_claim_complete():
    report = run_report(
        ROOT / "results/pusht/lewm/dynamics_subspace_robustness_20261003/dynamics_subspace_robustness.json",
        ROOT / "results/pusht/lewm/residual_subspace_robustness_20261003/residual_subspace_robustness.json",
        ROOT / "results/pusht/lewm/external_public_pusht_temporal_20261003/external_physical_dynamics_robustness.json",
        ROOT / "results/pusht/lewm/external_public_pusht_temporal_20261003/external_residual_transport_robustness.json",
        ROOT / "results/nonlinear_planner_metric_ablation_20261003/residual_strength_calibration_audit.json",
        ROOT / "results/nonlinear_planner_metric_ablation_20261003/residual_candidate_sensitivity.json",
    )
    evidence = report["evidence"]
    assert evidence["latent_dynamics_readout"]["above_covariance_all_splits"]
    assert evidence["external_rollout_sensitivity"]["ci_excludes_zero_all_splits"]
    assert evidence["external_physical_motion"]["predicted_incremental_r2_over_action"] < 0.0
    assert evidence["synthetic_residual_planner"]["candidate_count_sign_reversal"]
