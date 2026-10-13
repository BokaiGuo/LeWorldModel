import json
from pathlib import Path

from experiments.empirical_spectral_audit.summarize_deep_claim_matrix import run_report


ROOT = Path(__file__).resolve().parents[1]


def test_deep_claim_matrix_preserves_separate_estimands(tmp_path):
    output = tmp_path / "deep_claim_matrix.json"
    report = run_report(ROOT, output)

    assert report["status"] == "complete"
    assert len(report["claim_matrix"]) == 9
    assert report["external_transport"]["public_all_supported"]
    assert report["external_transport"]["raw_all_inconclusive"]
    assert report["external_horizon_transport"]["all_pairs_positive_vs_covariance"]
    assert report["external_horizon_transport"]["all_pairs_positive_vs_random"]
    assert report["external_action_residual_transport"]["all_arms_positive_vs_covariance"]
    assert report["external_action_residual_transport"]["all_arms_positive_vs_random"]
    assert report["joint_alignment_equivariance"]["predictor_plus_alignment"]["classification"][
        "all_metrics_all_seed_better"
    ] is False
    assert report["matched_nonlinear_alignment"]["conditional:0.01"]["classification"][
        "label"
    ] == "geometry-only improvement"
    assert report["composition_aware_training"]["composition_switch_mse"][
        "mean_delta_target_minus_baseline"
    ] > 0.0
    assert output.exists()
    json.loads(output.read_text())
