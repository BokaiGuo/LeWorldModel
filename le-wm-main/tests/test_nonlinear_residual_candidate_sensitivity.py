from experiments.empirical_spectral_audit.run_nonlinear_residual_candidate_sensitivity import (
    NonlinearLocalConfig,
    _metric_names,
    run_report,
)


def test_candidate_sensitivity_metric_protocol_is_complete():
    names = _metric_names()
    assert names[0] == "euclidean"
    assert len(names) == 9
    assert len(set(names)) == len(names)
    assert "heldout_calibration:aligned_a1" in names
    assert "heldout_calibration:random_basis_a1" in names


def test_candidate_sensitivity_smoke_report_is_complete():
    report = run_report(
        NonlinearLocalConfig(
            epochs=2,
            train_samples=16,
            validation_samples=8,
            test_samples=16,
            local_anchors=4,
        ),
        seeds=[0, 1],
        horizons=[2],
        candidate_counts=[4, 8],
        tasks=2,
        calibration_samples=16,
    )
    assert len(report["records"]) == 2 * 2 * 2 * 1 * 2 * 9
    assert len(report["focused_ood_aligned_minus_random"]) == 2
    assert all(record["planning"]["planning_regret"] == record["planning"]["planning_regret"] for record in report["records"])
