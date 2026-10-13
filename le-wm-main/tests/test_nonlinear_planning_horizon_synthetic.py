from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import NonlinearLocalConfig
from experiments.empirical_spectral_audit.run_nonlinear_planning_horizon_synthetic import run_report


def test_planning_horizon_smoke_has_all_conditions_and_horizons():
    report = run_report(
        NonlinearLocalConfig(
            train_samples=32,
            validation_samples=24,
            test_samples=32,
            epochs=2,
            local_anchors=16,
            planning_tasks=4,
            planning_candidates=6,
        ),
        seeds=[0],
        horizons=[2, 4],
    )
    assert len(report["records"]) == 10
    assert set(report["summary"]) == {"none", "global", "conditional", "predictor", "both"}
    assert set(report["summary"]["conditional"]) == {"2", "4"}
