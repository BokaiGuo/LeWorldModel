from experiments.empirical_spectral_audit.run_checkpoint_matched_geometry_synthetic import (
    NonlinearLocalConfig,
    run_report,
)


def test_checkpoint_matching_uses_two_intervention_comparisons():
    report = run_report(
        NonlinearLocalConfig(
            train_samples=32,
            validation_samples=24,
            test_samples=32,
            epochs=3,
            local_anchors=16,
            rollout_trajectories=8,
            planning_tasks=4,
            planning_candidates=6,
        ),
        seeds=[0],
    )
    assert len(report["records"]) == 2
    assert {row["mode"] for row in report["records"]} == {"global", "conditional"}
    assert all(row["relative_validation_gap"] >= 0.0 for row in report["records"])
    assert set(report["inference"]) == {"global", "conditional"}
