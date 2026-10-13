import math

from experiments.empirical_spectral_audit.run_nonlinear_alignment_phase_matched_synthetic import (
    run_phase,
)


def test_matched_phase_smoke_has_checkpoint_audit_and_paired_metrics():
    report = run_phase(
        [1e-3, 1e-1],
        [0],
        epochs=2,
        train_samples=32,
        validation_samples=24,
        test_samples=32,
        local_anchors=16,
    )
    assert len(report["records"]) == 4
    assert len(report["paired_inference"]) == 24
    assert all(row["relative_validation_gap"] >= 0.0 for row in report["records"])
    assert all(
        math.isfinite(float(row["mean_delta_target_minus_none"]))
        for row in report["paired_inference"].values()
    )
