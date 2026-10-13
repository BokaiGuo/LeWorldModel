import math

from experiments.empirical_spectral_audit.run_nonlinear_alignment_phase_synthetic import run_phase


def test_nonlinear_alignment_phase_smoke_has_shared_matrix_and_inference():
    report = run_phase(
        [1e-3, 1e-1],
        [0],
        epochs=2,
        train_samples=32,
        validation_samples=24,
        test_samples=32,
        local_anchors=16,
    )
    assert len(report["records"]) == 5
    assert set(report["protocol"]["modes"]) == {"none", "global", "conditional"}
    assert len(report["paired_inference"]) == 24
    assert all(math.isfinite(float(row["mean_delta_target_minus_none"])) for row in report["paired_inference"].values())
    assert all(row["n_paired_seeds"] == 1 for row in report["paired_inference"].values())
