import numpy as np

from experiments.empirical_spectral_audit.run_pixel_sigreg_alignment_phase import run_phase


def test_phase_report_is_finite_on_small_protocol():
    report = run_phase([0.0, 0.01], [0], [0.01], epochs=2)
    assert len(report["records"]) == 6
    assert len(report["summary"]) == 6
    assert all(
        np.isfinite(value["test_one_step_mse"]["mean"])
        for value in report["summary"].values()
    )
