import numpy as np

from experiments.empirical_spectral_audit.run_data_efficiency_synthetic import run_report


def test_data_efficiency_report_is_finite_on_small_protocol():
    report = run_report([64], [0], epochs=2)
    assert len(report["records"]) == 3
    assert len(report["summary"]) == 3
    assert all(np.isfinite(value["rollout_h8_mse"]["mean"]) for value in report["summary"].values())
