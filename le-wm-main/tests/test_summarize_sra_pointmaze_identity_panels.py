import numpy as np

from experiments.spectral_reachability.summarize_sra_pointmaze_identity_panels import _bootstrap_mean, _stage_gate_reason


def test_identity_panel_bootstrap_contains_sample_mean():
    values = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    low, high = _bootstrap_mean(values, seed=4, draws=200)
    assert low <= 0.0 <= high


def test_stage_gate_reason_reports_consistent_p1_failure_without_claiming_sign_disagreement():
    reason = _stage_gate_reason(stable_p1=False, stable_p2=False, p1_panel_means=[0.05, 0.20])
    assert "does not establish a spectral-selection advantage" in reason
    assert "signs disagree" not in reason
