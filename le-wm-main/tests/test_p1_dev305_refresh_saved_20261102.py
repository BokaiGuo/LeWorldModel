"""Check the sealed same-action refresh analysis and its dependency roster."""
from experiments.spectral_reachability import p1_dev305_refresh_saved_20261102 as saved
from experiments.spectral_reachability import p1_dev305_spectral_cp10_saved_20261102 as feedback


def test_refresh_analysis_pairs_and_call_boundaries():
    result = saved.analyze()
    assert len(result["rows"]) == 18
    assert all(sum(values.values()) == 72 for values in result["ranking"].values())
    assert 0 <= result["refresh_mse_lower_count"] <= 18
    assert not result["independent_confirmation"]


def test_real_feedback_physical_evidence_is_paired_and_development_only():
    result = feedback.analyze()
    assert len(result["rows"]) == 2
    assert result["frames_verified"] == 12
    assert not result["independent_confirmation"]
    assert not result["full_cost_certified"]
