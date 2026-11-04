from experiments.spectral_reachability import p1_dev304_spectral_cp10_saved_20261102 as saved


def test_cp10_feedback_physical_panel():
    result = saved.analyze()
    assert len(result["rows"]) == 4
    assert result["frames_verified"] == 24
    assert not result["p1_admission"]
    assert not result["independent_confirmation"]
    assert not result["full_cost_certified"]
