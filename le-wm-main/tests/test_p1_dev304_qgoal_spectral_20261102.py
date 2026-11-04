from experiments.spectral_reachability import p1_dev304_qgoal_spectral_saved_20261102 as saved


def test_qgoal_physical_panel_integrity():
    result = saved.analyze()
    assert len(result["rows"]) == 18
    assert result["frames_verified"] == 252
    assert sum(row["model_selected"] for row in result["rows"]) == 2
    assert not result["p1_admission"]
    assert not result["independent_confirmation"]
    assert not result["full_cost_certified"]
