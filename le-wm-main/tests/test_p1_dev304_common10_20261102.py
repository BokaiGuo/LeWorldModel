from experiments.spectral_reachability import p1_dev304_common10_saved_20261102 as saved


def test_common_ten_saved_panel():
    result = saved.analyze()
    assert len(result["rows"]) == 6
    assert result["frames_verified"] == 84
    assert all(row["status"] == "fixed_horizon_completed" and not row["native_success"]
               for row in result["rows"])
    assert not result["p1_admission"] and not result["independent_confirmation"]
