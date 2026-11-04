from experiments.spectral_reachability import p1_dev304_feedback_spectral_saved_20261102 as saved


def test_recorded_spectral_candidates_and_physical_paths():
    result = saved.analyze()
    assert len(result["rows"]) == 18
    assert result["frames_verified"] == 252
    assert all(not row["native_success"] for row in result["rows"])
    assert all(value < .05 for value in result["fd_relative_error"].values())
    assert not result["p1_admission"] and not result["independent_confirmation"]
