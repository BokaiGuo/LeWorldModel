"""Regression checks for the saved three-prefix development run."""
from experiments.spectral_reachability import p1_dev305_three_prefix_saved_20261102 as saved


def test_saved_three_prefix_roster_and_censoring():
    result = saved.analyze()
    assert [row["arm"] for row in result["rows"]] == ["candidate_02", "candidate_00", "zero"]
    assert result["frames_verified"] == 18
    assert all(row["status"] == "fixed_horizon_completed" and row["steps"] == 15
               and not row["terminated"] and not row["truncated"] for row in result["rows"])
    assert not result["independent_confirmation"]
    assert not result["full_cost_certified"]
