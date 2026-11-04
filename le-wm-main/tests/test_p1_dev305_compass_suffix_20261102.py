"""Check the complete fixed compass intervention and preservation of failures."""
from experiments.spectral_reachability import p1_dev305_compass_suffix_saved_20261102 as saved


def test_compass_panel_has_full_paired_roster_and_no_padding():
    result = saved.analyze()
    assert len(result["rows"]) == 170
    assert len({row["prefix"] for row in result["rows"]}) == 10
    assert len({row["tail"] for row in result["rows"]}) == 17
    assert result["frames_verified"] == 1020
    assert all(row["status"] == "fixed_horizon_completed" and row["steps"] == 15
               for row in result["rows"])
    assert not result["independent_confirmation"]
