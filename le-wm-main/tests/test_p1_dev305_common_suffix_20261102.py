"""Sealed fixed common-suffix intervention replay regression."""
from experiments.spectral_reachability import p1_dev305_common_suffix_saved_20261102 as saved


def test_all_fixed_prefix_suffix_pairs_and_censoring():
    result = saved.analyze()
    assert len(result["rows"]) == 30
    assert result["frames_verified"] == 180
    assert {r["tail"] for r in result["rows"]} == {"zero", "goal_original", "goal_coherent"}
    assert sum(row["selected"] for row in result["rows"]) == 3
    assert all(row["status"] == "fixed_horizon_completed" and row["steps"] == 15
               for row in result["rows"])
    assert not result["independent_confirmation"]
