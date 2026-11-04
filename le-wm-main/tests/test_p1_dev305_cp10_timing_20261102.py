"""Regression checks for sealed cp10 objective-time development controls."""
from experiments.spectral_reachability import p1_dev305_cp10_tail_pair_saved_20261102 as pair
from experiments.spectral_reachability import p1_dev305_cp10_prefix_cem_saved_20261102 as prefix


def test_tail_pair_contains_two_replayed_physical_comparisons():
    result = pair.analyze()
    assert len(result["rows"]) == 2
    assert not result["independent_confirmation"]
    assert not result["full_cost_certified"]


def test_prefix_score_physical_receipt_and_budget():
    result = prefix.analyze()
    assert len(result["rows"]) == 2
    assert result["frames_verified"] == 12
    assert all(not row["native_success"] for row in result["rows"])
    assert not result["full_cost_certified"]
