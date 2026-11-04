from experiments.spectral_reachability import p1_dev304_cp10_tail_pair_saved_20261102 as saved


def test_fixed_tail_pair_bindings():
    result = saved.analyze()
    assert len(result["rows"]) == 4
    assert len({row["arm"] for row in result["rows"]}) == 4
    assert not result["p1_admission"]
    assert not result["independent_confirmation"]
    assert not result["full_cost_certified"]
