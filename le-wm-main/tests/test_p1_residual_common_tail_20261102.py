from collections import Counter

from experiments.spectral_reachability import p1_residual_common_tail_20261102 as subject


def test_frozen_crossed_panel_has_identical_tails_and_prefixes():
    contract = subject.build()
    specs = contract["specs"]
    assert len(specs) == 20
    assert Counter(s["prefix_role"] for s in specs) == {
        "selected": 5, "zero": 5, "greedy": 5, "random": 5
    }
    assert Counter(s["suffix_role"] for s in specs) == {
        "selected": 4, "zero": 4, "greedy": 4, "random": 4, "zero_suffix": 4
    }
    for role in ("selected", "zero", "greedy", "random"):
        assert all(s["prefix"] == next(x["prefix"] for x in specs if x["prefix_role"] == role)
                   for s in specs if s["prefix_role"] == role)
    for role in ("selected", "zero", "greedy", "random", "zero_suffix"):
        assert all(s["tail"] == next(x["tail"] for x in specs if x["suffix_role"] == role)
                   for s in specs if s["suffix_role"] == role)
    assert contract["P1_admission"] is False
    assert contract["independent_confirmation"] is False
