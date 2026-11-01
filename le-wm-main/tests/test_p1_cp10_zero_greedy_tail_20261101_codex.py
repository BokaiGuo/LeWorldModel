import numpy as np

from experiments.spectral_reachability import p1_cp10_zero_greedy_tail_20261101_codex as m


def test_schedule():
    result = m.schedule(np.zeros((5, 2)), np.ones((5, 2)) / 2, np.ones((10, 2)) / 4)
    assert [(r["prefix_role"], r["suffix_role"]) for r in result] == [
        ("zero", "common"), ("zero", "zero"), ("greedy", "common"), ("greedy", "zero")]
    assert result[0]["suffix"] == result[2]["suffix"]
    assert result[1]["suffix"] == result[3]["suffix"]


def test_real_source_preflight():
    value = m.build()
    assert value["greedy_index"] == int(np.argmin(value["greedy_stage_scores"]))
    assert len(value["schedule"]) == 4
    assert not value["independent_confirmation"]
