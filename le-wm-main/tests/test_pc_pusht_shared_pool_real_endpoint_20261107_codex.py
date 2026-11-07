import numpy as np

from experiments.pc_pusht_shared_pool_real_endpoint_20261107_codex import summarize


def test_success_rank_pair_counts_and_ties():
    result = summarize(np.array([0., 2., 1., 1.]),
                       np.array([False, True, True, False]),
                       np.array([False, False, True, False]))
    assert result["winner"] == 0
    assert result["ever_best_rank"] == 2
    assert result["ever_pair_correct"] == 0
    assert result["ever_pair_ties"] == 1
    assert result["ever_pair_total"] == 4
    assert result["final_best_rank"] == 2


def test_no_success_has_no_best_rank_or_pairs():
    result = summarize(np.array([1., 2.]), np.array([False, False]), np.array([False, False]))
    assert result["ever_best_rank"] is None
    assert result["final_pair_total"] == 0
