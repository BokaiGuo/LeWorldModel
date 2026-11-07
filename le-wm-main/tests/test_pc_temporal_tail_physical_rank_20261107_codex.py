import numpy as np

from experiments.pc_temporal_tail_physical_rank_20261107_codex import metrics


def test_ranking_and_unfinished_prefix_filter():
    p = np.tile(np.arange(8, dtype=float), (8, 1))
    d = p.copy()
    success = p < 3
    eligible = np.array([True, False, False, False, False, False, False, False])
    result = metrics(p, d, success, eligible)
    assert result["valid_pairs"] == result["correct_pairs"] == 28
    assert result["chosen_success_mixed"] == 1
    assert result["eligible_prefixes"] == 1


def test_model_and_physical_ties_are_not_correct_pairs():
    p = np.zeros((8, 8))
    d = np.zeros((8, 8))
    result = metrics(p, d, np.zeros((8, 8), dtype=bool), np.ones(8, dtype=bool))
    assert result["valid_pairs"] == 0
    assert result["pred_tie_pairs"] == result["physical_tie_pairs"] == 8 * 28
