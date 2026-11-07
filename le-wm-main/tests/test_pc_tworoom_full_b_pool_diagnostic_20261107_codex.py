import numpy as np
import pytest

from experiments.pc_tworoom_full_b_pool_diagnostic_20261107_codex import summarize_root


def test_fixed_pool_rank_and_query_coverage():
    scores = np.arange(64, dtype=float)
    success = np.zeros(64, dtype=bool)
    success[[5, 12]] = True
    result = summarize_root(scores, success, [2, 12], 8.0)
    assert result["pool_success_count"] == 2
    assert result["queried_success_count"] == 1
    assert result["best_success_rank_1based"] == 6
    assert result["best_success_score_minus_incumbent"] == -3
    assert not result["model_best_success"]


def test_stable_tie_and_absent_success():
    result = summarize_root(np.zeros(64), np.zeros(64), [], 0)
    assert result["model_best_id"] == 0
    assert result["best_success_rank_1based"] is None
    assert result["best_success_score_minus_incumbent"] is None


def test_invalid_candidates_fail_closed():
    with pytest.raises(ValueError):
        summarize_root(np.zeros(64), np.zeros(64), [3, 3], 0)
