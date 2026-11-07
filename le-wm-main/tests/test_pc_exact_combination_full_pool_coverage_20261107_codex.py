import numpy as np
import pytest

from experiments.pc_exact_combination_full_pool_coverage_20261107_codex import summarize


def test_pool_coverage_distinguishes_unqueried_success():
    b = np.ones((8, 64))
    b[7, 63] = .01
    result = summarize(np.full(8, .4), np.full((8, 8), .3), b,
                       {"miss": list(range(8)), "hit": [511, 1, 2, 3, 4, 5, 6, 7]})
    assert result["pool_has_new_success"]
    assert not result["arms"]["miss"]["found_new_success"]
    assert result["arms"]["hit"]["found_new_success"]
    assert result["arms"]["miss"]["regret_to_pool_best"] == pytest.approx(.29)


def test_duplicate_query_rejected_and_initial_success_not_new():
    b = np.zeros((8, 64))
    assert not summarize(np.zeros(8), np.zeros((8, 8)), b, {"ok": list(range(8))})["pool_has_new_success"]
    with pytest.raises(ValueError):
        summarize(np.zeros(8), np.zeros((8, 8)), b, {"bad": [0] * 8})
