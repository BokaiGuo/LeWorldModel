import pytest

from experiments.pc_conditional_failure_pool_20261107_codex import summarize


def trace(success, final=None):
    if final is None:
        final = success
    return [{"terminated": success}, {"terminated": final}]


def test_success_exists_but_selected_failed():
    result = summarize([1, 2, 3], [trace(False), trace(True, False), trace(False)], 0)
    assert result["coverage_but_selection_failed"]
    assert result["ever_success_count"] == 1
    assert result["final_success_count"] == 0
    assert result["best_success_score_rank_one_based"] == 2


def test_no_success_and_stable_ties():
    result = summarize([1, 1], [trace(False), trace(False)], 0)
    assert result["best_success_score_rank_one_based"] is None
    assert not result["coverage_but_selection_failed"]
    with pytest.raises(ValueError):
        summarize([1, 1], [trace(False), trace(False)], 1)


def test_nonfinite_scores_fail():
    with pytest.raises(ValueError):
        summarize([float("nan")], [trace(False)], 0)
