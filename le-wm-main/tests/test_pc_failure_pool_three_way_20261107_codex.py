import pytest

from experiments.pc_failure_pool_three_way_20261107_codex import label_metrics


def test_pairwise_and_oracle_rank():
    result = label_metrics([0, 3, 2, 2], [False, True, True, False])
    assert result["best_success_rank"] == 2
    assert result["pair_count"] == 4
    assert result["success_better_pairs"] == 0
    assert result["tied_pairs"] == 1
    assert result["success_worse_pairs"] == 3
    assert not result["selected_success"]


def test_no_success_and_nonfinite():
    result = label_metrics([1, 2], [False, False])
    assert result["best_success_rank"] is None and result["pair_count"] == 0
    with pytest.raises(ValueError):
        label_metrics([float("nan")], [True])
