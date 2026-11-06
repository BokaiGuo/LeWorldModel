import pytest

from experiments.pc_prefix_heldout_allocation_dev_20261106_codex import retention_stats


def test_retention_uses_a_ranking_not_heldout_to_choose():
    result = retention_stats([0, 1, 2, 3], [9, 8, 0, 1], [5, 4, 3, 2], k=2)
    assert result == {"retained": [0, 1], "heldout_topk_hit": 0,
                      "best_heldout_cost": 8.0, "best_optimized_cost": 4.0}


def test_rejects_nonfinite_or_misaligned_input():
    with pytest.raises(ValueError):
        retention_stats([0, 1], [0], [0], k=1)
    with pytest.raises(ValueError):
        retention_stats([float("nan")], [0], [0], k=1)
