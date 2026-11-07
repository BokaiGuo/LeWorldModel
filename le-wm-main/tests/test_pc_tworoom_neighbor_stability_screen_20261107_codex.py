import numpy as np
import pytest

from experiments.pc_tworoom_neighbor_stability_screen_20261107_codex import select


def test_neighbors_exclude_self_and_costs_remain_within_prefix():
    tails = np.zeros((8, 3, 2))
    tails[:, 0, 0] = np.arange(8) / 10
    costs = np.repeat(np.arange(8)[:, None] * 100, 8, axis=1).astype(float)
    result = select(tails, costs, 1)
    for i, neighbors in enumerate(result["nearest"]):
        assert i not in neighbors
        assert len(set(neighbors)) == 2
    np.testing.assert_allclose(result["neighbor_scores"], costs)
    assert result["baseline"] == result["neighbor"] == 0


def test_random_control_reproducible_and_rejects_invalid_actions():
    tails = np.zeros((8, 3, 2))
    costs = np.arange(64).reshape(8, 8)
    assert select(tails, costs, 3) == select(tails, costs, 3)
    tails[0, 0, 0] = 1.1
    with pytest.raises(ValueError, match="illegal input"):
        select(tails, costs, 3)
