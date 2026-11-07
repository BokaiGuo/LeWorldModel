import numpy as np
import pytest

from experiments.pc_tworoom_local_perturb_stability_20261107_codex import choose, perturb


def test_perturb_preserves_prefix_blocks_and_bounds():
    parents = np.zeros((8, 25, 2), np.float32)
    parents[:, :10] = 0.2
    parents[:, 10:] = 0.99
    result, clipped = perturb(parents, 42)
    assert result.shape == (8, 4, 25, 2)
    assert np.all(result[:, :, :10] == parents[:, None, :10])
    assert np.max(np.abs(result)) <= 1
    assert clipped > 0
    assert np.all(result[:, :, 10:].reshape(8, 4, 3, 5, 2) ==
                  result[:, :, 10:].reshape(8, 4, 3, 5, 2)[:, :, :, :1])


def test_choose_queries_all_candidates_and_deranges_identity():
    costs = np.arange(64, dtype=float)
    child = np.ones((8, 4))
    child[2, 1] = -1
    result = choose(costs, child, 123)
    assert result["direct"] == 64 + 2 * 4 + 1
    assert result["parents"] == list(range(8))
    assert sorted(result["mapping"]) == list(range(8))
    assert all(i != j for i, j in enumerate(result["mapping"]))
    np.testing.assert_allclose(result["robust_cost"], (costs[:8] + child.sum(axis=1)) / 5)
    with pytest.raises(ValueError):
        choose(costs, np.full((8, 4), np.nan), 123)
