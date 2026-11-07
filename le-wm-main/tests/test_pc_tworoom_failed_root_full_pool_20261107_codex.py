import numpy as np
import pytest

from experiments.pc_tworoom_failed_root_full_pool_20261107_codex import candidates_for


def test_candidate_reconstruction_preserves_prefix_tail_and_cost():
    generation = np.zeros((300, 25, 2))
    pool = np.zeros((8, 25, 2))
    pool[3, :10] = 0.5
    tails = np.zeros((8, 15, 2))
    tails[4] = -0.25
    matrix = np.arange(64).reshape(8, 8)
    row = {"generation_actions": generation.tolist(), "generation_costs": list(range(300)),
           "pool": pool.tolist(), "parent_cost": list(range(8)),
           "a_tails": tails.tolist(), "a_cost": matrix.tolist()}
    candidates = candidates_for(row)
    assert len(candidates) == 368
    name, chain, cost = candidates[304 + 3 * 8 + 4]
    assert name == "cross_3_4"
    np.testing.assert_array_equal(chain[:10], pool[3, :10])
    np.testing.assert_array_equal(chain[10:], tails[4])
    assert cost == 28


def test_candidate_reconstruction_rejects_incomplete_pool():
    row = {"generation_actions": [], "generation_costs": [], "pool": [],
           "a_tails": [], "a_cost": []}
    with pytest.raises(ValueError):
        candidates_for(row)
