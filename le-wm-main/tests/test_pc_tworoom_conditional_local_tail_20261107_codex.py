import numpy as np
import pytest

from experiments.pc_tworoom_conditional_local_tail_20261107_codex import pick, proposals


def test_proposals_fixed_prefix_bounded_and_deterministic():
    rng = np.random.default_rng(7)
    a = np.repeat(rng.uniform(-1, 1, (8, 8, 5, 2)), 5, axis=2).astype(np.float32)
    a[:, :, :10] = a[:, :1, :10]
    cost = rng.random((8, 8))
    p, winners = proposals(a, cost, 10)
    q, w2 = proposals(a, cost, 10)
    np.testing.assert_array_equal(p, q)
    np.testing.assert_array_equal(winners, w2)
    np.testing.assert_array_equal(p[:, :, :10], np.broadcast_to(a[:, :1, :10], (8, 8, 10, 2)))
    assert np.abs(p).max() <= 1
    assert all(np.array_equal(p[:, :, t:t+5], np.broadcast_to(p[:, :, t:t+1], (8, 8, 5, 2)))
               for t in (10, 15, 20))


def test_selection_common_tie_and_nonretained_excluded():
    ids = np.arange(24)
    cost = np.full(24, 5.0)
    cost[16] = -100
    assert pick(5, cost, ids, [0, 1]) == ("common", -1, 5)
    cost[9] = 4
    assert pick(5, cost, ids, [0, 1]) == ("local", 9, 4)
    with pytest.raises(ValueError, match="missing"):
        pick(5, cost[:8], ids[:8], [0, 1])


def test_nonconstant_block_rejected():
    a = np.zeros((8, 8, 25, 2), dtype=np.float32)
    a[0, 0, 11, 0] = 0.1
    with pytest.raises(ValueError, match="block-constant"):
        proposals(a, np.zeros((8, 8)), 10)
