import numpy as np
import pytest

from experiments.pc_tworoom_local_b_tail_20261107_codex import local_chains


def test_local_candidates_retain_prefix_center_identity_and_bounds():
    rng = np.random.default_rng(3)
    prefixes = rng.uniform(-1, 1, (8, 10, 2)).astype(np.float32)
    blocks = rng.uniform(-1, 1, (8, 8, 3, 2)).astype(np.float32)
    a = np.concatenate((np.broadcast_to(prefixes[:, None], (8, 8, 10, 2)),
                        np.repeat(blocks, 5, axis=2)), axis=2)
    costs = rng.uniform(size=(8, 8))
    actions, centers = local_chains(a, costs, [3, 6], 22)
    assert actions.shape == (16, 25, 2)
    assert centers == [int(np.argmin(costs[3])), int(np.argmin(costs[6]))]
    assert np.array_equal(actions[:8, :10], np.broadcast_to(prefixes[3], (8, 10, 2)))
    assert np.array_equal(actions[8:, :10], np.broadcast_to(prefixes[6], (8, 10, 2)))
    tail = actions[:, 10:].reshape(16, 3, 5, 2)
    assert np.array_equal(tail, np.broadcast_to(tail[:, :, :1], tail.shape))
    assert np.max(np.abs(actions)) <= 1
    assert np.array_equal(actions, local_chains(a, costs, [3, 6], 22)[0])


def test_invalid_tail_and_duplicate_prefix_rejected():
    a = np.zeros((8, 8, 25, 2), dtype=np.float32)
    a[0, 0, 11, 0] = 0.5
    with pytest.raises(ValueError):
        local_chains(a, np.zeros((8, 8)), [0, 1], 1)
    with pytest.raises(ValueError):
        local_chains(a, np.zeros((8, 8)), [1, 1], 1)
