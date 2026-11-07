import numpy as np
import pytest

from experiments.pc_exact_block_chain_exploration_20261107_codex import block_chains


def test_repeats_five_steps_without_aliasing():
    blocks = np.array([[[0.2, -0.3], [0.1, 0.4], [0., 1.]]])
    chains = block_chains(blocks)
    assert chains.shape == (1, 15, 2)
    for k in range(3):
        np.testing.assert_array_equal(chains[0, k * 5:(k + 1) * 5], np.tile(blocks[0, k], (5, 1)))
    chains[0, 0, 0] = 0.8
    assert blocks[0, 0, 0] == 0.2


def test_rejects_invalid_shapes_and_bounds():
    with pytest.raises(ValueError):
        block_chains(np.zeros((1, 2, 2)))
    with pytest.raises(ValueError):
        block_chains(np.ones((1, 3, 2)) * 1.1)
