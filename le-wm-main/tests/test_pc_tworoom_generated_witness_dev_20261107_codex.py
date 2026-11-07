import numpy as np
import pytest
from experiments.pc_tworoom_generated_witness_dev_20261107_codex import witness_actions


def test_witness_actions_are_reproducible_bounded_five_step_blocks():
    actions = witness_actions(0)
    assert actions.shape == (25, 2)
    assert np.abs(actions).max() <= 1
    np.testing.assert_array_equal(actions, witness_actions(0))
    assert not np.array_equal(actions, witness_actions(1))
    for i in range(5):
        np.testing.assert_array_equal(actions[5*i:5*i+5], np.tile(actions[5*i], (5, 1)))


@pytest.mark.parametrize("index", [-1, 40, 0.5, True])
def test_invalid_witness_index(index):
    with pytest.raises(ValueError):
        witness_actions(index)
