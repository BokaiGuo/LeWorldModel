"""CPU-only checks for the existing moving witness and exported snapshot primitives."""
import numpy as np
import pytest

from experiments.spectral_reachability import p1_constructive_moving_history_source_20261027_v1 as moving


@pytest.mark.parametrize("sign", [1, -1])
def test_witness_is_fixed_float32_geometry(sign):
    state = np.array([0., 0., 2., 0., 0., 0., 0.])
    logits, actions = moving.witness(state, sign, .75 * np.sqrt(5))
    expected = np.array([[.5, 0.]] * 5 + [[.25, sign * .5]] * 5, dtype=np.float32)
    assert logits.dtype == actions.dtype == np.float32
    np.testing.assert_array_equal(logits, expected)
    np.testing.assert_array_equal(actions, np.tanh(expected))


def test_witness_rejects_degenerate_and_cap():
    with pytest.raises((ValueError, AssertionError)):
        moving.witness(np.zeros(7), 1, .75 * np.sqrt(5))
    with pytest.raises((ValueError, AssertionError)):
        moving.witness(np.array([0., 0., 2., 0., 0., 0., 0.]), 1, .1)
