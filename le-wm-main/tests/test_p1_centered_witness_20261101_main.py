import numpy as np
import pytest

from experiments.spectral_reachability.p1_centered_witness_20261101_main import translated


def test_translation_preserves_relative_geometry_and_nonposition_components():
    state = np.array([410., 450., 335., 386., .69, 3., 37.])
    original = state.copy()
    value = translated(state)
    np.testing.assert_array_equal(value[:4], [330., 350., 255., 286.])
    np.testing.assert_array_equal(value[:2] - value[2:4], state[:2] - state[2:4])
    np.testing.assert_array_equal(value[4:], state[4:])
    np.testing.assert_array_equal(state, original)


@pytest.mark.parametrize("state", [np.zeros(6), np.zeros((1, 7)), [0, 0, 0, 0, 0, 0, np.nan]])
def test_translation_rejects_invalid_state(state):
    with pytest.raises(ValueError):
        translated(state)
