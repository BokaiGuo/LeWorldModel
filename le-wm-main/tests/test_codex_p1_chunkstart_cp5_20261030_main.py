import numpy as np
import pytest

from experiments.spectral_reachability.codex_p1_chunkstart_cp5_20261030_main import initial_actions


def test_chunk_starts_are_fixed_bounded_and_temporally_constant():
    first = initial_actions(0)
    assert first.shape == (4, 2, 5, 2)
    assert np.array_equal(first, initial_actions(0))
    assert not np.array_equal(first, initial_actions(1))
    assert np.all(first[0] == 0)
    assert np.all(np.abs(first) <= 1)
    assert np.all(first == first[:, :, :1, :])


@pytest.mark.parametrize("value", [-1, 8, True, 1.0])
def test_rejects_invalid_logical_pair(value):
    with pytest.raises(ValueError):
        initial_actions(value)
