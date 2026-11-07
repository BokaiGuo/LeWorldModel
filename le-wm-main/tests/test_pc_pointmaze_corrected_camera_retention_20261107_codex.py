import numpy as np
import pytest

from experiments.pc_pointmaze_corrected_camera_retention_20261107_codex import bank, choose_retained, matrix_chains


def test_banks_are_bounded_and_preserve_prefix():
    parents, probes, tails = bank()
    chains = matrix_chains(parents, tails)
    assert chains.shape == (8, 4, 25, 2)
    np.testing.assert_array_equal(chains[:, :, :10], np.broadcast_to(parents[:, None, :10], (8, 4, 10, 2)))
    assert max(np.abs(x).max() for x in (parents, probes, tails)) <= .4
    assert not np.array_equal(probes, tails)


def test_retention_is_deterministic_and_fails_closed():
    parent = np.arange(8, dtype=float)
    probes = np.arange(32, dtype=float).reshape(8, 4)
    assert choose_retained("uniform", parent, probes, 1) == [0, 1]
    assert choose_retained("lookahead", parent, probes, 1) == [0, 1]
    assert choose_retained("shuffled", parent, probes, 1) == choose_retained("shuffled", parent, probes, 1)
    with pytest.raises(ValueError):
        choose_retained("lookahead", parent, probes[:7], 1)
