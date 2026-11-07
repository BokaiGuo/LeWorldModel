import numpy as np
import pytest

from experiments.pc_pointmaze_goal25_retention_cem_20261107_codex import cem_candidates


def test_cem_candidates_are_bounded_and_include_mean():
    mean = np.full((25, 2), .25, dtype=np.float32)
    std = np.full((25, 2), .5, dtype=np.float32)
    result = cem_candidates(np.random.default_rng(42), mean, std)
    assert result.shape == (16, 25, 2)
    assert np.abs(result).max() <= 1
    np.testing.assert_array_equal(result[0], mean)


def test_cem_rejects_incorrect_clock():
    with pytest.raises(ValueError):
        cem_candidates(np.random.default_rng(0), np.zeros((24, 2)), np.ones((24, 2)))
