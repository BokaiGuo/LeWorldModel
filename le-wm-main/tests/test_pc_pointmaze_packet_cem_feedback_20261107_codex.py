import numpy as np
import pytest

from experiments.pc_pointmaze_packet_cem_feedback_20261107_codex import candidates


def test_old_tail_is_a_scored_legal_candidate():
    rng = np.random.default_rng(5)
    old = np.full((3, 2), .7, np.float32)
    pool = candidates(rng, np.zeros((3, 2)), np.ones((3, 2)), old, True)
    assert pool.shape == (16, 3, 2)
    np.testing.assert_array_equal(pool[0], old)
    assert np.abs(pool).max() <= 1
    with pytest.raises(ValueError):
        candidates(rng, np.zeros((2, 2)), np.ones((2, 2)), old, True)
