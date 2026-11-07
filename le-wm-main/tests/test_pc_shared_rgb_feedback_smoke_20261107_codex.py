import numpy as np
import pytest

from experiments.pc_shared_rgb_feedback_smoke_20261107_codex import build_candidates


def test_old_tail_is_retained_and_padded_not_discarded():
    tail = np.ones((20, 2), dtype=np.float32) * .5
    parents, probes, tails = build_candidates(123, tail)
    np.testing.assert_array_equal(parents[0, :20], tail)
    np.testing.assert_array_equal(parents[0, 20:], 0)
    assert parents.shape == (3, 25, 2) and probes.shape == tails.shape == (2, 15, 2)
    np.testing.assert_array_equal(build_candidates(123, tail)[0], parents)


def test_invalid_tail_rejected():
    with pytest.raises(ValueError):
        build_candidates(123, np.zeros((26, 2)))
