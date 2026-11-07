import numpy as np
import pytest

from experiments.pc_tworoom_first_pool_triple_alignment_20261107_codex import difference


def test_hold_difference_preserves_action_identity():
    a = np.array([[5, 3, 8], [3, 1, 2], [9, 4, 7]])
    np.testing.assert_array_equal(difference(a), [[2, -3], [2, 1], [5, 2]])


def test_invalid_matrix_fails():
    with pytest.raises(ValueError):
        difference(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        difference(np.full((3, 3), np.nan))
