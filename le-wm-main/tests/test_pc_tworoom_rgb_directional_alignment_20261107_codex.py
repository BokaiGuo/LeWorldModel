import numpy as np
import pytest

from experiments.pc_tworoom_rgb_directional_alignment_20261107_codex import directions, finite_difference


def test_four_physical_directions_cover_first_and_last_packet():
    bank = directions()
    assert bank.shape == (4, 25, 2)
    assert np.count_nonzero(bank) == 20
    np.testing.assert_array_equal(bank.sum(axis=(1, 2)), np.full(4, 5))
    assert np.all(bank[0, :5, 0] == 1)
    assert np.all(bank[3, 20:, 1] == 1)
    assert not bank[:, 5:20].any()


def test_symmetric_difference_and_nonfinite_rejection():
    assert finite_difference(3., 1., .5) == 2.
    with pytest.raises(ValueError):
        finite_difference(3., float("nan"), .5)
    with pytest.raises(ValueError):
        finite_difference(3., 1., 0.)
