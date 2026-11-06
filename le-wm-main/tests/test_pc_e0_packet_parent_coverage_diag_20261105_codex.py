import numpy as np
from experiments.pc_e0_packet_parent_coverage_diag_20261105_codex import packet_bank


def test_packet_bank_bounded_repeatable_and_constant():
    x = packet_bank(42)
    assert x.shape == (64, 15, 2) and np.abs(x).max() <= 1
    assert np.count_nonzero(x[0]) == 0
    np.testing.assert_array_equal(x, packet_bank(42))
    for t in (0, 5, 10):
        np.testing.assert_array_equal(x[:, t:t+5], np.repeat(x[:, t:t+1], 5, axis=1))
