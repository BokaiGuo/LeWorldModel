import numpy as np
from experiments.spectral_reachability.p1_candidate_intervention_20261101_main import schedule


def test_full_roster_and_shared_tails():
    bank = np.arange(240, dtype=np.float32).reshape(8, 3, 5, 2) / 256
    rows = schedule(bank, 7)
    assert len(rows) == 24
    for index in range(8):
        own, common, zero = rows[index * 3:index * 3 + 3]
        for row in (own, common, zero):
            np.testing.assert_array_equal(row["commands"][:5], bank[index, 0])
        np.testing.assert_array_equal(own["commands"][5:], bank[index, 1:].reshape(10, 2))
        np.testing.assert_array_equal(common["commands"][5:], bank[7, 1:].reshape(10, 2))
        np.testing.assert_array_equal(zero["commands"][5:], np.zeros((10, 2)))
