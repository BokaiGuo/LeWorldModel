import numpy as np

from experiments.pc_action_opportunity_c_probe_dev_20261104_codex import checked_probes
from experiments.pc_action_opportunity_random_probe_dev_20261104_codex import probes


def test_c_probes_are_deterministic_and_disjoint_from_a_b():
    for length in (5, 10):
        c = checked_probes(length)
        assert c.shape == (8, length, 2)
        np.testing.assert_array_equal(c, checked_probes(length))
        for split in (0, 1):
            assert not any(np.array_equal(x, y) for x in c for y in probes(length, split))
        assert np.max(np.abs(c)) <= 0.5
