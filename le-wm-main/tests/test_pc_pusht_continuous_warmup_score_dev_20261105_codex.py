import numpy as np

from experiments.pc_pusht_continuous_warmup_score_dev_20261105_codex import bank


def test_fixed_bank_is_reproducible_packet_constant_and_unconditioned():
    actions = bank()
    assert actions.shape == (64, 15, 2)
    assert actions.dtype == np.float32
    np.testing.assert_array_equal(actions, bank())
    assert np.all(actions[0] == 0)
    assert np.max(np.abs(actions)) <= .2
    assert len({a.tobytes() for a in actions}) == 64
    for t in (0, 5, 10):
        np.testing.assert_array_equal(actions[:, t:t+5], np.repeat(actions[:, t:t+1], 5, axis=1))
