import numpy as np

from experiments.pc_jepa_v22_unqueried_suffix_dev_20261105_codex import make_bank


def test_bank_is_deterministic_and_has_shared_disjoint_suffixes():
    a = np.full((15, 2), 0.25, np.float32)
    b = np.full((15, 2), -0.25, np.float32)
    prefixes, suffixes, bank = make_bank(a, b)
    assert bank.shape == (64, 15, 2)
    assert np.array_equal(bank, make_bank(a, b)[2])
    assert np.max(np.abs(bank)) <= 1
    assert len({x.tobytes() for x in suffixes[1:8]}) == 7
    assert not ({x.tobytes() for x in suffixes[1:8]} & {x.tobytes() for x in suffixes[8:]})
    for i in range(4):
        assert np.array_equal(bank[i*16:(i+1)*16, :5], np.broadcast_to(prefixes[i], (16, 5, 2)))
        assert np.array_equal(bank[i*16:(i+1)*16, 5:], suffixes)
