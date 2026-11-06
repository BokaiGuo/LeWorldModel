import numpy as np

from experiments import pc_jepa_visible_local_bank_dev_20261105_codex as run


def test_visible_bank_is_deterministic_bounded_and_packet_constant():
    bank = run.bank()
    assert bank.shape == (65, 15, 2)
    assert bank.dtype == np.float32
    np.testing.assert_array_equal(bank, run.bank())
    assert np.all(np.abs(bank) <= 1)
    assert len({chain.tobytes() for chain in bank}) == 65
    for offset in (0, 5, 10):
        np.testing.assert_array_equal(bank[:, offset:offset + 5],
                                      np.repeat(bank[:, offset:offset + 1], 5, axis=1))


def test_near_candidates_condition_on_reference_but_far_candidates_are_distinct():
    bank = run.bank()
    assert np.max(np.abs(bank[:32] - bank[64])) <= .010001
    assert not np.any(np.all(bank[:64] == bank[64], axis=(1, 2)))
