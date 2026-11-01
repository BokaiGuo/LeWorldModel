import numpy as np
import pytest

from experiments.spectral_reachability.e3_cp10_dls_feedback_mix_20261031 import mix_bank


def test_mix_is_fixed_bounded_and_preserves_four_dls_candidates():
    bank = np.arange(8 * 3 * 5 * 2, dtype=np.float32).reshape(8, 3, 5, 2) / 240
    first = mix_bank(bank, 5)
    assert first.shape == bank.shape and first.dtype == np.float32
    assert np.array_equal(first[:4], bank[[3, 5, 6, 7]])
    assert np.array_equal(first, mix_bank(bank, 5))
    assert not np.array_equal(first[4:], mix_bank(bank, 10)[4:])
    assert np.max(np.abs(first)) <= 1


def test_malformed_and_unsupported_inputs_rejected():
    with pytest.raises(ValueError):
        mix_bank(np.zeros((8, 3, 5, 2), dtype=np.float32), 0)
    with pytest.raises(ValueError):
        mix_bank(np.zeros((8, 3, 5, 2), dtype=np.float64), 5)
