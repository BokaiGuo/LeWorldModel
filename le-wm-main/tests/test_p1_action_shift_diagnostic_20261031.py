import numpy as np
import pytest

from experiments.spectral_reachability.p1_action_shift_diagnostic_20261031 import action_variants


def test_shift_uses_exact_neighbor_actions():
    prefix = (np.arange(10, dtype=np.float32) / 50).reshape(5, 2)
    future = (np.arange(20, dtype=np.float32) / 50).reshape(2, 5, 2)
    variants = action_variants(prefix, future)
    flat = future.reshape(10, 2)
    assert variants["correct"].tobytes() == flat[:5].tobytes()
    assert variants["earlier"].tobytes() == np.concatenate((prefix[-1:], flat[:4])).tobytes()
    assert variants["later"].tobytes() == flat[1:6].tobytes()


def test_shift_rejects_invalid_action_inputs():
    prefix = np.zeros((5, 2), dtype=np.float32)
    future = np.zeros((2, 5, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="ABI"):
        action_variants(prefix[:-1], future)
    future[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="ABI"):
        action_variants(prefix, future)
    future[0, 1, 0] = 1.1
    with pytest.raises(ValueError, match="ABI"):
        action_variants(prefix, future)
