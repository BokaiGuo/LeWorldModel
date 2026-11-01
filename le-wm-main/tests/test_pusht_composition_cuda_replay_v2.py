"""Array/layout tests only; not live CUDA validation."""
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability.pusht_composition_cuda_replay_v2 import packets, CudaReplay


def test_packet_order_preserved_without_repetition_or_clipping():
    raw = np.arange(20, dtype=np.float32).reshape(10, 2)/20
    np.testing.assert_array_equal(packets(raw).reshape(10, 2), raw)


@pytest.mark.parametrize('value', [np.zeros((4, 2)), np.zeros((5, 3)), np.full((5, 2), np.nan), np.ones((5, 2))*2, np.zeros((5, 2), bool), [['x','y']]*5])
def test_bad_raw_packets(value):
    with pytest.raises(ValueError): packets(value)


def test_reject_cpu_before_touching_model():
    with pytest.raises(ValueError, match='CUDA'):
        CudaReplay(object(), SimpleNamespace(type='cpu'))
