import numpy as np
import pytest

from experiments.spectral_reachability.d2p_pusht_cuda_backend import PushTCudaBackend


def test_backend_rejects_non_cuda_without_importing_a_model():
    backend = PushTCudaBackend.__new__(PushTCudaBackend)
    with pytest.raises(ValueError):
        backend._validate_raw_actions(np.zeros((4, 2)))


def test_backend_action_contract_is_five_step_bounded_packets():
    backend = PushTCudaBackend.__new__(PushTCudaBackend)
    assert backend._validate_raw_actions(np.zeros((5, 2))).shape == (5, 2)
    with pytest.raises(ValueError):
        backend._validate_raw_actions(np.zeros((6, 2)))
    with pytest.raises(ValueError):
        backend._validate_raw_actions(np.full((5, 2), 1.01))
