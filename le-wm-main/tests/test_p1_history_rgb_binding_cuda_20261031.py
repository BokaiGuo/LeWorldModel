import numpy as np
import pytest

from experiments.spectral_reachability.p1_history_rgb_binding_cuda_20261031 import normalized_actions


def test_normalization_keeps_packet_time_order():
    raw = np.arange(20, dtype=np.float32).reshape(10, 2)
    actual = normalized_actions(raw, [1, 2], [2, 4])
    expected = ((raw - np.array([1, 2], np.float32)) / np.array([2, 4], np.float32)).reshape(2, 10)
    assert actual.tobytes() == expected.tobytes()
    assert actual[0, 2] == 0.5 and actual[1, 0] == 4.5


@pytest.mark.parametrize("raw,mean,std", [
    (np.zeros((9, 2)), [0, 0], [1, 1]),
    (np.zeros((10, 2)), [0, 0], [1, 0]),
    (np.zeros((10, 2)), [0, float("nan")], [1, 1]),
])
def test_rejects_invalid_history(raw, mean, std):
    with pytest.raises(ValueError):
        normalized_actions(raw, mean, std)
