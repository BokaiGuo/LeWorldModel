import numpy as np
import pytest

from experiments.spectral_reachability.p1_live_packet_dispatch_20261031 import LivePacketDispatch


def test_dispatch_uses_actual_new_packet_and_requires_midpoint():
    first = np.arange(10, dtype=np.float32).reshape(5, 2) / 20
    second = np.full((5, 2), -0.75, dtype=np.float32)
    dispatch = LivePacketDispatch(first)
    with pytest.raises(ValueError):
        dispatch.bind_second(second)
    for action in first:
        np.testing.assert_array_equal(dispatch.next_action(), action)
    with pytest.raises(ValueError):
        dispatch.next_action()
    digest = dispatch.bind_second(second)
    assert len(digest) == 64
    with pytest.raises(ValueError):
        dispatch.bind_second(second)
    second[:] = 0  # caller mutation must not alter the dispatched packet
    for _ in range(5):
        np.testing.assert_array_equal(dispatch.next_action(), [-0.75, -0.75])
    with pytest.raises(ValueError):
        dispatch.next_action()


@pytest.mark.parametrize("bad", [np.zeros((4, 2)), np.full((5, 2), np.nan), np.full((5, 2), 1.01)])
def test_reject_bad_packets(bad):
    with pytest.raises(ValueError):
        LivePacketDispatch(bad)
    dispatch = LivePacketDispatch(np.zeros((5, 2)))
    for _ in range(5):
        dispatch.next_action()
    with pytest.raises(ValueError):
        dispatch.bind_second(bad)
    with pytest.raises(ValueError):
        dispatch.next_action()
