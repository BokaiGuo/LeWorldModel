import numpy as np
import pytest

from experiments.pc_pointmaze_shadow_closed_loop_adapter_20261103_codex import RGBClock


class FakeBridge:
    def __init__(self, wrong_clock=False, corrupt=False):
        self.initial_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        self.step_count = 0
        self.wrong_clock, self.corrupt = wrong_clock, corrupt
        self.received = []

    def step(self, packet):
        self.received.append(packet.copy())
        self.step_count += 2 if self.wrong_clock else 1
        return np.zeros((1,)) if self.corrupt else np.full((224, 224, 3), self.step_count, dtype=np.uint8)


def test_action_and_rgb_feedback_are_bound_and_copied():
    bridge = FakeBridge()
    stream = RGBClock(bridge)
    action = np.array([0.25, -0.5])
    stream.advance(action)
    action[:] = 0
    np.testing.assert_array_equal(bridge.received[0], [[0.25, -0.5]])
    previous, current = stream.pair()
    assert previous[0, 0, 0] == 0 and current[0, 0, 0] == 1
    current[:] = 99
    assert stream.pair()[1][0, 0, 0] == 1
    assert stream.clock == 1


@pytest.mark.parametrize("kwargs", [{"wrong_clock": True}, {"corrupt": True}])
def test_failed_step_poisoned_without_feedback(kwargs):
    stream = RGBClock(FakeBridge(**kwargs))
    with pytest.raises(RuntimeError):
        stream.advance([0, 0])
    with pytest.raises(RuntimeError):
        stream.pair()
    with pytest.raises(RuntimeError):
        stream.advance([0, 0])


def test_invalid_action_rejected_before_step():
    bridge = FakeBridge()
    stream = RGBClock(bridge)
    with pytest.raises(ValueError):
        stream.advance([2, 0])
    assert not bridge.received
    with pytest.raises(RuntimeError):
        stream.pair()
