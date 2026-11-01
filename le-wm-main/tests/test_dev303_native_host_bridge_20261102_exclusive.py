import numpy as np
import pytest
import torch

from experiments.spectral_reachability import dev303_native_host_bridge_20261102_exclusive as m


class Model:
    def action_encoder(self, x):
        return x

    def predict(self, states, actions):
        return states


def test_live_feedback_context_and_actual_actions():
    events = []
    captures = []
    def capture(step):
        captures.append(step)
        return np.full((4, 4, 3), step, np.uint8)
    def encode(rgb):
        return torch.full((2,), float(rgb[0, 0, 0]))
    host = m.Host(Model(), encode, capture, lambda action: (action.copy(), False),
                  [torch.tensor([float(i)] * 2) for i in range(3)],
                  torch.zeros(2), np.zeros((10, 2), np.float32),
                  torch.zeros(2), torch.ones(2), events.append, "a" * 64)
    result = host.run(seed=3)
    assert len(result["actual_actions"]) == 15
    assert captures == [5]
    assert host.cost["rgb_encode_returned"] == 1
    assert host.cost["control_returned"] == 15
    assert host.cost["predict_returned"] == 325
    assert host.cost["action_encode_returned"] == 325
    assert events.index(next(e for e in events if e["type"] == "control_return" and e["step"] == 5)) < events.index(next(e for e in events if e["type"] == "feedback" and e["step"] == 5))


def test_stop_and_wrong_action_fail_closed():
    def host(step):
        return m.Host(Model(), lambda rgb: torch.zeros(2),
                      lambda _: np.zeros((4, 4, 3), np.uint8), step,
                      [torch.zeros(2) for _ in range(3)], torch.zeros(2),
                      np.zeros((10, 2), np.float32), torch.zeros(2), torch.ones(2),
                      lambda event: None, "a" * 64)
    early = host(lambda a: (a, True))
    assert len(early.run(seed=1)["actual_actions"]) == 1
    assert early.cost["rgb_encode_attempted"] == 0
    bad = host(lambda a: (a + 1, False))
    with pytest.raises(ValueError, match="action mismatch"):
        bad.run(seed=1)
    assert bad.cost["control_attempted"] == 1
    assert bad.cost["control_returned"] == 0
