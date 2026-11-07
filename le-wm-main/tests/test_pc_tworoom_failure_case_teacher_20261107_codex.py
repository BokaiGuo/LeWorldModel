import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from experiments.pc_tworoom_failure_case_teacher_20261107_codex import packets, teacher


def test_packet_roundtrip_and_chronology():
    raw = np.linspace(-0.9, 0.9, 50, dtype=np.float32).reshape(1, 25, 2)
    scaler = StandardScaler().fit(np.array([[-1, -2], [1, 2]], dtype=np.float32))
    packed, error = packets(raw, scaler)
    assert error < 1e-6
    np.testing.assert_allclose(packed[0, 1], scaler.transform(raw[0, 5:10]).reshape(10))


def test_teacher_uses_matching_recent_history():
    class Model:
        action_encoder = staticmethod(lambda x: x)
        @staticmethod
        def predict(observations, actions):
            return observations + actions[..., :1]
    observations = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    actions = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1).expand(1, 5, 10)
    result = teacher(Model(), observations, actions)
    assert result.flatten().tolist() == [0, 2, 4, 6, 8]


def test_packet_rejects_illegal_action():
    with pytest.raises(ValueError):
        packets(np.full((1, 25, 2), 2), StandardScaler())
