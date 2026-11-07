"""Expose the installed LeWM rollout's returned prediction timing."""
import torch
from torch import nn

from stable_worldmodel.wm.lewm.lewm import LeWM


class TinyLeWM(LeWM):
    def __init__(self):
        super().__init__(nn.Linear(1, 1), nn.Identity(), nn.Identity())
        self.calls = []

    def encode(self, info):
        info["emb"] = torch.zeros((info["pixels"].shape[0], 1, 1))
        return info

    def predict(self, emb, action_emb):
        self.calls.append((emb.shape[1], action_emb.shape[1]))
        return emb + action_emb


def test_one_packet_returns_one_successive_prediction():
    model = TinyLeWM()
    info = {"pixels": torch.zeros((1, 1, 1, 1, 1, 1))}
    out = model.rollout(info, torch.tensor([[[[3.0]]]]))
    assert model.calls == [(1, 1)]
    torch.testing.assert_close(out["predicted_emb"].flatten(), torch.tensor([0., 3.]))


def test_two_packets_return_two_successive_predictions():
    model = TinyLeWM()
    info = {"pixels": torch.zeros((1, 1, 1, 1, 1, 1))}
    out = model.rollout(info, torch.tensor([[[[3.0], [4.0]]]]))
    assert model.calls == [(1, 1), (2, 2)]
    torch.testing.assert_close(out["predicted_emb"].flatten(), torch.tensor([0., 3., 7.]))
