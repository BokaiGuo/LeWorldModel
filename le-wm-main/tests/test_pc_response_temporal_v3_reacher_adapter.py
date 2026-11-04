import numpy as np
import pytest
import torch

from experiments.spectral_reachability import pc_response_temporal_v3_reacher_adapter as adapter


def test_raw_packet_time_order_and_inverse():
    raw = np.linspace(-1, 1, 60, dtype=np.float32).reshape(2, 15, 2)
    mean = np.linspace(-.1, .1, 10, dtype=np.float32)
    std = np.linspace(.5, 1.5, 10, dtype=np.float32)
    packet = adapter.raw_to_packets(raw, mean, std, device="cpu")
    recovered = (packet.numpy() * std + mean).reshape(raw.shape)
    np.testing.assert_allclose(recovered, raw, rtol=0, atol=1e-7)
    with pytest.raises(ValueError):
        adapter.raw_to_packets(np.full((1, 5, 2), 2), mean, std, device="cpu")


class TinyModel:
    def action_encoder(self, actions):
        return actions

    def predict(self, embeddings, actions):
        return embeddings + actions


def test_cached_rollout_obeys_causal_history():
    encoded = torch.zeros((1, 3, 10))
    history = torch.zeros((1, 3, 10))
    future = torch.stack([torch.full((10,), x) for x in (.1, .2, .3)])[None]
    out = adapter.cached_terminal(TinyModel(), encoded, history, future,
                                  torch.full((10,), -1.), torch.full((10,), 1.))
    torch.testing.assert_close(out, torch.full((1, 10), .6))


def test_chart_is_orthonormal_and_seeded():
    encoded = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    goal = torch.tensor([1., 2., 3., 5.])
    basis = adapter.goal_chart(encoded, goal, 42)
    torch.testing.assert_close(basis.T @ basis, torch.eye(2), rtol=0, atol=1e-6)
    torch.testing.assert_close(basis, adapter.goal_chart(encoded, goal, 42))
