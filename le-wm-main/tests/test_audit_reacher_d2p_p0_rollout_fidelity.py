import numpy as np
import torch

from experiments.spectral_reachability.audit_reacher_d2p_p0_rollout_fidelity import (
    _contiguous_rgb,
    _mean_squared_terminal_cost,
    _rollout_predictions,
)


class _Model:
    @staticmethod
    def encode(info):
        return {"emb": info["pixels"]}

    @staticmethod
    def action_encoder(actions):
        return actions

    @staticmethod
    def predict(embeddings, actions):
        return embeddings[:, -1:] + actions[:, -1:]


def test_rollout_fidelity_reuses_root_anchored_action_packet_order():
    predictions = _rollout_predictions(
        _Model(),
        torch.tensor([[[1.0], [2.0], [3.0]]]),
        torch.tensor([[[10.0], [20.0], [30.0]]]),
        torch.tensor([[1.0], [2.0]]),
    )
    assert [float(value) for value in predictions] == [4.0, 6.0]


def test_rollout_fidelity_materializes_negative_stride_renderer_rgb():
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    renderer_view = rgb[:, :, ::-1]

    materialized = _contiguous_rgb(renderer_view)

    assert renderer_view.strides[-1] < 0
    assert materialized.flags.c_contiguous
    assert np.array_equal(materialized, renderer_view)


def test_rollout_fidelity_uses_the_runner_terminal_cost_definition():
    cost = _mean_squared_terminal_cost(torch.tensor([3.0, 5.0]), torch.tensor([1.0, 2.0]))

    assert float(cost) == 6.5
