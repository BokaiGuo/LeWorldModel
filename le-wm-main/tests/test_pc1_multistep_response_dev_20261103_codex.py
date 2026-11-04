import numpy as np
import torch
import json

from experiments.pc1_multistep_response_dev_20261103_codex import (
    MODEL, PHYSICAL, PROTOCOL, direction, metrics, old,
)
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history


def test_direction_and_zero_norm_metrics():
    np.testing.assert_array_equal(direction(5, 1), np.tile([0, 1], (5, 1)))
    assert metrics([0, 0], [1, 0])["cosine"] is None
    assert metrics([0, 0], [1, 0])["relative_to_reference"] is None


def test_normalization_and_temporal_unroll_keep_action_gradient():
    class Core:
        def encode_act(self, actions):
            return actions

        def forward_pred(self, video, actions, prop):
            effect = actions.sum(dim=-1)
            return video, None, prop + effect[..., None, None]

    class Model:
        proprio_mode = "predict_proprio"
        model = Core()

    context = {"visual": torch.zeros(1, 3, 1, 1, 1, 1), "proprio": torch.zeros(1, 3, 1, 1)}
    history = torch.zeros(2, 1, 10)
    actions = torch.zeros(10, 2, requires_grad=True)
    mean, std = torch.tensor([1., -1.]), torch.tensor([2., 4.])
    packets = ((actions - mean) / std).reshape(1, 2, 10).transpose(0, 1)
    output = unroll_with_action_history(Model(), context, history, packets, 3)["proprio"][-1].sum()
    grad = torch.autograd.grad(output, actions)[0]
    torch.testing.assert_close(grad[:, 0], torch.full((10,), .5))
    torch.testing.assert_close(grad[:, 1], torch.full((10,), .25))


def test_saved_development_results_when_available():
    if not MODEL.exists() or not PHYSICAL.exists():
        return
    model = json.loads(MODEL.read_text())
    physical = json.loads(PHYSICAL.read_text())
    assert model["protocol_sha256"] == old.prior.sha(PROTOCOL)
    assert physical["model_sha256"] == old.prior.sha(MODEL)
    assert physical["backend"] == "2.1.2.14"
    assert len(model["rows"]) == 16
    assert len(physical["rows"]) == 32
    assert len(physical["traces"]) == 64
    for row in physical["rows"]:
        prefix = f'{row["root"]}-{row["horizon"]}-{row["axis"]}-{row["epsilon"]}'
        endpoints = [np.asarray(physical["traces"][f"{prefix}-{s}"][-1]["state"][:2]) for s in (-1, 1)]
        np.testing.assert_allclose((endpoints[1] - endpoints[0]) / (2 * row["epsilon"]),
                                   row["physical_derivative"])
