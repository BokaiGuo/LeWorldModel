import pytest
import torch

from experiments.pc_response_temporal_v3_reacher_adapter_preflight_v2_20261104_codex import inverse_bounded


def test_inverse_maps_normalized_actions_back_to_reference_params():
    low, high = torch.tensor([-2., -1.]), torch.tensor([1., 3.])
    actions = torch.tensor([[[0., 1.], [-1., 2.]]])
    params = inverse_bounded(actions, low, high)
    torch.testing.assert_close((low + high) / 2 + (high - low) / 2 * torch.tanh(params), actions)


def test_inverse_refuses_boundary_instead_of_clipping():
    with pytest.raises(ValueError):
        inverse_bounded(torch.tensor([[[1., 0.]]]), torch.tensor([-1., -1.]), torch.tensor([1., 1.]))
