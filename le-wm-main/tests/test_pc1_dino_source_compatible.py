import numpy as np
import pytest
import torch
from transformers import Dinov2Config, Dinov2Model

from experiments.pc1_layer_audit.dino_source_compatible import install_source_interpolation


def small_model():
    torch.manual_seed(71)
    return Dinov2Model(Dinov2Config(
        image_size=28, patch_size=2, hidden_size=12, num_hidden_layers=2,
        num_attention_heads=3, intermediate_size=24, hidden_dropout_prob=0.,
    )).eval().requires_grad_(False)


def reference(position, shape, patch_size):
    h, w = (n // patch_size for n in shape)
    side = int(np.sqrt(position.shape[1] - 1))
    if (h, w) == (side, side):
        return position
    grid = position[:, 1:].float().reshape(1, side, side, -1).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(
        grid, scale_factor=((h+.1)/side, (w+.1)/side), mode="bicubic",
        align_corners=False, antialias=False,
    )
    return torch.cat((position[:, :1], grid.permute(0, 2, 3, 1).flatten(1, 2)), dim=1)


@pytest.mark.parametrize("shape", [(14, 14), (16, 22), (28, 28)])
def test_source_positions_and_all_parameter_identity(shape):
    model = small_model()
    before = {k: (p, p.clone()) for k, p in model.named_parameters()}
    install_source_interpolation(model)
    tokens = torch.zeros(1, (shape[0]//2)*(shape[1]//2)+1, 12)
    actual = model.embeddings.interpolate_pos_encoding(tokens, *shape)
    torch.testing.assert_close(actual, reference(model.embeddings.position_embeddings, shape, 2),
                               atol=0, rtol=0)
    for key, p in model.named_parameters():
        assert p is before[key][0]
        torch.testing.assert_close(p, before[key][1], atol=0, rtol=0)
    assert not any(m.training for m in model.modules())
    assert not any(p.requires_grad for p in model.parameters())


def test_interpolation_changes_only_resized_path():
    model = small_model()
    pixels = torch.rand(1, 3, 28, 28)
    small = pixels[:, :, :14, :14]
    with torch.no_grad():
        original = model(pixels).last_hidden_state
        old_small = model(small).last_hidden_state
        install_source_interpolation(model)
        torch.testing.assert_close(model(pixels).last_hidden_state, original, atol=0, rtol=0)
        assert not torch.equal(model(small).last_hidden_state, old_small)
    with pytest.raises(ValueError, match="already installed"):
        install_source_interpolation(model)


def test_input_gradient_still_available_with_frozen_parameters():
    model = install_source_interpolation(small_model())
    pixels = torch.rand(1, 3, 14, 14, requires_grad=True)
    cost = model(pixels).last_hidden_state[..., 0].square().sum()
    grad, = torch.autograd.grad(cost, pixels)
    assert torch.isfinite(grad).all() and grad.abs().max() > 0
    assert all(p.grad is None for p in model.parameters())
