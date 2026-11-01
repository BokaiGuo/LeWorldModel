import pytest
import torch
from experiments.spectral_reachability.p1_normalized_exchange_q_20261101_main import normalized_update


def test_loss_scale_invariant_action_step():
    raw = torch.zeros((1, 3, 5, 2))
    gradient = torch.arange(30.).reshape_as(raw)
    a = normalized_update(raw, gradient, .05)
    b = normalized_update(raw, gradient * .001, .05)
    torch.testing.assert_close(a, b)
    assert abs(float(a.abs().max()) - .05) < 1e-7


def test_zero_gradient_and_protected_prefix():
    raw = torch.full((1, 3, 5, 2), .5)
    assert torch.equal(normalized_update(raw, torch.zeros_like(raw), .05), raw)
    gradient = torch.ones_like(raw)
    gradient[:, 0] = 0
    result = normalized_update(raw, gradient, .05)
    assert torch.equal(result[:, 0], raw[:, 0])
    assert bool((result.abs() <= 1).all())
    with pytest.raises(ValueError):
        normalized_update(raw, gradient * float("nan"), .05)
