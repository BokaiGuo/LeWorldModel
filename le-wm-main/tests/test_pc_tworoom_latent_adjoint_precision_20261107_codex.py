import pytest
import torch

from experiments.pc_tworoom_latent_adjoint_precision_20261107_codex import dtype_aware_embedder, projected


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 1e-4), (torch.float64, 1e-9)])
def test_projected_jvp_vjp_and_fd(dtype, tolerance):
    x = torch.tensor([0.2, -0.3], dtype=dtype)
    v = torch.tensor([0.6, 0.8], dtype=dtype)
    u = torch.tensor([0.8, -0.6], dtype=dtype)
    result = projected(lambda z: torch.stack((z[0] * z[1], z[0] ** 2)), x, v, u, [0.01])
    assert result["jvp"] == pytest.approx(-0.16, abs=tolerance)
    assert result["vjp"] == pytest.approx(result["jvp"], abs=tolerance)
    assert result["projected_fd"]["0.01"] == pytest.approx(result["jvp"], abs=tolerance)


def test_action_encoder_adapter_respects_dtype_and_gradients():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = torch.nn.Conv1d(10, 10, 1)
            self.embed = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.SiLU())
    module = Encoder().double()
    x = torch.randn(1, 2, 10, dtype=torch.float64, requires_grad=True)
    out = dtype_aware_embedder(module, x)
    grad, = torch.autograd.grad(out.sum(), x)
    assert out.dtype == torch.float64 and grad.shape == x.shape and torch.isfinite(grad).all()
