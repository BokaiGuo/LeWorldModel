import pytest
import torch

from experiments.pc_tworoom_search_region_latent_jvp_20261107_codex import compare


def test_vector_jvp_fd_and_adjoint():
    raw = torch.tensor([0.2, -0.3], dtype=torch.float64)
    direction = torch.tensor([0.6, 0.8], dtype=torch.float64)
    projection = torch.tensor([0.8, -0.6], dtype=torch.float64)
    result = compare(lambda x: torch.stack((x[0] * x[1], x[0].square())), raw, direction,
                     projection, (0.001,))
    assert result["jvp_norm"] == pytest.approx(((-0.02)**2 + 0.24**2)**0.5)
    assert result["finite_difference"]["0.001"]["l2_error"] < 1e-10
    assert result["adjoint_error"] < 1e-10


def test_physical_bounds_enforced():
    raw = torch.tensor([1.0, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="illegal perturbation"):
        compare(lambda x: x.square(), raw, torch.ones_like(raw), torch.ones_like(raw), (0.01,))
