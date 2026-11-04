import pytest
import torch

from experiments.spectral_reachability.run_reacher_pc_selected_direction_fd import probe_direction


def test_direction_probe_linear_and_nonlinear():
    x = torch.zeros(2, dtype=torch.float64)
    selected = torch.tensor([0.5, 0.0], dtype=torch.float64)
    linear = probe_direction(lambda v: torch.stack((2 * v[0], v[1])), x, selected)
    assert linear["jvp_norm"] == pytest.approx(2)
    assert all(r["centered_fd_relative_error"] < 1e-12 for r in linear["samples"])
    curved = probe_direction(lambda v: torch.stack((2 * v[0] + v[0] ** 2, v[1])), x, selected)
    assert curved["samples"][-1]["forward_linear_relative_error"] > 0.1


def test_direction_probe_rejects_zero_vector():
    with pytest.raises(ValueError, match="invalid selected"):
        probe_direction(lambda v: v, torch.zeros(2), torch.zeros(2))
