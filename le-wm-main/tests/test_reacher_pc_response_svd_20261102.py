import pytest
import torch

from experiments.spectral_reachability import reacher_pc_response_svd_20261102 as subject


def evaluation(x):
    immediate = x[0] + 1
    values = torch.stack([
        torch.stack((x[0], x[1] * factor + x[2] * factor * factor))
        for factor in (0., 1., -1., 2., -2.)
    ]).double()
    return immediate, values.square().mean(dim=1), values


def test_svd_bank_bounds_dtype_and_original_selector():
    origin = torch.zeros(3)
    bank, result = subject.generate(origin, evaluation)
    assert bank.dtype == origin.dtype
    assert bank.shape == (13, 3)
    assert torch.all((bank - origin).norm(dim=1) <= 0.200001)
    assert result["response_jacobian_shape"] == [8, 3]
    assert result["finite_difference_relative_error"] < 1e-4
    assert result["strict_selected_index"] is None
    assert not any(row["eligible"] for row in result["proposal_rows"])
    # All proposals intentionally increase immediate cost in this fixture.
    assert torch.all(bank[1:, 0] > 0)


def test_common_coordinate_shift_does_not_change_response_spectrum():
    def shifted(x):
        immediate, costs, coordinates = evaluation(x)
        return immediate, costs, coordinates + x[1].double() * 123

    _, base = subject.generate(torch.zeros(3), evaluation)
    _, changed = subject.generate(torch.zeros(3), shifted)
    assert changed["singular_values"] == pytest.approx(base["singular_values"], abs=1e-5)


def test_flat_tangent_response_rejected():
    def flat(x):
        coords = torch.stack([x.double(), x.double(), x.double(), x.double(), x.double()])
        return x[0], coords.square().mean(dim=1), coords

    with pytest.raises(ValueError, match="no nonzero tangent response"):
        subject.generate(torch.zeros(3), flat)


def test_invalid_coordinates_rejected():
    def invalid(x):
        return x[0], torch.zeros(5), torch.zeros(4, 2)

    with pytest.raises(ValueError, match="invalid terminal coordinates"):
        subject.generate(torch.zeros(3), invalid)
