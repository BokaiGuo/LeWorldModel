import copy

import numpy as np
import pytest
import torch

from experiments.spectral_reachability.audit_reacher_d2p_post_prefix_geometry import (
    angular_difference, geometry_metrics, validate_reference,
)


def test_wrapped_difference_crossing_pi():
    assert np.allclose(angular_difference([-np.pi + .01, 0], [np.pi - .01, 0]), [.02, 0])


def test_invalid_angles_rejected():
    with pytest.raises(ValueError):
        angular_difference([np.nan, 0], [0, 0])


def test_zero_response_gives_zero_gain():
    value = geometry_metrics(torch.zeros(2, 20, dtype=torch.float64), torch.tensor([1., 2.], dtype=torch.float64))
    assert value["bounded_quadratic_gain"] == 0
    assert value["singular_values"] == [0, 0]


def test_bounded_quadratic_diagonal_closed_form():
    response = torch.zeros(2, 20, dtype=torch.float64)
    response[0, 0], response[1, 1] = 1, 2
    residual = torch.tensor([2., .5], dtype=torch.float64)
    value = geometry_metrics(response, residual)
    action = torch.tensor([1., 1 / 4.01], dtype=torch.float64)
    objective = (residual - torch.tensor([1., 2.], dtype=torch.float64) * action).square().sum() + .01 * action.square().sum()
    assert value["bounded_quadratic_objective"] == pytest.approx(float(objective), abs=1e-10)
    assert value["projected_gradient_residual"] < 1e-10


def test_response_rotations_preserve_gain():
    generator = torch.Generator().manual_seed(51)
    response = torch.randn(2, 20, generator=generator, dtype=torch.float64) * .1
    residual = torch.tensor([.2, -.3], dtype=torch.float64)
    rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    first = geometry_metrics(response, residual)
    second = geometry_metrics(rotation @ response, rotation @ residual)
    assert first["bounded_quadratic_gain"] == pytest.approx(second["bounded_quadratic_gain"])


def test_wrong_response_shape_rejected():
    with pytest.raises(ValueError):
        geometry_metrics(torch.zeros(2, 10), torch.zeros(2))


def reference():
    actions = np.zeros((15, 2)).tolist()
    row = {"seed": 1, "methods": {name: {"action_witness_raw": actions, "first_action_witness_raw": actions[:5]} for name in ("dispatch", "same_bank_greedy", "zero")}}
    return {"schema": "reacher_d2p_full_bank_refined_dispatch.v1", "status": "completed", "gates": {"checked": True}, "sealed_protocol": {"root_seeds": [1]}, "rows": [row]}


def test_reference_validation_duplicate_and_prefix():
    report = reference()
    validate_reference(report)
    duplicate = copy.deepcopy(report)
    duplicate["rows"].append(copy.deepcopy(duplicate["rows"][0]))
    with pytest.raises(ValueError, match="identity"):
        validate_reference(duplicate)
    report["rows"][0]["methods"]["dispatch"]["first_action_witness_raw"] = np.ones((5, 2)).tolist()
    with pytest.raises(ValueError, match="disagree"):
        validate_reference(report)


def test_failed_reference_rejected():
    report = reference()
    report["gates"]["checked"] = False
    with pytest.raises(ValueError, match="integrity"):
        validate_reference(report)


def test_stored_float32_roundoff_kept_not_clipped():
    report = reference()
    # A single ULP above one is already accepted by the producing runner.
    value = float(np.nextafter(np.float32(1), np.float32(2)))
    report["rows"][0]["methods"]["dispatch"]["action_witness_raw"][7][0] = value
    validate_reference(report)
    assert report["rows"][0]["methods"]["dispatch"]["action_witness_raw"][7][0] == value
    report["rows"][0]["methods"]["dispatch"]["action_witness_raw"][7][0] = 1.0001
    with pytest.raises(ValueError, match="bounded"):
        validate_reference(report)
