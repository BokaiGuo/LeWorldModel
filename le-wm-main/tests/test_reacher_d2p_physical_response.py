import pytest
import torch

from experiments.spectral_reachability.audit_reacher_d2p_physical_response import response_metrics


def test_response_error_is_separate_from_same_model_derivative_error():
    analytic = torch.eye(2, dtype=torch.float64)
    row = response_metrics(analytic, analytic, -analytic, torch.ones(2, dtype=torch.float64))
    assert row["model_autograd_vs_model_fd_relative_frobenius_error"] == 0
    assert row["model_vs_physical_fd_relative_frobenius_error"] == 2
    assert row["common_observed_goal_residual_slope_sign_agreements"] == 0
    assert row["model_physical_column_cosines"] == [-1, -1]


def test_matching_responses_preserve_singular_values_and_slopes():
    matrix = torch.tensor([[2., 0.], [0., .5]], dtype=torch.float64)
    row = response_metrics(matrix, matrix, matrix, torch.ones(2, dtype=torch.float64))
    assert row["model_singular_values"] == row["physical_fd_singular_values"] == [2, .5]
    assert row["common_observed_goal_residual_slope_sign_agreements"] == 2


@pytest.mark.parametrize("matrix", [torch.ones(3, 2), torch.ones(2, 3), torch.full((2, 2), float("nan"))])
def test_mismatching_or_nonfinite_response_fails_closed(matrix):
    with pytest.raises(ValueError):
        response_metrics(torch.eye(2), matrix, torch.eye(2), torch.ones(2))
