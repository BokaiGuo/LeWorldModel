from __future__ import annotations

import numpy as np
import pytest

from experiments.decision_benchmark.decision_spectral_mismatch import (
    candidate_pair_leverage,
    decision_spectral_mismatch,
    fit_supported_response,
)


def test_supported_response_recovers_known_response_on_candidate_support() -> None:
    rng = np.random.default_rng(12)
    actions = rng.normal(size=(24, 3))
    response = np.asarray([[1.0, -2.0, 0.5], [-0.25, 0.75, 1.5]])
    outputs = actions @ response.T + np.asarray([2.0, -1.0])
    fitted = fit_supported_response(actions, outputs)
    assert fitted["support_rank"] == 3
    assert np.allclose(fitted["response"] @ fitted["basis"].T, response, atol=1.0e-10)


def test_mismatch_is_invariant_to_invertible_task_output_reparameterization() -> None:
    real = np.asarray([[1.0, -0.5], [0.2, 0.8]])
    predicted = np.asarray([[0.8, -0.2], [0.4, 0.5]])
    metric = np.asarray([[2.0, 0.3], [0.3, 1.0]])
    covariance = np.asarray([[1.3, 0.1], [0.1, 0.7]])
    original = decision_spectral_mismatch(real, predicted, task_metric=metric, action_covariance=covariance)
    transform = np.asarray([[1.2, -0.4], [0.3, 0.9]])
    transformed = decision_spectral_mismatch(
        transform @ real,
        transform @ predicted,
        task_metric=np.linalg.inv(transform).T @ metric @ np.linalg.inv(transform),
        action_covariance=covariance,
    )
    assert np.allclose(original["singular_values"], transformed["singular_values"], atol=1.0e-10)
    assert original["frobenius_norm"] == pytest.approx(transformed["frobenius_norm"], abs=1.0e-10)


def test_pair_leverage_reconstructs_the_declared_first_order_mismatch() -> None:
    real = np.zeros((1, 1))
    predicted = np.asarray([[2.0]])
    metric = np.asarray([[4.0]])
    covariance = np.asarray([[9.0]])
    mismatch = decision_spectral_mismatch(real, predicted, task_metric=metric, action_covariance=covariance)
    leverage = candidate_pair_leverage(
        mismatch,
        action_delta=np.asarray([3.0]), task_gradient=np.asarray([5.0]),
        action_basis=np.asarray([[1.0]]), action_covariance=covariance, task_metric=metric,
    )
    assert leverage["first_order_cost_mismatch"] == pytest.approx(30.0)
    assert leverage["modal_absolute_contributions"][0] == pytest.approx(30.0)


def test_pair_leverage_supports_more_action_than_task_output_directions() -> None:
    mismatch = decision_spectral_mismatch(
        np.zeros((2, 3)), np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        task_metric=np.eye(2), action_covariance=np.eye(3),
    )
    leverage = candidate_pair_leverage(
        mismatch,
        action_delta=np.asarray([1.0, 2.0, 3.0]), task_gradient=np.asarray([2.0, 1.0]),
        action_basis=np.eye(3), action_covariance=np.eye(3), task_metric=np.eye(2),
    )
    assert leverage["first_order_cost_mismatch"] == pytest.approx(6.0)
    assert len(leverage["modal_signed_contributions"]) == 2


def test_rank_deficient_candidate_bank_is_fit_only_on_its_supported_subspace() -> None:
    coordinate = np.linspace(-1.0, 1.0, 8)
    actions = np.stack((coordinate, 2.0 * coordinate, np.zeros_like(coordinate)), axis=1)
    outputs = np.stack((3.0 * coordinate, -coordinate), axis=1)
    fitted = fit_supported_response(actions, outputs)
    assert fitted["support_rank"] == 1
    assert fitted["response"].shape == (2, 1)


def test_response_fit_rejects_candidate_bank_without_action_variation() -> None:
    with pytest.raises(ValueError, match="no varying supported direction"):
        fit_supported_response(np.zeros((4, 2)), np.ones((4, 1)))
