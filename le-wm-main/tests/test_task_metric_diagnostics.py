import numpy as np

from experiments.task_metric_diagnostics import (
    metric_geometry,
    metric_orthonormal_operator,
    pusht_task_metric,
    raw_chart_jacobian,
)


def test_raw_jacobian_chain_rule_matches_direct_diagonal_similarity_transform():
    normalized = np.array([[0.8, 0.1], [-0.2, 0.7]])
    std = np.array([2.0, 0.5])
    expected = np.diag(std) @ normalized @ np.diag(1.0 / std)
    assert np.allclose(raw_chart_jacobian(normalized, std), expected)


def test_task_metric_is_external_and_non_degenerate():
    metric = pusht_task_metric()
    assert np.all(np.linalg.eigvalsh(metric) > 0.0)
    assert not np.allclose(metric, np.eye(6))


def test_metric_geometry_is_invariant_under_covariant_coordinate_change():
    rng = np.random.default_rng(4)
    covariance = rng.normal(size=(6, 6))
    covariance = covariance @ covariance.T + np.eye(6)
    predictor = rng.normal(size=(6, 6)) * 0.1
    metric = pusht_task_metric()
    transform = np.eye(6)
    transform[0, 1] = 0.6
    transform[2, 4] = -0.35
    inverse = np.linalg.inv(transform)
    transformed_covariance = transform @ covariance @ transform.T
    transformed_predictor = transform @ predictor @ inverse
    transformed_metric = inverse.T @ metric @ inverse
    left = metric_geometry(covariance, predictor, metric)
    right = metric_geometry(transformed_covariance, transformed_predictor, transformed_metric)
    for key in left:
        assert np.isclose(left[key], right[key], atol=1e-9), key


def test_metric_orthonormal_operator_reduces_to_euclidean_for_identity():
    rng = np.random.default_rng(8)
    operator = rng.normal(size=(4, 4))
    assert np.allclose(metric_orthonormal_operator(operator, np.eye(4)), operator)
