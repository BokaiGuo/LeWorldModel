"""Finite-horizon, task-output decision-spectral mismatch diagnostics.

The module intentionally works in a declared task-output coordinate system,
not in an arbitrary world-model latent coordinate system.  For each context it
fits only the response directions supported by that context's candidate bank;
unsupported action directions are never filled in with a pseudoinverse.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _matrix(value: np.ndarray, *, name: str, columns: int | None = None, minimum_rows: int = 1) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < int(minimum_rows) or (columns is not None and array.shape[1] != columns):
        raise ValueError(f"{name} must be a finite two-dimensional matrix with at least {minimum_rows} rows")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _symmetric_positive_definite(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite square matrix")
    matrix = 0.5 * (matrix + matrix.T)
    if np.min(np.linalg.eigvalsh(matrix)) <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    return matrix


def _symmetric_root(value: np.ndarray) -> np.ndarray:
    eigenvalue, eigenvector = np.linalg.eigh(value)
    return (eigenvector * np.sqrt(eigenvalue)) @ eigenvector.T


def supported_action_basis(
    action_samples: np.ndarray,
    *,
    rank_relative_tolerance: float = 1.0e-10,
    maximum_condition: float = 1.0e8,
) -> dict[str, Any]:
    """Build the empirically identifiable action-support subspace.

    The candidate bank is often lower rank than its flattened action-vector
    ambient dimension.  This is expected, so the fit is performed in the SVD
    support subspace.  A genuinely ill-conditioned *retained* subspace fails
    closed instead of producing a misleading full-dimensional response.
    """

    actions = _matrix(action_samples, name="action_samples", minimum_rows=2)
    if not 0.0 < float(rank_relative_tolerance) < 1.0:
        raise ValueError("rank_relative_tolerance must lie strictly between zero and one")
    if not np.isfinite(maximum_condition) or float(maximum_condition) <= 1.0:
        raise ValueError("maximum_condition must be finite and greater than one")
    mean = actions.mean(axis=0)
    centered = actions - mean
    _left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise ValueError("candidate actions have no varying supported direction")
    rank = int(np.sum(singular_values > singular_values[0] * float(rank_relative_tolerance)))
    if rank < 1:
        raise ValueError("candidate actions have no numerically identifiable support")
    retained = singular_values[:rank]
    condition = float(retained[0] / retained[-1])
    if condition > float(maximum_condition):
        raise ValueError(
            f"candidate action support is ill-conditioned (condition={condition}, maximum={maximum_condition})"
        )
    basis = right[:rank].T
    coordinates = centered @ basis
    covariance = coordinates.T @ coordinates / float(len(actions) - 1)
    covariance = _symmetric_positive_definite(covariance, name="supported_action_covariance")
    return {
        "action_mean": mean,
        "basis": basis,
        "coordinates": coordinates,
        "covariance": covariance,
        "support_rank": rank,
        "ambient_action_dimension": int(actions.shape[1]),
        "sample_count": int(len(actions)),
        "condition_number": condition,
        "singular_values": retained,
    }


def fit_supported_response(
    action_samples: np.ndarray,
    output_samples: np.ndarray,
    *,
    rank_relative_tolerance: float = 1.0e-10,
    maximum_condition: float = 1.0e8,
) -> dict[str, Any]:
    """Fit an affine output response on only candidate-supported directions."""

    actions = _matrix(action_samples, name="action_samples", minimum_rows=2)
    outputs = _matrix(output_samples, name="output_samples", minimum_rows=2)
    if len(actions) != len(outputs):
        raise ValueError("action_samples and output_samples must have the same row count")
    support = supported_action_basis(
        actions, rank_relative_tolerance=rank_relative_tolerance, maximum_condition=maximum_condition,
    )
    centered_output = outputs - outputs.mean(axis=0)
    coefficient, _residual, rank, _singular = np.linalg.lstsq(support["coordinates"], centered_output, rcond=None)
    if int(rank) != int(support["support_rank"]):
        raise ValueError("supported action response fit lost an identifiable action direction")
    response = coefficient.T
    if not np.isfinite(response).all():
        raise ValueError("supported action response fit is non-finite")
    return {
        **support,
        "response": response,
        "output_mean": outputs.mean(axis=0),
        "output_dimension": int(outputs.shape[1]),
    }


def decision_spectral_mismatch(
    real_response: np.ndarray,
    predicted_response: np.ndarray,
    *,
    task_metric: np.ndarray,
    action_covariance: np.ndarray,
) -> dict[str, Any]:
    """Return ``Q^(1/2)(G_pred-G_real)Sigma^(1/2)`` and its SVD."""

    real = _matrix(real_response, name="real_response")
    predicted = _matrix(predicted_response, name="predicted_response", columns=real.shape[1])
    if predicted.shape != real.shape:
        raise ValueError("real_response and predicted_response must have identical shapes")
    metric = _symmetric_positive_definite(task_metric, name="task_metric")
    covariance = _symmetric_positive_definite(action_covariance, name="action_covariance")
    if metric.shape[0] != real.shape[0] or covariance.shape[0] != real.shape[1]:
        raise ValueError("task metric or action covariance does not match response shape")
    operator = _symmetric_root(metric) @ (predicted - real) @ _symmetric_root(covariance)
    left, singular_values, right_transpose = np.linalg.svd(operator, full_matrices=False)
    return {
        "operator": operator,
        "left_singular_vectors": left,
        "singular_values": singular_values,
        "right_singular_vectors": right_transpose.T,
        "frobenius_norm": float(np.linalg.norm(operator, ord="fro")),
        "spectral_norm": float(singular_values[0]) if singular_values.size else 0.0,
    }


def candidate_pair_leverage(
    mismatch: dict[str, Any],
    *,
    action_delta: np.ndarray,
    task_gradient: np.ndarray,
    action_basis: np.ndarray,
    action_covariance: np.ndarray,
    task_metric: np.ndarray,
) -> dict[str, Any]:
    """Decompose a candidate-pair's first-order task-cost mismatch.

    The returned modal absolute contributions are diagnostic rather than an
    exact regret decomposition: squared costs can have a material second-order
    term near a goal.
    """

    basis = _matrix(action_basis, name="action_basis")
    covariance = _symmetric_positive_definite(action_covariance, name="action_covariance")
    metric = _symmetric_positive_definite(task_metric, name="task_metric")
    delta = np.asarray(action_delta, dtype=np.float64).reshape(-1)
    gradient = np.asarray(task_gradient, dtype=np.float64).reshape(-1)
    if delta.shape[0] != basis.shape[0] or gradient.shape[0] != metric.shape[0]:
        raise ValueError("candidate pair or task gradient dimension is incompatible with the declared metric")
    if covariance.shape[0] != basis.shape[1]:
        raise ValueError("action covariance does not match action basis")
    if not np.isfinite(delta).all() or not np.isfinite(gradient).all():
        raise ValueError("candidate pair and task gradient must be finite")
    operator = _matrix(np.asarray(mismatch["operator"]), name="mismatch_operator")
    left = _matrix(np.asarray(mismatch["left_singular_vectors"]), name="left_singular_vectors")
    right = _matrix(np.asarray(mismatch["right_singular_vectors"]), name="right_singular_vectors")
    singular_values = np.asarray(mismatch["singular_values"], dtype=np.float64).reshape(-1)
    rank = len(singular_values)
    if (
        operator.shape != (metric.shape[0], basis.shape[1])
        or left.shape != (metric.shape[0], rank)
        or right.shape != (basis.shape[1], rank)
    ):
        raise ValueError("mismatch SVD dimensions are inconsistent")
    output_weight = np.linalg.solve(_symmetric_root(metric), gradient)
    action_coordinate = basis.T @ delta
    action_weight = np.linalg.solve(_symmetric_root(covariance), action_coordinate)
    signed = singular_values * (left.T @ output_weight) * (right.T @ action_weight)
    return {
        "modal_absolute_contributions": np.abs(signed),
        "modal_signed_contributions": signed,
        "first_order_cost_mismatch": float(output_weight @ operator @ action_weight),
        "absolute_modal_sum": float(np.abs(signed).sum()),
        "claim_boundary": "first-order local candidate-pair diagnostic; it omits nonlinear and second-order task-cost terms",
    }
