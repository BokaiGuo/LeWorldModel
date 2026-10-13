"""Metric-aware geometry for the fixed PushT observable pose chart.

The learned model is trained in a normalized six-dimensional chart
``[agent_xy, block_xy, sin(theta), cos(theta)]``.  Planning, however, has a
declared task loss in the *raw* chart.  This module keeps that distinction
explicit: the task metric is external to latent covariance and is therefore
not the degenerate covariance-only whitening construction.

All operators are represented as maps on column vectors.  The metric
orthonormal representation is

    A_M = M^(1/2) A M^(-1/2),

so ordinary Euclidean diagnostics applied to ``A_M`` are invariant when the
metric is transformed covariantly with a change of coordinates.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


EPS = 1e-12


def pusht_task_metric(
    *, agent_weight: float = 0.25,
    block_position_weight: float = 1.0,
    orientation_weight: float = 4.0,
) -> np.ndarray:
    """Return the pre-declared external metric for the six-dimensional chart.

    The block position and orientation terms follow the planning loss scale;
    the agent term is retained with a smaller weight because it is an
    approach/contact variable rather than the terminal task objective.
    """
    weights = np.asarray(
        [agent_weight, agent_weight, block_position_weight, block_position_weight,
         orientation_weight, orientation_weight],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("all task-metric weights must be finite and positive")
    return np.diag(weights)


def raw_chart_jacobian(normalized_jacobian: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Convert a Jacobian in normalized coordinates to the raw pose chart."""
    jacobian = np.asarray(normalized_jacobian, dtype=np.float64)
    scale = np.asarray(std, dtype=np.float64).reshape(-1)
    if jacobian.ndim != 2 or jacobian.shape[0] != jacobian.shape[1]:
        raise ValueError("normalized_jacobian must be square")
    if scale.shape != (jacobian.shape[0],) or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("std must be a positive finite vector matching the Jacobian")
    return scale[:, None] * jacobian / scale[None, :]


def _metric_sqrt(metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    metric = np.asarray(metric, dtype=np.float64)
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("metric must be square")
    if not np.allclose(metric, metric.T, atol=1e-10):
        raise ValueError("metric must be symmetric")
    values, vectors = np.linalg.eigh(metric)
    if np.any(values <= 0.0) or np.any(~np.isfinite(values)):
        raise ValueError("metric must be positive definite")
    root = (vectors * np.sqrt(values)[None, :]) @ vectors.T
    inverse_root = (vectors * (1.0 / np.sqrt(values))[None, :]) @ vectors.T
    return root, inverse_root


def metric_orthonormal_operator(operator: np.ndarray, metric: np.ndarray) -> np.ndarray:
    """Represent ``operator`` in an orthonormal frame induced by ``metric``."""
    root, inverse_root = _metric_sqrt(metric)
    operator = np.asarray(operator, dtype=np.float64)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1] or operator.shape != metric.shape:
        raise ValueError("operator and metric must have the same square shape")
    return root @ operator @ inverse_root


def metric_geometry(
    covariance: np.ndarray,
    predictor: np.ndarray,
    metric: np.ndarray,
    *,
    rank: int = 3,
    horizons: Iterable[int] = range(1, 9),
) -> dict[str, float]:
    """Compute task-metric alignment and finite-horizon non-normal diagnostics."""
    covariance = np.asarray(covariance, dtype=np.float64)
    predictor = np.asarray(predictor, dtype=np.float64)
    if covariance.shape != predictor.shape or covariance.ndim != 2:
        raise ValueError("covariance and predictor must have matching square shapes")
    root, inverse_root = _metric_sqrt(metric)
    covariance_m = root @ covariance @ root
    predictor_m = root @ predictor @ inverse_root
    covariance_m = 0.5 * (covariance_m + covariance_m.T)
    # Use the complex Schur form here.  A real Schur form contains 2x2 blocks
    # for complex-conjugate eigenpairs; the apparent strictly-upper energy can
    # then change when an orthogonal similarity permutes those blocks.  The
    # complex Schur strictly-upper norm is instead the unitary-invariant
    # departure from normality ``sqrt(||A||_F^2 - sum |lambda|^2)``.
    triangular, _ = __import__("scipy.linalg", fromlist=["schur"]).schur(
        predictor_m, output="complex",
    )
    values, covariance_vectors = np.linalg.eigh(covariance_m)
    order = np.argsort(values)[::-1]
    rank = min(max(int(rank), 1), covariance.shape[0])
    covariance_vectors = covariance_vectors[:, order[:rank]]
    predictor_vectors = np.linalg.svd(predictor_m, full_matrices=False)[0][:, :rank]
    cosines = np.linalg.svd(covariance_vectors.T @ predictor_vectors, compute_uv=False)
    cosines = np.clip(np.asarray(cosines, dtype=np.float64), 0.0, 1.0)
    principal_angles = np.arccos(cosines)
    alignment = float(np.sqrt(max(0.0, 1.0 - float(np.mean(np.square(cosines))))))
    commutator = covariance_m @ predictor_m - predictor_m @ covariance_m
    denominator = max(float(np.linalg.norm(covariance_m, "fro") * np.linalg.norm(predictor_m, "fro")), EPS)
    normality_denominator = max(float(np.linalg.norm(predictor_m, "fro") ** 2), EPS)
    gains = [
        float(np.linalg.svd(np.linalg.matrix_power(predictor_m, int(h)), compute_uv=False)[0])
        for h in horizons
    ]
    if not gains:
        raise ValueError("at least one horizon is required")
    identity = np.eye(predictor_m.shape[0], dtype=np.complex128)
    resolvent = max(
        float(np.linalg.norm(np.linalg.pinv(np.exp(1j * theta) * identity - predictor_m), 2))
        for theta in np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    )
    return {
        "task_metric_alignment": alignment,
        "principal_angle": float(np.mean(principal_angles)),
        "projector_overlap": float(np.mean(np.square(cosines))),
        "task_metric_commutator": float(np.linalg.norm(commutator, "fro") / denominator),
        "task_metric_schur_offdiagonal_energy": float(
            np.linalg.norm(np.triu(triangular, k=1), "fro")
            / max(float(np.linalg.norm(triangular, "fro")), EPS)
        ),
        "task_metric_departure_from_normality": float(
            np.linalg.norm(predictor_m.T @ predictor_m - predictor_m @ predictor_m.T, "fro")
            / normality_denominator
        ),
        "task_metric_transient_amplification": max(gains),
        "resolvent_norm": resolvent,
        "task_metric_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(predictor_m)))),
    }


def task_metric_metadata(metric: np.ndarray) -> dict[str, object]:
    metric = np.asarray(metric, dtype=np.float64)
    return {
        "name": "pusht_task_space_v1",
        "chart": "decoded_observable_pose_raw=[agent_x,agent_y,block_x,block_y,sin_theta,cos_theta]",
        "construction": "external fixed task-space weights; not latent covariance whitening",
        "matrix": metric.tolist(),
        "positive_definite": bool(np.all(np.linalg.eigvalsh(metric) > 0.0)),
    }
