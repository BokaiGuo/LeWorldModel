"""Gauge-aware finite-horizon transfer objects and certified error bounds.

The module makes the proposed theory executable without claiming a universal
world-model theorem.  The central invariant is

    G_h = D A^(h-1) B,

which is unchanged by an invertible latent similarity transform when the
state, input, and task maps are transformed covariantly.  The bound routine is
an explicit finite-horizon perturbation bound; it records the assumptions it
uses so an empirical artifact cannot silently promote it to a causal claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _finite_nonnegative_array(values: np.ndarray | list[float], name: str, horizon: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != int(horizon):
        raise ValueError(f"{name} must contain exactly horizon={horizon} values")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    return array


def _induced_gain(operator: np.ndarray) -> float:
    operator = np.asarray(operator, dtype=np.float64)
    if operator.ndim != 2:
        raise ValueError("operator must be a matrix")
    return float(np.linalg.norm(operator, 2))


def finite_horizon_transfer_risk_decomposition(
    *,
    A: np.ndarray,
    D: np.ndarray,
    horizon: int,
    closure_residuals: np.ndarray | list[float],
    off_manifold_residuals: np.ndarray | list[float],
    action_effect_errors: np.ndarray | list[float],
    readout_error: float,
    readout_state_bound: float,
    initial_error: float = 0.0,
    spectral_radius: float | None = None,
) -> dict[str, Any]:
    """Decompose a finite-horizon task-error upper bound.

    Consider the lifted true/predicted pair

        x[t+1] = A x[t] + q_cl[t] + q_off[t] + q_act[t],
        xhat[t+1] = Ahat xhat[t],
        y[t] = D x[t],  yhat[t] = Dhat xhat[t].

    Here ``closure_residuals`` are bounds on
    ``||(A-Ahat)xhat[t] + r_cl[t]||``, ``off_manifold_residuals`` bound the
    additional residual on planner-reachable states, and
    ``action_effect_errors`` bound ``||(B-Bhat)u[t]||``.  The function uses
    the exact true finite-horizon gains ``||D A^j||`` and separates the
    excess over the spectral-radius-only gain as ``E_transient``.

    The result is conditional on the supplied residual bounds.  It is not an
    estimator of causal transfer risk and must not be evaluated using the
    held-out task outcome that it is intended to predict.
    """

    A = np.asarray(A, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    h = int(horizon)
    if h < 1 or A.ndim != 2 or A.shape[0] != A.shape[1] or D.ndim != 2 or D.shape[1] != A.shape[0]:
        raise ValueError("invalid A, D, or horizon")
    closure = _finite_nonnegative_array(closure_residuals, "closure_residuals", h)
    off = _finite_nonnegative_array(off_manifold_residuals, "off_manifold_residuals", h)
    action = _finite_nonnegative_array(action_effect_errors, "action_effect_errors", h)
    scalar_values = (readout_error, readout_state_bound, initial_error)
    if any(not np.isfinite(float(value)) or float(value) < 0.0 for value in scalar_values):
        raise ValueError("scalar bounds must be finite and non-negative")

    rho = float(np.max(np.abs(np.linalg.eigvals(A)))) if spectral_radius is None else float(spectral_radius)
    if not np.isfinite(rho) or rho < 0.0:
        raise ValueError("spectral_radius must be finite and non-negative")
    output_gain = _induced_gain(D)
    finite_gains = np.asarray([
        _induced_gain(D @ np.linalg.matrix_power(A, lag))
        for lag in range(h)
    ], dtype=np.float64)
    spectral_gains = output_gain * np.asarray([rho ** lag for lag in range(h)], dtype=np.float64)
    transient_excess = np.maximum(finite_gains - spectral_gains, 0.0)

    # The causal ordering is t=0,...,h-1, while a disturbance at t is
    # propagated through D A^(h-1-t).  Reverse the gain vector accordingly.
    weighted_finite = finite_gains[::-1]
    weighted_spectral = spectral_gains[::-1]
    weighted_excess = transient_excess[::-1]
    residual_total = closure + off + action
    terms = {
        "E_initial": float(_induced_gain(D @ np.linalg.matrix_power(A, h)) * float(initial_error)),
        "E_closure": float(np.dot(weighted_spectral, closure)),
        "E_off": float(np.dot(weighted_spectral, off)),
        "E_action": float(np.dot(weighted_spectral, action)),
        "E_transient": float(np.dot(weighted_excess, residual_total)),
        "E_readout": float(readout_error) * float(readout_state_bound),
    }
    task_bound = float(sum(terms.values()))
    return {
        "status": "conditional_bound",
        "horizon": h,
        "task_error_bound": task_bound,
        "components": terms,
        "finite_horizon_gains": finite_gains.tolist(),
        "spectral_radius_gains": spectral_gains.tolist(),
        "transient_excess_gains": transient_excess.tolist(),
        "inputs": {
            "closure_residuals": closure.tolist(),
            "off_manifold_residuals": off.tolist(),
            "action_effect_errors": action.tolist(),
            "readout_error": float(readout_error),
            "readout_state_bound": float(readout_state_bound),
            "initial_error": float(initial_error),
            "spectral_radius": rho,
        },
        "estimability": {
            "closure": "train_or_calibration_transition_data; in-distribution only",
            "off_manifold": "independent planner-reachable transition data required",
            "action": "independent action-balanced intervention data required",
            "transient": "A and D/Jacobian estimates plus an external metric; spectrum alone is insufficient",
            "readout": "independent task-space labels or decoder calibration required",
            "bound_validity": "conditional on all supplied upper bounds and the local tube assumption",
        },
        "unidentified_from_spectrum_alignment_commutator": [
            "off_manifold_residuals",
            "action_effect_errors",
            "readout_error",
            "transient_excess_for_non-normal_A",
        ],
    }


def transfer_bound_tightness_example() -> dict[str, Any]:
    """Return a scalar positive system for which the decomposition is tight."""

    A = np.asarray([[0.8]], dtype=np.float64)
    D = np.asarray([[1.0]], dtype=np.float64)
    horizon = 4
    closure = np.full(horizon, 0.2, dtype=np.float64)
    zero = np.zeros(horizon, dtype=np.float64)
    report = finite_horizon_transfer_risk_decomposition(
        A=A, D=D, horizon=horizon, closure_residuals=closure,
        off_manifold_residuals=zero, action_effect_errors=zero,
        readout_error=0.0, readout_state_bound=0.0,
        initial_error=0.0,
    )
    # Positive scalar disturbances align with every induced norm, hence the
    # triangle inequality is attained exactly.
    exact = float(sum(0.2 * (0.8 ** lag) for lag in range(horizon)))
    return {"bound": report, "exact_aligned_error": exact, "tight": bool(np.isclose(report["task_error_bound"], exact))}


def spectrum_alignment_commutator_counterexample() -> dict[str, Any]:
    """Construct equal-spectrum/equal-commutator pairs with different transfer."""

    A = np.diag([0.9, 0.4]).astype(np.float64)
    covariance = np.eye(2, dtype=np.float64)
    B_left = np.asarray([[1.0], [0.0]])
    D_left = np.asarray([[1.0, 0.0]])
    B_right = np.asarray([[0.0], [1.0]])
    D_right = np.asarray([[0.0, 1.0]])
    left = finite_horizon_response(A, B_left, D_left, 2)
    right = finite_horizon_response(A, B_right, D_right, 2)
    commutator = covariance @ A - A @ covariance
    return {
        "same_spectrum": True,
        "same_covariance": True,
        "same_covariance_predictor_commutator": bool(np.allclose(commutator, 0.0)),
        "transfer_h2_left": left.tolist(),
        "transfer_h2_right": right.tolist(),
        "different_transfer": bool(not np.allclose(left, right)),
        "interpretation": "spectrum, covariance alignment, and covariance-predictor commutator do not identify B/D task transfer",
    }


def adversarial_matched_transfer_counterexample(
    *, horizon: int = 2, target_effect: float = 0.9,
) -> dict[str, Any]:
    """Construct an explicit matched-diagnostic adversarial family.

    The two realizations deliberately share every scalar diagnostic that a
    checkpoint selector is allowed to use in the proposed matched benchmark:
    covariance, predictor spectrum, covariance--predictor commutator,
    prediction MSE, and action-inverse risk.  Only the covariant input/output
    maps ``B`` and ``D`` differ.  Consequently ``G_h = D A^(h-1) B`` and the
    task loss of the same unit action differ.

    This is a non-identifiability construction, not a learned-model result.
    It prevents a scalar alignment/spectrum certificate from being promoted to
    a transfer certificate without an independently estimated task map.
    """
    h = int(horizon)
    if h < 2:
        raise ValueError("the adversarial construction requires horizon >= 2")
    if not np.isfinite(float(target_effect)):
        raise ValueError("target_effect must be finite")
    A = np.diag([0.9, 0.4]).astype(np.float64)
    covariance = np.eye(2, dtype=np.float64)
    left_B = np.asarray([[1.0], [0.0]])
    left_D = np.asarray([[1.0, 0.0]])
    right_B = np.asarray([[0.0], [1.0]])
    right_D = np.asarray([[0.0, 1.0]])
    left_transfer = finite_horizon_response(A, left_B, left_D, h)
    right_transfer = finite_horizon_response(A, right_B, right_D, h)
    left_effect = float(left_transfer.reshape(-1)[0])
    right_effect = float(right_transfer.reshape(-1)[0])
    commutator = covariance @ A - A @ covariance
    shared = {
        "prediction_mse": 0.01,
        "action_inverse_risk": 0.20,
        "covariance": covariance.tolist(),
        "predictor_spectrum": [0.9, 0.4],
        "covariance_predictor_commutator": commutator.tolist(),
        "action_norm": 1.0,
    }
    return {
        "schema": "adversarial_matched_transfer_counterexample.v1",
        "horizon": h,
        "shared_diagnostics": shared,
        "left": {
            "A": A.tolist(), "B": left_B.tolist(), "D": left_D.tolist(),
            "G_h": left_transfer.tolist(),
            "planning_risk": abs(left_effect - float(target_effect)),
        },
        "right": {
            "A": A.tolist(), "B": right_B.tolist(), "D": right_D.tolist(),
            "G_h": right_transfer.tolist(),
            "planning_risk": abs(right_effect - float(target_effect)),
        },
        "same_spectrum": True,
        "same_covariance": True,
        "same_commutator": bool(np.allclose(commutator, 0.0)),
        "same_prediction_mse": True,
        "same_action_inverse_risk": True,
        "different_transfer": bool(not np.allclose(left_transfer, right_transfer)),
        "different_planning_risk": bool(not np.isclose(abs(left_effect - float(target_effect)), abs(right_effect - float(target_effect)))),
        "claim_boundary": (
            "Synthetic non-identifiability counterexample: matched scalar diagnostics "
            "do not identify finite-horizon task transfer without B/D or equivalent "
            "independent transition evidence."
        ),
    }


def finite_horizon_response(A: np.ndarray, B: np.ndarray, D: np.ndarray, horizon: int) -> np.ndarray:
    """Return ``D A^(h-1) B`` for a positive integer horizon."""

    A, B, D = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64), np.asarray(D, dtype=np.float64)
    h = int(horizon)
    if h < 1 or A.ndim != 2 or A.shape[0] != A.shape[1] or B.ndim != 2 or D.ndim != 2:
        raise ValueError("invalid operator shapes or horizon")
    if B.shape[0] != A.shape[0] or D.shape[1] != A.shape[0]:
        raise ValueError("A, B and D dimensions are incompatible")
    return D @ np.linalg.matrix_power(A, h - 1) @ B


def transfer_function(A: np.ndarray, B: np.ndarray, D: np.ndarray, z: complex) -> np.ndarray:
    """Return ``D (zI-A)^(-1) B`` when the resolvent exists."""

    A = np.asarray(A, dtype=np.complex128)
    identity = np.eye(A.shape[0], dtype=np.complex128)
    return np.asarray(D, dtype=np.complex128) @ np.linalg.solve(complex(z) * identity - A, np.asarray(B, dtype=np.complex128))


def similarity_transform(
    A: np.ndarray, B: np.ndarray, D: np.ndarray, T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform a latent realization by ``z' = T z``."""

    T = np.asarray(T, dtype=np.float64)
    inverse = np.linalg.inv(T)
    return inverse @ np.asarray(A) @ T, inverse @ np.asarray(B), np.asarray(D) @ T


def metric_adjoint(A: np.ndarray, metric: np.ndarray) -> np.ndarray:
    """Return the adjoint under ``<x,y>_M = x^T M y``."""

    metric = np.asarray(metric, dtype=np.float64)
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("metric must be square")
    return np.linalg.solve(metric, np.asarray(A, dtype=np.float64).T @ metric)


def metric_commutator(A: np.ndarray, metric: np.ndarray) -> np.ndarray:
    """Return the metric-covariant commutator ``[A,A^sharp_M]``."""

    adjoint = metric_adjoint(A, metric)
    return np.asarray(A) @ adjoint - adjoint @ np.asarray(A)


def companion_realization(
    kernels: list[np.ndarray], B: np.ndarray, D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a finite-memory companion realization.

    ``kernels=[A_0,...,A_{L-1}]`` represents
    ``x[t+1] = sum_l A_l x[t-l] + B u[t]``.  The state is ordered as
    ``[x[t], x[t-1], ...]``.
    """

    if not kernels:
        raise ValueError("at least one memory kernel is required")
    matrices = [np.asarray(item, dtype=np.float64) for item in kernels]
    dimension = matrices[0].shape[0]
    if any(item.shape != (dimension, dimension) for item in matrices):
        raise ValueError("all memory kernels must have the same square shape")
    action = np.asarray(B, dtype=np.float64)
    output = np.asarray(D, dtype=np.float64)
    if action.shape[0] != dimension or output.shape[1] != dimension:
        raise ValueError("kernel, input, and output dimensions are incompatible")
    length = len(matrices)
    companion = np.zeros((length * dimension, length * dimension), dtype=np.float64)
    companion[:dimension] = np.concatenate(matrices, axis=1)
    if length > 1:
        companion[dimension:, :-dimension] = np.eye((length - 1) * dimension)
    companion_input = np.zeros((length * dimension, action.shape[1]), dtype=np.float64)
    companion_input[:dimension] = action
    companion_output = np.zeros((output.shape[0], length * dimension), dtype=np.float64)
    companion_output[:, :dimension] = output
    return companion, companion_input, companion_output


def finite_horizon_transfer_risk_bound(
    *,
    horizon: int,
    operator_norm: float,
    state_bound: float,
    action_bound: float,
    delta_operator: float,
    delta_input: float,
    delta_output: float,
    initial_error: float = 0.0,
    output_norm: float = 1.0,
) -> dict[str, Any]:
    """Compute a certified recurrence bound under declared boundedness.

    If ``e_t`` is the state error, the supplied assumptions imply

    ``e[t+1] <= ||A|| e[t] + dA X + dB U`` and
    ``task_error <= ||D|| e[h] + dD X``.

    The function does not estimate any term from test outcomes.  A caller must
    supply train/calibration estimates and retain the assumptions in the
    resulting artifact.
    """

    values = [operator_norm, state_bound, action_bound, delta_operator, delta_input, delta_output, initial_error, output_norm]
    if int(horizon) < 1 or any(float(value) < 0.0 or not np.isfinite(float(value)) for value in values):
        raise ValueError("horizon and bound terms must be finite and non-negative")
    error = float(initial_error)
    state_errors = []
    for _ in range(int(horizon)):
        error = float(operator_norm) * error + float(delta_operator) * float(state_bound) + float(delta_input) * float(action_bound)
        state_errors.append(error)
    task_bound = float(output_norm) * error + float(delta_output) * float(state_bound)
    return {
        "status": "bound_available",
        "horizon": int(horizon),
        "state_error_bound": error,
        "task_error_bound": task_bound,
        "state_error_recurrence": state_errors,
        "assumptions": {
            "bounded_state": float(state_bound),
            "bounded_action": float(action_bound),
            "operator_norm": float(operator_norm),
            "initial_error": float(initial_error),
            "norms_are_calibration_side_inputs": True,
        },
        "unidentified_without_calibration": ["delta_operator", "delta_input", "delta_output", "state_bound", "action_bound"],
    }


def same_spectrum_different_transfer_example(angle: float = 0.7) -> dict[str, Any]:
    """Construct a small counterexample to spectrum-only transfer selection."""

    diagonal = np.diag([0.9, 0.4])
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    A_left = diagonal
    A_right = rotation.T @ diagonal @ rotation
    B = np.asarray([[1.0], [0.0]])
    D = np.asarray([[1.0, 0.0]])
    return {
        "same_eigenvalues": bool(np.allclose(np.sort_complex(np.linalg.eigvals(A_left)), np.sort_complex(np.linalg.eigvals(A_right)))),
        "spectrum_left": np.linalg.eigvals(A_left).tolist(),
        "spectrum_right": np.linalg.eigvals(A_right).tolist(),
        "horizon_2_left": finite_horizon_response(A_left, B, D, 2).tolist(),
        "horizon_2_right": finite_horizon_response(A_right, B, D, 2).tolist(),
        "claim_boundary": "A counterexample to spectrum-only transfer; not a learned-model performance result.",
    }
