"""Covariant finite-horizon transfer-risk control.

This module turns the covariance--predictor analysis into a control interface.
The controller is deliberately certificate-gated: a task metric, a finite
horizon Markov map ``G_h = D A^(h-1) B``, independently estimated error
envelopes, action identifiability, and real/predicted transition consistency
are all required before a learned action may be released.  Missing evidence
causes a deterministic fallback rather than an optimistic score.

The result is a metric-specified robust tube controller, not a claim that a
commutator is a universal transfer certificate.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

try:  # package import for tests; direct sibling import for the audit runner
    from .finite_horizon_transfer import finite_horizon_response, finite_horizon_transfer_risk_decomposition
except ImportError:  # pragma: no cover - exercised by direct CLI execution
    from finite_horizon_transfer import finite_horizon_response, finite_horizon_transfer_risk_decomposition


def _metric_root(metric: np.ndarray) -> np.ndarray:
    metric = np.asarray(metric, dtype=np.float64)
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("metric must be square")
    values, vectors = np.linalg.eigh(0.5 * (metric + metric.T))
    if np.any(values <= 0.0) or not np.isfinite(values).all():
        raise ValueError("metric must be positive definite")
    return (vectors * np.sqrt(values)[None, :]) @ vectors.T


def markov_parameters(A: np.ndarray, B: np.ndarray, D: np.ndarray, horizon: int) -> list[np.ndarray]:
    """Return the finite-horizon input-output maps ``G_l`` for ``l=1..h``."""

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    h = int(horizon)
    if h < 1 or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix and horizon must be positive")
    if B.ndim != 2 or D.ndim != 2 or B.shape[0] != A.shape[0] or D.shape[1] != A.shape[0]:
        raise ValueError("A, B, and D dimensions are incompatible")
    return [finite_horizon_response(A, B, D, lag) for lag in range(1, h + 1)]


def nominal_task_trajectory(
    A: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    x0: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Propagate a candidate action sequence through the declared linear model."""

    A, B, D = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64), np.asarray(D, dtype=np.float64)
    state = np.asarray(x0, dtype=np.float64).reshape(-1)
    sequence = np.asarray(actions, dtype=np.float64)
    if sequence.ndim != 2 or state.shape != (A.shape[0],) or sequence.shape[1] != B.shape[1]:
        raise ValueError("invalid state or action sequence shape")
    if not np.isfinite(sequence).all() or not np.isfinite(state).all():
        raise ValueError("state and actions must be finite")
    outputs = []
    for action in sequence:
        state = A @ state + B @ action
        outputs.append(D @ state)
    return np.asarray(outputs, dtype=np.float64)


def certified_metric_tube(
    *,
    A: np.ndarray,
    B: np.ndarray,
    metric: np.ndarray,
    action_bound: float,
    disturbance_bound: float,
    local_tube_radius: float,
) -> dict[str, Any]:
    """Compute a local robust positively invariant tube certificate.

    In the metric coordinates ``v=M^(1/2)x``, let ``alpha=||A||_M`` and
    ``beta=||B||_M``.  If ``alpha<1`` and every action/disturbance obeys the
    declared bounds, ``r=(beta*u+epsilon)/(1-alpha)`` is invariant.  The
    certificate is rejected when the bound leaves the local nonlinear tube.
    """

    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    metric = np.asarray(metric, dtype=np.float64)
    u = float(action_bound)
    epsilon = float(disturbance_bound)
    tube = float(local_tube_radius)
    if any(not np.isfinite(value) or value < 0.0 for value in (u, epsilon, tube)):
        raise ValueError("bounds must be finite and non-negative")
    root = _metric_root(metric)
    inverse = np.linalg.inv(root)
    A_metric = root @ A @ inverse
    B_metric = root @ B
    alpha = float(np.linalg.svd(A_metric, compute_uv=False)[0])
    beta = float(np.linalg.svd(B_metric, compute_uv=False)[0])
    if alpha >= 1.0:
        radius = float("inf")
        status = "rejected_unstable_local_operator"
    else:
        radius = float((beta * u + epsilon) / max(1.0 - alpha, 1e-12))
        status = "certified" if radius <= tube else "rejected_outside_local_tube"
    return {
        "schema": "covariant_metric_tube.v1",
        "status": status,
        "metric_alpha": alpha,
        "metric_beta": beta,
        "invariant_radius": radius,
        "declared_local_tube_radius": tube,
        "action_bound": u,
        "disturbance_bound": epsilon,
        "assumptions": [
            "positive-definite declared task metric",
            "local Jacobian/operator remains in the declared tube",
            "per-step disturbance is bounded by disturbance_bound",
            "action norm is bounded by action_bound",
        ],
        "claim_boundary": "local robust-tube certificate; not a global nonlinear or universal-world-model theorem",
    }


def _error_defaults(horizon: int, errors: dict[str, Any] | None) -> dict[str, Any] | None:
    if errors is None:
        return None
    required = ("closure_residuals", "off_manifold_residuals", "action_effect_errors", "readout_error", "readout_state_bound")
    if any(key not in errors for key in required):
        return None
    return errors


def covariant_transfer_risk_mpc(
    *,
    A: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    metric: np.ndarray,
    x0: np.ndarray,
    goal_task: np.ndarray,
    candidate_actions: Iterable[np.ndarray],
    risk_errors: Iterable[dict[str, Any] | None] | None,
    risk_threshold: float,
    risk_weight: float,
    action_identifiability: dict[str, Any],
    transition_consistency: dict[str, Any],
    tube_certificate: dict[str, Any] | None,
    fallback_index: int = 0,
) -> dict[str, Any]:
    """Select a short action sequence under a declared transfer-risk gate.

    The nominal objective is terminal task cost plus ``risk_weight`` times the
    conditional finite-horizon bound.  If either independent data gate fails,
    or a candidate lacks all required error components, the learned selector
    is not permitted to act and the declared fallback candidate is returned.
    """

    A, B, D, metric = map(lambda value: np.asarray(value, dtype=np.float64), (A, B, D, metric))
    candidates = [np.asarray(value, dtype=np.float64) for value in candidate_actions]
    if not candidates:
        raise ValueError("at least one candidate action sequence is required")
    h = candidates[0].shape[0]
    if any(value.ndim != 2 or value.shape != candidates[0].shape for value in candidates):
        raise ValueError("all candidate action sequences must have the same 2D shape")
    if h < 1 or candidates[0].shape[1] != B.shape[1]:
        raise ValueError("candidate action dimensions do not match B")
    if not np.isfinite(float(risk_threshold)) or float(risk_threshold) < 0.0:
        raise ValueError("risk_threshold must be finite and non-negative")
    if not np.isfinite(float(risk_weight)) or float(risk_weight) < 0.0:
        raise ValueError("risk_weight must be finite and non-negative")
    gates = {
        "action_identifiability": bool(action_identifiability.get("passed", False)),
        "transition_consistency": bool(transition_consistency.get("passed", False)),
        "local_metric_tube": bool(tube_certificate is not None and tube_certificate.get("status") == "certified"),
    }
    errors = list(risk_errors) if risk_errors is not None else [None] * len(candidates)
    if len(errors) != len(candidates):
        raise ValueError("risk_errors must have one entry per candidate")
    markov = markov_parameters(A, B, D, h)
    root = _metric_root(metric)
    goal = np.asarray(goal_task, dtype=np.float64).reshape(-1)
    if goal.shape != (D.shape[0],):
        raise ValueError("goal_task must match task output dimension")

    reports: list[dict[str, Any]] = []
    for index, (actions, error_values) in enumerate(zip(candidates, errors)):
        trajectory = nominal_task_trajectory(A, B, D, x0, actions)
        terminal = trajectory[-1]
        task_delta = root @ (terminal - goal)
        task_cost = float(task_delta @ task_delta)
        normalized = _error_defaults(h, error_values)
        risk = None
        if normalized is not None:
            risk = finite_horizon_transfer_risk_decomposition(
                A=A,
                D=D,
                horizon=h,
                closure_residuals=normalized["closure_residuals"],
                off_manifold_residuals=normalized["off_manifold_residuals"],
                action_effect_errors=normalized["action_effect_errors"],
                readout_error=float(normalized["readout_error"]),
                readout_state_bound=float(normalized["readout_state_bound"]),
            )
        reports.append({
            "index": index,
            "task_cost": task_cost,
            "risk_bound": None if risk is None else risk["task_error_bound"],
            "objective": None if risk is None else task_cost + float(risk_weight) * risk["task_error_bound"],
            "risk_status": None if risk is None else risk["status"],
            "terminal_task": terminal.tolist(),
        })

    gate_passed = all(gates.values())
    eligible = [item for item in reports if item["risk_bound"] is not None and item["risk_bound"] <= float(risk_threshold)]
    if gate_passed and eligible:
        selected = min(eligible, key=lambda item: (float(item["objective"]), int(item["index"])))
        status = "permitted"
        reason = "all independent gates passed and candidate risk is below threshold"
    else:
        selected_index = int(fallback_index)
        if selected_index < 0 or selected_index >= len(candidates):
            raise ValueError("fallback_index is out of range")
        selected = reports[selected_index]
        status = "fallback_fail_closed"
        reason = "independent action/transition evidence or finite risk envelope is missing"
    return {
        "schema": "covariant_transfer_risk_mpc.v1",
        "status": status,
        "reason": reason,
        "selected_index": int(selected["index"]),
        "gates": gates,
        "risk_threshold": float(risk_threshold),
        "risk_weight": float(risk_weight),
        "markov_parameters": [value.tolist() for value in markov],
        "candidate_reports": reports,
        "selected": selected,
        "claim_boundary": (
            "The controller is certified only for the declared metric, local model, "
            "finite horizon, and independently estimated residual envelopes. "
            "A commutator alone is not a certificate."
        ),
    }
