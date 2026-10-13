"""Auditable action-lifted system diagnostics.

This module turns a lifted latent sequence into a controlled local system

    eta[t+1] = A_phi eta[t] + B_phi a[t] + eps[t]
    y[t]     = D_phi eta[t] + nu[t]

and reports quantities that are invisible to predictor eigenvalues alone:
persistent excitation, finite-horizon observability/controllability Gramians,
PBH ranks, action-to-task transfer operators, Schur/transient geometry, and
matched counterfactual transfer error.

The implementation is deliberately a CPU diagnostic primitive.  It does not
claim causal identification, does not query runtime simulator successors, and
does not replace the frozen-LeWM planning protocol.  It is intended to be
called by later manifest-driven audits after train/calibration/test splits have
already been frozen.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import schur


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty rank-2 array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _symmetric(value: np.ndarray) -> np.ndarray:
    return (value + value.T) / 2.0


def _ridge_solve(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    if ridge < 0.0 or not math.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")
    gram = design.T @ design
    scale = max(float(np.trace(gram)) / max(design.shape[1], 1), 1.0)
    regularizer = ridge * scale * np.eye(design.shape[1], dtype=np.float64)
    return np.linalg.solve(gram + regularizer, design.T @ target)


def fit_action_lifted_system(
    eta: np.ndarray,
    next_eta: np.ndarray,
    action: np.ndarray,
    task: np.ndarray,
    *,
    ridge: float = 1e-5,
    include_interaction: bool = False,
) -> dict[str, Any]:
    """Fit local controlled dynamics and task readout from aligned transitions.

    ``eta``, ``next_eta``, ``action`` and ``task`` must share the transition
    count.  If ``include_interaction`` is true, the fit includes the bilinear
    features ``eta_i * action_j`` and returns one interaction matrix per action
    coordinate.  This is a local bilinear approximation, not a global causal
    model.
    """

    eta = _matrix(eta, "eta")
    next_eta = _matrix(next_eta, "next_eta")
    action = _matrix(action, "action")
    task = _matrix(task, "task")
    n, state_dim = eta.shape
    if next_eta.shape != eta.shape:
        raise ValueError("eta and next_eta must have identical shapes")
    if action.shape[0] != n or task.shape[0] != n:
        raise ValueError("all transition arrays must share the first dimension")
    action_dim = action.shape[1]

    features = [eta, action]
    if include_interaction:
        features.append((eta[:, :, None] * action[:, None, :]).reshape(n, -1))
    features.append(np.ones((n, 1), dtype=np.float64))
    design = np.concatenate(features, axis=1)
    coefficients = _ridge_solve(design, next_eta, ridge)
    offset = 0
    A = coefficients[offset : offset + state_dim].T
    offset += state_dim
    B = coefficients[offset : offset + action_dim].T
    offset += action_dim
    bilinear: list[np.ndarray] = []
    if include_interaction:
        raw = coefficients[offset : offset + state_dim * action_dim].T
        bilinear = [raw[:, j * state_dim : (j + 1) * state_dim] for j in range(action_dim)]
        offset += state_dim * action_dim
    intercept = coefficients[offset]
    dynamics_prediction = design @ coefficients

    task_design = np.concatenate((eta, np.ones((n, 1), dtype=np.float64)), axis=1)
    task_coefficients = _ridge_solve(task_design, task, ridge)
    D = task_coefficients[:-1].T
    task_intercept = task_coefficients[-1]
    task_prediction = task_design @ task_coefficients

    return {
        "A": A,
        "B": B,
        "D": D,
        "bilinear_B": bilinear,
        "intercept": intercept,
        "task_intercept": task_intercept,
        "dynamics_rmse": float(np.sqrt(np.mean((dynamics_prediction - next_eta) ** 2))),
        "task_rmse": float(np.sqrt(np.mean((task_prediction - task) ** 2))),
        "n_samples": int(n),
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "task_dim": int(task.shape[1]),
        "include_interaction": bool(include_interaction),
    }


def persistent_excitation(
    eta: np.ndarray,
    action: np.ndarray,
    *,
    threshold: float = 1e-6,
) -> dict[str, Any]:
    """Measure the empirical excitation of the state-action regression vector."""

    eta = _matrix(eta, "eta")
    action = _matrix(action, "action")
    if eta.shape[0] != action.shape[0]:
        raise ValueError("eta and action must share the first dimension")
    if threshold < 0.0 or not math.isfinite(threshold):
        raise ValueError("threshold must be finite and non-negative")
    psi = np.concatenate((eta, action), axis=1)
    covariance = _symmetric((psi.T @ psi) / float(psi.shape[0]))
    eigenvalues = np.linalg.eigvalsh(covariance)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    return {
        "gram": covariance,
        "eigenvalues": eigenvalues,
        "lambda_min": minimum,
        "lambda_max": maximum,
        "rank": int(np.linalg.matrix_rank(covariance)),
        "dimension": int(covariance.shape[0]),
        "condition_number": float(maximum / minimum) if minimum > 0.0 else float("inf"),
        "threshold": float(threshold),
        "passes": bool(minimum >= threshold),
    }


def action_excitation(
    action: np.ndarray,
    *,
    threshold: float = 1e-6,
) -> dict[str, Any]:
    """Check excitation in the input coordinates alone.

    Joint ``[eta, action]`` rank is usually impossible for high-dimensional
    latents and short exports.  This input-only diagnostic is the relevant
    first gate for whether the columns of B can be separated at all.
    """

    action = _matrix(action, "action")
    if threshold < 0.0 or not math.isfinite(threshold):
        raise ValueError("threshold must be finite and non-negative")
    centered = action - np.mean(action, axis=0, keepdims=True)
    gram = _symmetric((centered.T @ centered) / float(len(action)))
    values = np.linalg.eigvalsh(gram)
    minimum, maximum = float(values[0]), float(values[-1])
    return {
        "gram": gram,
        "eigenvalues": values,
        "lambda_min": minimum,
        "lambda_max": maximum,
        "rank": int(np.linalg.matrix_rank(gram)),
        "dimension": int(gram.shape[0]),
        "condition_number": float(maximum / minimum) if minimum > 0.0 else float("inf"),
        "threshold": float(threshold),
        "passes": bool(minimum >= threshold),
    }


def residualized_action_excitation(
    eta: np.ndarray,
    action: np.ndarray,
    *,
    ridge: float = 1e-5,
    threshold: float = 1e-6,
) -> dict[str, Any]:
    """Check input excitation left after linear prediction from the state."""

    eta = _matrix(eta, "eta")
    action = _matrix(action, "action")
    if len(eta) != len(action):
        raise ValueError("eta and action must align")
    design = np.concatenate((eta, np.ones((len(eta), 1))), axis=1)
    coefficients = _ridge_solve(design, action, ridge)
    residual = action - design @ coefficients
    result = action_excitation(residual, threshold=threshold)
    result["ridge"] = float(ridge)
    result["residual_rmse"] = float(np.sqrt(np.mean(residual**2)))
    return result


def _metric_square(metric: np.ndarray | None, dimension: int, name: str) -> np.ndarray:
    if metric is None:
        return np.eye(dimension, dtype=np.float64)
    metric = _matrix(metric, name)
    if metric.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape {(dimension, dimension)}")
    eigenvalues = np.linalg.eigvalsh(_symmetric(metric))
    if float(eigenvalues[0]) < -1e-8:
        raise ValueError(f"{name} must be positive semidefinite")
    return _symmetric(metric)


def finite_horizon_diagnostics(
    A: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    *,
    horizon: int,
    task_metric: np.ndarray | None = None,
    action_metric: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute Gramians, PBH ranks, transfer operators and transient gains."""

    A = _matrix(A, "A")
    B = _matrix(B, "B")
    D = _matrix(D, "D")
    state_dim = A.shape[0]
    if A.shape[1] != state_dim or B.shape[0] != state_dim or D.shape[1] != state_dim:
        raise ValueError("A, B, D have incompatible dimensions")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    task_metric = _metric_square(task_metric, D.shape[0], "task_metric")
    action_metric = _metric_square(action_metric, B.shape[1], "action_metric")

    observability = np.zeros((state_dim, state_dim), dtype=np.float64)
    controllability = np.zeros((state_dim, state_dim), dtype=np.float64)
    power = np.eye(state_dim, dtype=np.float64)
    transfers: list[np.ndarray] = []
    transient_gains: list[float] = []
    for _step in range(1, horizon + 1):
        observability += power.T @ D.T @ task_metric @ D @ power
        controllability += power @ B @ action_metric @ B.T @ power.T
        transfers.append(D @ power @ B)
        transient_gains.append(float(np.linalg.norm(power, ord=2)))
        power = A @ power
    observability = _symmetric(observability)
    controllability = _symmetric(controllability)

    eigenvalues = np.linalg.eigvals(A)
    pbh_rows: list[dict[str, Any]] = []
    for value in eigenvalues:
        pencil = value * np.eye(state_dim, dtype=np.complex128) - A
        controllability_rank = int(np.linalg.matrix_rank(np.concatenate((pencil, B), axis=1)))
        observability_rank = int(np.linalg.matrix_rank(np.concatenate((pencil, D), axis=0)))
        pbh_rows.append({
            "eigenvalue_real": float(np.real(value)),
            "eigenvalue_imag": float(np.imag(value)),
            "controllability_rank": controllability_rank,
            "observability_rank": observability_rank,
            "controllable": bool(controllability_rank == state_dim),
            "observable": bool(observability_rank == state_dim),
        })

    schur_q, schur_t = schur(A, output="complex")
    diagonal = np.diag(np.diag(schur_t))
    schur_offdiag = schur_t - diagonal
    return {
        "horizon": int(horizon),
        "observability_gramian": observability,
        "controllability_gramian": controllability,
        "observability_eigenvalues": np.linalg.eigvalsh(observability),
        "controllability_eigenvalues": np.linalg.eigvalsh(controllability),
        "transfer_operators": transfers,
        "transient_gains": transient_gains,
        "max_transient_gain": float(max(transient_gains)),
        "spectrum": eigenvalues,
        "pbh": pbh_rows,
        "schur_offdiag_frobenius": float(np.linalg.norm(schur_offdiag, ord="fro")),
        "schur_unitary_residual": float(np.linalg.norm(schur_q.conj().T @ schur_q - np.eye(state_dim))),
    }


def counterfactual_transfer_error(
    predicted_task: np.ndarray,
    real_task: np.ndarray,
    *,
    task_metric: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare same-source candidate contrasts in predicted and real task space.

    Inputs have shape ``[sources, candidates, task_dim]``.  Pairwise contrasts
    remove the source-specific center and therefore target intervention effects,
    not merely endpoint reconstruction quality.
    """

    predicted_task = np.asarray(predicted_task, dtype=np.float64)
    real_task = np.asarray(real_task, dtype=np.float64)
    if predicted_task.ndim != 3 or real_task.shape != predicted_task.shape:
        raise ValueError("predicted_task and real_task must have identical rank-3 shapes")
    task_dim = predicted_task.shape[-1]
    metric = _metric_square(task_metric, task_dim, "task_metric")
    predicted_contrast = predicted_task[:, :, None, :] - predicted_task[:, None, :, :]
    real_contrast = real_task[:, :, None, :] - real_task[:, None, :, :]
    difference = predicted_contrast - real_contrast
    upper = np.triu(np.ones(difference.shape[1:3], dtype=bool), k=1)
    selected = difference[:, upper]
    quadratic = np.einsum("npi,ij,npj->np", selected, metric, selected)
    rmse = float(np.sqrt(np.mean(quadratic))) if selected.size else 0.0
    return {
        "rmse": rmse,
        "n_sources": int(predicted_task.shape[0]),
        "n_candidates": int(predicted_task.shape[1]),
        "n_pairwise_contrasts": int(selected.shape[0] * selected.shape[1]),
    }


def build_report(
    eta: np.ndarray,
    next_eta: np.ndarray,
    action: np.ndarray,
    task: np.ndarray,
    *,
    horizons: tuple[int, ...] = (1, 2, 4),
    ridge: float = 1e-5,
    pe_threshold: float = 1e-6,
    include_interaction: bool = True,
) -> dict[str, Any]:
    """Build a JSON-safe audit report for one frozen split."""

    fit = fit_action_lifted_system(
        eta, next_eta, action, task, ridge=ridge, include_interaction=include_interaction
    )
    pe = persistent_excitation(eta, action, threshold=pe_threshold)
    diagnostics = {
        str(horizon): finite_horizon_diagnostics(
            fit["A"], fit["B"], fit["D"], horizon=int(horizon)
        )
        for horizon in horizons
    }
    report: dict[str, Any] = {
        "protocol": {
            "study_scope": "covariance_predictor_unified",
            "lift_family": "action_history_local_linear_bilinear",
            "claim_boundary": "system-identification and action-to-task diagnostic; not causal identification, online planning, or universal transfer",
            "ridge": float(ridge),
            "persistent_excitation_threshold": float(pe_threshold),
        },
        "fit": fit,
        "persistent_excitation": pe,
        "finite_horizon": diagnostics,
    }
    return _json_safe(report)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _synthetic_fixture(seed: int = 7, samples: int = 512) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    A = np.diag([0.92, 0.78, 0.55])
    B = np.asarray([[1.0, 0.0], [0.0, 0.35], [0.1, 0.2]])
    D = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    eta = rng.normal(size=(samples, 3))
    action = rng.normal(size=(samples, 2))
    next_eta = eta @ A.T + action @ B.T + 0.01 * rng.normal(size=(samples, 3))
    task = eta @ D.T
    return eta, next_eta, action, task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=512)
    args = parser.parse_args()
    print("Running action-lifted diagnostics to separate observability, controllability, and transfer.")
    arrays = _synthetic_fixture(seed=args.seed, samples=args.samples)
    report = build_report(*arrays)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote action-lifted audit to {output}")


if __name__ == "__main__":
    main()
