"""CPU primitives for projected, memory-bearing, hybrid latent dynamics.

This module complements ``run_action_lifted_system_audit.py``.  It keeps the
diagnostic object explicit:

    projected state + finite memory + action input + task readout.

The functions are deliberately small and auditable.  They support memory
kernel fitting, a body-frame SE(2) twist, mode-conditioned local operators,
memory-augmented transfer/resolvent quantities, discrete passivity residuals,
and multiscale transfer comparison.  They are not a causal-identification
theorem and never query runtime simulator successors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from experiments.empirical_spectral_audit.run_action_lifted_system_audit import (
        _matrix,
        _ridge_solve,
        _symmetric,
    )
except ModuleNotFoundError:  # direct invocation from the experiment directory
    from run_action_lifted_system_audit import _matrix, _ridge_solve, _symmetric


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap radians to [-pi, pi)."""

    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def se2_body_twist(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Return log(before^{-1} after) as [vx, vy, omega] in the body frame."""

    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.shape != (3,) or after.shape != (3,):
        raise ValueError("SE(2) poses must have shape (3,) = [x, y, theta]")
    theta = float(wrap_angle(after[2] - before[2]))
    c, s = math.cos(float(before[2])), math.sin(float(before[2]))
    world_delta = after[:2] - before[:2]
    body_delta = np.asarray([[c, s], [-s, c]], dtype=np.float64) @ world_delta
    if abs(theta) < 1e-7:
        v_matrix = np.eye(2) + 0.5 * theta * np.asarray([[0.0, -1.0], [1.0, 0.0]])
    else:
        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / theta
        v_matrix = np.asarray([[a, -b], [b, a]], dtype=np.float64)
    velocity = np.linalg.solve(v_matrix, body_delta)
    return np.asarray([velocity[0], velocity[1], theta], dtype=np.float64)


def fit_memory_kernel(
    eta: np.ndarray,
    next_eta: np.ndarray,
    action: np.ndarray,
    *,
    memory_length: int,
    ridge: float = 1e-5,
) -> dict[str, Any]:
    """Fit a finite-memory model without duplicating the current state.

    The returned convention is

        eta[t+1] = A eta[t] + K[0] eta[t-1] + ... + B action[t] + c.

    Thus ``memory_length=1`` is the Markov model and returns an empty kernel
    list.  Keeping the current-state coefficient separate is important: the
    previous implementation included ``eta[t]`` both in ``A`` and in the
    first kernel, which silently changed the companion dynamics.
    """

    eta = _matrix(eta, "eta")
    next_eta = _matrix(next_eta, "next_eta")
    action = _matrix(action, "action")
    if memory_length < 1:
        raise ValueError("memory_length must be positive")
    if next_eta.shape != eta.shape or action.shape[0] != eta.shape[0]:
        raise ValueError("aligned arrays have incompatible shapes")
    if eta.shape[0] <= memory_length:
        raise ValueError("not enough transitions for the requested memory length")
    rows = []
    targets = []
    for t in range(memory_length - 1, eta.shape[0] - 1):
        history = [eta[t - lag] for lag in range(memory_length)]
        rows.append(np.concatenate((*history, action[t], [1.0])))
        targets.append(next_eta[t])
    design = np.asarray(rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    coefficients = _ridge_solve(design, target, ridge)
    state_dim = eta.shape[1]
    action_dim = action.shape[1]
    offset = 0
    A = coefficients[offset : offset + state_dim].T
    offset += state_dim
    kernels = []
    for _lag in range(1, memory_length):
        kernels.append(coefficients[offset : offset + state_dim].T)
        offset += state_dim
    B = coefficients[offset : offset + action_dim].T
    intercept = coefficients[offset + action_dim]
    prediction = design @ coefficients
    return {
        "A": A,
        "kernels": kernels,
        "B": B,
        "intercept": intercept,
        "memory_length": int(memory_length),
        "n_samples": int(design.shape[0]),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
    }


def memory_companion_operator(
    A: np.ndarray,
    kernels: list[np.ndarray],
) -> np.ndarray:
    """Build the companion operator for [eta_t,...,eta_{t-L+1}]."""

    A = _matrix(A, "A")
    state_dim = A.shape[0]
    if A.shape[1] != state_dim:
        raise ValueError("A must be square")
    if any(np.asarray(kernel).shape != (state_dim, state_dim) for kernel in kernels):
        raise ValueError("every memory kernel must match A's square shape")
    length = len(kernels) + 1
    companion = np.zeros((state_dim * length, state_dim * length), dtype=np.float64)
    top_row = [A, *kernels]
    companion[:state_dim, : state_dim * length] = np.concatenate(top_row, axis=1)
    if length > 1:
        companion[state_dim:, :-state_dim] = np.eye(state_dim * (length - 1))
    return companion


def memory_transfer(
    A: np.ndarray,
    kernels: list[np.ndarray],
    B: np.ndarray,
    D: np.ndarray,
    *,
    horizon: int,
) -> dict[str, Any]:
    """Compute transfer and resolvent quantities for finite-memory dynamics."""

    A_aug = memory_companion_operator(A, kernels)
    B = _matrix(B, "B")
    D = _matrix(D, "D")
    state_dim = np.asarray(A).shape[0]
    if B.shape[0] != state_dim or D.shape[1] != state_dim or horizon <= 0:
        raise ValueError("incompatible dimensions or non-positive horizon")
    B_aug = np.vstack((B, np.zeros((A_aug.shape[0] - state_dim, B.shape[1]))))
    D_aug = np.hstack((D, np.zeros((D.shape[0], A_aug.shape[0] - state_dim))))
    power = np.eye(A_aug.shape[0], dtype=np.float64)
    transfers = []
    gains = []
    for _step in range(horizon):
        transfers.append(D_aug @ power @ B_aug)
        gains.append(float(np.linalg.norm(D_aug @ power @ B_aug, ord=2)))
        power = A_aug @ power
    resolvent_points = (0.0, 0.5, 0.9, 1.1)
    resolvent_norms = {}
    identity = np.eye(A_aug.shape[0], dtype=np.complex128)
    for point in resolvent_points:
        matrix = point * identity - A_aug
        try:
            resolvent_norms[str(point)] = float(np.linalg.norm(np.linalg.inv(matrix), ord=2))
        except np.linalg.LinAlgError:
            resolvent_norms[str(point)] = None
    return {
        "memory_length": int(len(kernels) + 1),
        "augmented_state_dim": int(A_aug.shape[0]),
        "companion_operator": A_aug,
        "transfer_operators": transfers,
        "task_transfer_gains": gains,
        "max_task_transfer_gain": float(max(gains)),
        "resolvent_norms": resolvent_norms,
    }


def _psd_square_root(metric: np.ndarray) -> np.ndarray:
    """Return the symmetric square root of a PSD matrix."""

    values, vectors = np.linalg.eigh(_symmetric(metric))
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def _metric_square(metric: np.ndarray | None, dimension: int, name: str) -> np.ndarray:
    """Validate an output/input metric for projected-system diagnostics."""

    if metric is None:
        return np.eye(dimension, dtype=np.float64)
    value = _matrix(metric, name)
    if value.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape {(dimension, dimension)}")
    eigenvalues = np.linalg.eigvalsh(_symmetric(value))
    if float(eigenvalues[0]) < -1e-8:
        raise ValueError(f"{name} must be positive semidefinite")
    return _symmetric(value)


def finite_horizon_hankel_geometry(
    A: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    *,
    horizon: int,
    task_metric: np.ndarray | None = None,
    action_metric: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute finite-horizon controllable-observable Hankel geometry.

    The singular values are those of the finite input-output map after
    controllability and observability are combined.  They are invariant to a
    similarity change of latent coordinates when the input/output metrics are
    held fixed, unlike Euclidean covariance alignment.
    """

    A = _matrix(A, "A")
    B = _matrix(B, "B")
    D = _matrix(D, "D")
    if A.shape[0] != A.shape[1] or B.shape[0] != A.shape[0] or D.shape[1] != A.shape[0]:
        raise ValueError("A, B, D have incompatible dimensions")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    task_metric = _symmetric(_metric_square(task_metric, D.shape[0], "task_metric"))
    action_metric = _symmetric(_metric_square(action_metric, B.shape[1], "action_metric"))
    wc = np.zeros_like(A)
    wo = np.zeros_like(A)
    power = np.eye(A.shape[0], dtype=np.float64)
    for _ in range(horizon):
        wc += power @ B @ action_metric @ B.T @ power.T
        wo += power.T @ D.T @ task_metric @ D @ power
        power = A @ power
    wc = _symmetric(wc)
    wo = _symmetric(wo)
    sqrt_wc = _psd_square_root(wc)
    hankel = _symmetric(sqrt_wc @ wo @ sqrt_wc)
    values = np.clip(np.linalg.eigvalsh(hankel), 0.0, None)
    singular_values = np.sqrt(values[::-1])
    return {
        "horizon": int(horizon),
        "controllability_gramian": wc,
        "observability_gramian": wo,
        "finite_hankel_operator": hankel,
        "hankel_singular_values": singular_values,
        "hankel_effective_rank": int(np.sum(singular_values > max(float(singular_values[0]) * 1e-8, 1e-12))) if singular_values.size else 0,
    }


def frequency_response(
    A: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    frequencies: np.ndarray,
    *,
    task_metric: np.ndarray | None = None,
    action_metric: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate the finite-dimensional input-output resolvent on the unit circle."""

    A = _matrix(A, "A")
    B = _matrix(B, "B")
    D = _matrix(D, "D")
    frequencies = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    if frequencies.size == 0 or not np.isfinite(frequencies).all():
        raise ValueError("frequencies must be a non-empty finite vector")
    task_metric = _symmetric(_metric_square(task_metric, D.shape[0], "task_metric"))
    action_metric = _symmetric(_metric_square(action_metric, B.shape[1], "action_metric"))
    task_root = _psd_square_root(task_metric)
    action_root = _psd_square_root(action_metric)
    responses = []
    gains = []
    identity = np.eye(A.shape[0], dtype=np.complex128)
    for omega in frequencies:
        system_matrix = np.exp(1j * omega) * identity - A
        # Solve only for the action columns.  Forming a dense inverse is both
        # slower and less stable for the 384-dimensional real exports.
        response = task_root @ D @ np.linalg.solve(system_matrix, B) @ action_root
        responses.append(response)
        gains.append(float(np.linalg.norm(response, ord=2)))
    return {
        "frequencies": frequencies,
        "responses": np.asarray(responses),
        "gains": np.asarray(gains),
        "max_gain": float(max(gains)),
        "argmax_frequency": float(frequencies[int(np.argmax(gains))]),
    }


def pathwise_hybrid_transfer(
    mode_path: list[str] | tuple[str, ...],
    operators: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    """Return D_{m_H} A_{m_{H-1}}...A_{m_0} B_{m_0}."""

    path = [str(mode) for mode in mode_path]
    if len(path) < 2:
        raise ValueError("mode_path must contain at least an initial and terminal mode")
    if any(mode not in operators for mode in path):
        raise ValueError("mode_path contains an operator-free mode")
    initial = operators[path[0]]
    state_map = _matrix(initial["B"], f"B[{path[0]}]")
    for mode in path[:-1]:
        state_map = _matrix(operators[mode]["A"], f"A[{mode}]") @ state_map
    terminal_D = _matrix(operators[path[-1]]["D"], f"D[{path[-1]}]")
    if terminal_D.shape[1] != state_map.shape[0]:
        raise ValueError("hybrid path operator dimensions do not match")
    return terminal_D @ state_map


def hybrid_path_gain_summary(
    paths: list[list[str] | tuple[str, ...]],
    operators: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Summarize pathwise transfer gains without averaging away switches."""

    records = []
    for path in paths:
        transfer = pathwise_hybrid_transfer(path, operators)
        records.append({"path": [str(mode) for mode in path], "gain": float(np.linalg.norm(transfer, ord=2))})
    gains = [record["gain"] for record in records]
    return {
        "paths": records,
        "max_gain": float(max(gains)) if gains else 0.0,
        "mean_gain": float(np.mean(gains)) if gains else 0.0,
        "n_paths": len(records),
    }


def fit_mode_conditioned_operators(
    eta: np.ndarray,
    next_eta: np.ndarray,
    action: np.ndarray,
    modes: np.ndarray,
    *,
    ridge: float = 1e-5,
) -> dict[str, Any]:
    """Fit separate local operators and excitation diagnostics per mode."""

    eta = _matrix(eta, "eta")
    next_eta = _matrix(next_eta, "next_eta")
    action = _matrix(action, "action")
    modes = np.asarray(modes)
    if modes.ndim != 1 or len(modes) != len(eta) or next_eta.shape != eta.shape:
        raise ValueError("mode labels and transition arrays are incompatible")
    result: dict[str, Any] = {}
    for mode in sorted(set(modes.tolist()), key=str):
        mask = modes == mode
        if int(mask.sum()) < eta.shape[1] + action.shape[1] + 2:
            result[str(mode)] = {"status": "insufficient_support", "n_samples": int(mask.sum())}
            continue
        design = np.concatenate((eta[mask], action[mask], np.ones((int(mask.sum()), 1))), axis=1)
        coefficients = _ridge_solve(design, next_eta[mask], ridge)
        state_dim = eta.shape[1]
        action_dim = action.shape[1]
        result[str(mode)] = {
            "status": "fit",
            "n_samples": int(mask.sum()),
            "A": coefficients[:state_dim].T,
            "B": coefficients[state_dim : state_dim + action_dim].T,
            "intercept": coefficients[-1],
            "rank": int(np.linalg.matrix_rank(design)),
            "condition_number": float(np.linalg.cond(design)),
        }
    return result


def passivity_residual(
    energy_before: np.ndarray,
    energy_after: np.ndarray,
    port_output: np.ndarray,
    action: np.ndarray,
    *,
    dissipation_margin: float = 0.0,
) -> dict[str, Any]:
    """Audit the discrete inequality E_after <= E_before + y^T u - margin."""

    before = np.asarray(energy_before, dtype=np.float64).reshape(-1)
    after = np.asarray(energy_after, dtype=np.float64).reshape(-1)
    output = _matrix(port_output, "port_output")
    action = _matrix(action, "action")
    if len(before) != len(after) or output.shape[0] != len(before) or action.shape[0] != len(before):
        raise ValueError("energy, output and action counts must match")
    if output.shape[1] != action.shape[1]:
        raise ValueError("port_output and action must have the same width")
    work = np.sum(output * action, axis=1)
    residual = before + work - after - float(dissipation_margin)
    return {
        "residual": residual,
        "mean_residual": float(np.mean(residual)),
        "violation_rate": float(np.mean(residual < 0.0)),
        "min_residual": float(np.min(residual)),
        "passes_nonnegative": bool(np.all(residual >= -1e-10)),
    }


def multiscale_transfer_consistency(
    transfer_by_scale: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare normalized transfer operators estimated at multiple time scales."""

    if len(transfer_by_scale) < 2:
        raise ValueError("at least two scales are required")
    matrices = {str(key): _matrix(value, f"transfer[{key}]") for key, value in transfer_by_scale.items()}
    reference_key = sorted(matrices)[0]
    reference = matrices[reference_key]
    reference_norm = max(float(np.linalg.norm(reference, ord="fro")), 1e-12)
    pairwise: dict[str, float] = {}
    for key in sorted(matrices):
        if key == reference_key:
            continue
        value = matrices[key]
        if value.shape != reference.shape:
            raise ValueError("all scale transfer operators must share a shape")
        value_norm = max(float(np.linalg.norm(value, ord="fro")), 1e-12)
        pairwise[f"{reference_key}__{key}"] = float(
            np.linalg.norm(reference / reference_norm - value / value_norm, ord="fro")
        )
    return {
        "reference_scale": reference_key,
        "normalized_pairwise_frobenius": pairwise,
        "mean_discrepancy": float(np.mean(list(pairwise.values()))),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Running projected-memory, SE(2), hybrid-mode, and transfer diagnostics.")
    before = np.asarray([0.0, 1.0, 0.2])
    after = np.asarray([0.1, 1.08, 0.25])
    report = {
        "protocol": {
            "study_scope": "covariance_predictor_unified",
            "claim_boundary": "projected hybrid physics diagnostics; not causal identification or online planning",
        },
        "se2_example_twist": se2_body_twist(before, after),
        "memory_resolvent_example": memory_transfer(
            np.diag([0.8, 0.6]),
            [np.diag([0.1, 0.05])],
            np.eye(2),
            np.eye(2),
            horizon=4,
        ),
        "multiscale_example": multiscale_transfer_consistency(
            {"1": np.eye(2), "2": 0.95 * np.eye(2), "4": 0.8 * np.eye(2)}
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n")
    print(f"Wrote projected hybrid physics audit to {output}")


if __name__ == "__main__":
    main()
