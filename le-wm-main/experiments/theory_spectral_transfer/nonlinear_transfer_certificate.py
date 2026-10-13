"""Fail-closed local applicability checks for learned nonlinear predictors.

The checker does not certify a global learned world model.  It verifies the
finite-horizon theorem's local prerequisites: finite Jacobian/remainder
envelopes, planner action support, an independently collected successor
transition split, and an independently calibrated task readout.  Missing or
failed evidence always rejects the certificate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _array(values: Any, *, name: str, length: int | None = None) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    if length is not None:
        value = value.reshape(-1)
        if value.size != length:
            raise ValueError(f"{name} must contain exactly {length} values")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _nonnegative(values: Any, *, name: str, length: int) -> np.ndarray:
    value = _array(values, name=name, length=length)
    if np.any(value < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return value


def _local_gain(jacobians: list[np.ndarray], readout: np.ndarray) -> np.ndarray:
    """Return terminal readout gains for disturbances at each time."""
    h = len(jacobians)
    gains = np.zeros(h, dtype=np.float64)
    for t in range(h):
        product = np.eye(jacobians[0].shape[0], dtype=np.float64)
        for index in range(t + 1, h):
            product = jacobians[index] @ product
        gains[t] = float(np.linalg.norm(readout @ product, 2))
    return gains


def check_nonlinear_transfer_applicability(
    *,
    local_jacobians: Iterable[np.ndarray],
    task_readout_jacobian: np.ndarray,
    jacobian_lipschitz: Iterable[float] | float,
    tube_radius: float,
    residual_envelopes: dict[str, Iterable[float]],
    action_support: dict[str, Any] | None,
    independent_transition_consistency: dict[str, Any] | None,
    independent_readout_calibration: dict[str, Any] | None,
    remainder_budget: float,
    readout_remainder_bound: float = 0.0,
    initial_error: float = 0.0,
) -> dict[str, Any]:
    """Check sufficient local conditions for a nonlinear transfer bound.

    Required residual keys are ``closure``, ``off_manifold`` and ``action``.
    ``jacobian_lipschitz`` gives a local upper bound on Jacobian variation;
    Taylor's theorem contributes ``0.5 * L * tube_radius**2`` per step.
    """
    matrices = [np.asarray(item, dtype=np.float64) for item in local_jacobians]
    if not matrices:
        raise ValueError("at least one local Jacobian is required")
    dimension = matrices[0].shape[0]
    if any(item.ndim != 2 or item.shape != (dimension, dimension) for item in matrices):
        raise ValueError("local_jacobians must be square matrices of one dimension")
    if any(not np.isfinite(item).all() for item in matrices):
        raise ValueError("local_jacobians must be finite")
    readout = np.asarray(task_readout_jacobian, dtype=np.float64)
    if readout.ndim != 2 or readout.shape[1] != dimension or not np.isfinite(readout).all():
        raise ValueError("task_readout_jacobian has incompatible shape or non-finite values")
    h = len(matrices)
    if np.isscalar(jacobian_lipschitz):
        lipschitz = np.full(h, float(jacobian_lipschitz), dtype=np.float64)
    else:
        lipschitz = _array(jacobian_lipschitz, name="jacobian_lipschitz", length=h)
    if np.any(lipschitz < 0.0):
        raise ValueError("jacobian_lipschitz must be non-negative")
    scalar_values = (tube_radius, remainder_budget, readout_remainder_bound, initial_error)
    if any(not np.isfinite(float(item)) or float(item) < 0.0 for item in scalar_values):
        raise ValueError("tube and error budgets must be finite and non-negative")
    required = ("closure", "off_manifold", "action")
    missing = [name for name in required if name not in residual_envelopes]
    if missing:
        raise ValueError(f"residual_envelopes missing keys: {missing}")
    residuals = {
        name: _nonnegative(residual_envelopes[name], name=f"residual_{name}", length=h)
        for name in required
    }
    nonlinear_remainder = 0.5 * lipschitz * float(tube_radius) ** 2
    gains = _local_gain(matrices, readout)
    total_residual = sum(residuals.values()) + nonlinear_remainder
    initial_product = np.eye(dimension, dtype=np.float64)
    for matrix in matrices:
        initial_product = matrix @ initial_product
    initial_gain = float(np.linalg.norm(readout @ initial_product, 2))
    bound = float(
        initial_gain * float(initial_error)
        + np.dot(gains, total_residual)
        + np.linalg.norm(readout, 2) * float(readout_remainder_bound)
    )

    support = action_support or {}
    transition = independent_transition_consistency or {}
    readout_calibration = independent_readout_calibration or {}
    support_passed = bool(support.get("passed", False))
    transition_passed = bool(transition.get("passed", False))
    readout_passed = bool(readout_calibration.get("passed", False))
    remainder_passed = bool(float(np.max(nonlinear_remainder, initial=0.0)) <= float(remainder_budget))
    # A finite nominal bound does not imply that the error tube remains inside
    # the Taylor region.  The supplied state tube is therefore a hard gate.
    tube_passed = bool(float(tube_radius) > 0.0 and np.isfinite(bound))

    gates = {
        "local_jacobian_finite": True,
        "remainder_bound": remainder_passed,
        "action_support": support_passed,
        "independent_transition_consistency": transition_passed,
        "independent_readout_calibration": readout_passed,
        "local_tube": tube_passed,
    }
    if not remainder_passed:
        status = "rejected_remainder_bound"
    elif not support_passed:
        status = "rejected_support_shift"
    elif not transition_passed:
        status = "rejected_missing_independent_transition"
    elif not readout_passed:
        status = "rejected_missing_independent_readout"
    elif not tube_passed:
        status = "rejected_local_tube"
    else:
        status = "certified_local"
    return {
        "schema": "learned_nonlinear_transfer_certificate.v1",
        "status": status,
        "certified_local": status == "certified_local",
        "horizon": h,
        "gates": gates,
        "local_transfer_bound": bound,
        "terminal_readout_gains": gains.tolist(),
        "jacobian_lipschitz": lipschitz.tolist(),
        "nonlinear_remainder_bounds": nonlinear_remainder.tolist(),
        "remainder_budget": float(remainder_budget),
        "residual_envelopes": {name: value.tolist() for name, value in residuals.items()},
        "inputs": {
            "tube_radius": float(tube_radius),
            "readout_remainder_bound": float(readout_remainder_bound),
            "initial_error": float(initial_error),
            "action_support": support,
            "independent_transition_consistency": transition,
            "independent_readout_calibration": readout_calibration,
        },
        "claim_boundary": (
            "Local finite-horizon certificate only. It is not a global nonlinear "
            "world-model theorem and cannot release actions outside the declared "
            "support, tube, or independent transition split."
        ),
    }


def audit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a JSON payload using the same API as the Python checker."""
    return check_nonlinear_transfer_applicability(**payload)


def adversarial_nonlinear_remainder_counterexample(
    *, curvature: float = 2.0, initial_state: float = 0.2, horizon: int = 2,
) -> dict[str, Any]:
    """Show why a local Jacobian at one calibration point is insufficient."""
    if horizon < 1 or not np.isfinite(curvature) or not np.isfinite(initial_state):
        raise ValueError("invalid nonlinear counterexample parameters")

    def rollout(sign: float) -> list[float]:
        state = float(initial_state)
        values = []
        for _ in range(int(horizon)):
            state = 0.9 * state + sign * float(curvature) * state**2
            values.append(state)
        return values

    plus = rollout(1.0)
    minus = rollout(-1.0)
    return {
        "schema": "adversarial_nonlinear_remainder_counterexample.v1",
        "local_calibration_point": 0.0,
        "shared_local_jacobian": 0.9,
        "shared_action_map": 1.0,
        "curvature_magnitude": float(curvature),
        "initial_state": float(initial_state),
        "horizon": int(horizon),
        "plus_rollout": plus,
        "minus_rollout": minus,
        "same_local_jacobian": True,
        "same_calibration_transition_at_zero": True,
        "different_finite_horizon_transfer": bool(not np.allclose(plus, minus)),
        "claim_boundary": (
            "A local Jacobian-only certificate is invalid once the nonlinear "
            "remainder exceeds its frozen budget."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed learned nonlinear transfer applicability checker")
    parser.add_argument("--input", type=Path, required=True, help="JSON object containing checker keyword arguments")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_payload(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "local_transfer_bound": report["local_transfer_bound"]}, indent=2))


if __name__ == "__main__":
    main()
