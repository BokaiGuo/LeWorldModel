"""HTSC release gate for horizon-conditioned learned-memory control.

This module is the executable part of
``HORIZON_CONDITIONED_TRANSFER_SUFFICIENT_CONTROL_20261013.md``.  It keeps
memory selection and control release separate: a selector may be useful for
diagnosis, but only the declared horizon policy can be released.
"""

from __future__ import annotations

from typing import Any


def evaluate_horizon_sufficiency(
    *,
    horizon: int,
    declared_variant: str,
    observed_variant: str,
    support_gate: dict[str, Any],
    action_identifiability_gate: dict[str, Any],
    successor_consistency_gate: dict[str, Any],
    risk_bound: float | None,
    risk_threshold: float,
    selector_stability: float | None = None,
) -> dict[str, Any]:
    """Evaluate the four HTSC gates without tuning on held-out outcomes."""

    if int(horizon) < 1:
        raise ValueError("horizon must be positive")
    if risk_bound is not None and risk_bound < 0.0:
        raise ValueError("risk_bound must be non-negative")
    if risk_threshold < 0.0:
        raise ValueError("risk_threshold must be non-negative")
    gates = {
        "declared_policy": observed_variant == declared_variant,
        "support": bool(support_gate.get("passed", False)),
        "action_identifiability": bool(action_identifiability_gate.get("passed", False)),
        "successor_consistency": bool(successor_consistency_gate.get("passed", False)),
        "finite_horizon_risk": bool(risk_bound is not None and risk_bound <= float(risk_threshold)),
    }
    passed = all(gates.values())
    return {
        "schema": "horizon_transfer_sufficiency_gate.v1",
        "status": "permitted" if passed else "fallback_fail_closed",
        "horizon": int(horizon),
        "declared_variant": declared_variant,
        "observed_variant": observed_variant,
        "selector_stability": selector_stability,
        "selector_is_not_a_gate": True,
        "gates": gates,
        "risk_bound": None if risk_bound is None else float(risk_bound),
        "risk_threshold": float(risk_threshold),
        "fallback_required": not passed,
        "claim_boundary": (
            "local finite-horizon transfer-sufficiency release contract; not a global "
            "causal or universal-world-model certificate"
        ),
    }
