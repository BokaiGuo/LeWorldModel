"""Fail-closed contracts for the remaining learned-world-model gates.

These functions deliberately contain no simulator access and no model-specific
heuristics.  They make the missing evidence requirements executable:

* RGB-only successor verification cannot receive simulator state/contact;
* matched-quality transfer records must declare all nuisance factors;
* planning-risk prediction is evaluated on an independent held-out outcome;
* pose closure requires both pose/contact tolerances and a declared hold;
* incomplete inputs return ``blocked``/``insufficient_support`` rather than
  silently promoting a claim.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


FORBIDDEN_RGB_FIELDS = frozenset({
    "state", "goal_state", "sim_state", "contact", "contact_mode",
    "successor_state", "true_successor", "privileged_state",
})
REQUIRED_MATCHED_FACTORS = frozenset({
    "prediction_mse", "latent_dimension", "predictor_spectrum",
    "training_stage", "action_distribution",
})


def validate_rgb_only_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Validate the runtime input boundary for a pure-RGB policy."""

    if not isinstance(observation, dict):
        raise TypeError("RGB-only observation must be a mapping")
    forbidden = sorted(FORBIDDEN_RGB_FIELDS.intersection(observation))
    if forbidden:
        raise ValueError(f"RGB-only runtime received forbidden privileged fields: {forbidden}")
    if "rgb" not in observation and "pixels" not in observation:
        raise ValueError("RGB-only observation must provide rgb or pixels")
    pixels = observation.get("rgb", observation.get("pixels"))
    array = np.asarray(pixels)
    if array.ndim not in (2, 3, 4) or not np.isfinite(array.astype(np.float64, copy=False)).all():
        raise ValueError("RGB-only pixels must be finite and image-shaped")
    return {
        "status": "pass",
        "input_fields": sorted(str(key) for key in observation),
        "pixel_shape": [int(value) for value in array.shape],
        "privileged_fields": [],
    }


def rgb_successor_verification_gate(
    predicted_rgb: np.ndarray | None,
    observed_rgb: np.ndarray | None,
    *,
    max_normalized_error: float = 0.10,
) -> dict[str, Any]:
    """Compare a predicted RGB successor with the observed RGB successor.

    The gate is an image-space runtime consistency check.  It does not use
    simulator state and cannot certify physical correctness by itself.  A
    missing prediction or observation is explicitly ``blocked``.
    """

    if predicted_rgb is None or observed_rgb is None:
        return {
            "status": "blocked",
            "passed": False,
            "reason": "predicted_and_observed_rgb_are_both_required",
        }
    predicted = np.asarray(predicted_rgb, dtype=np.float64)
    observed = np.asarray(observed_rgb, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.size == 0:
        return {
            "status": "blocked",
            "passed": False,
            "reason": "rgb_shape_mismatch_or_empty",
        }
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        return {"status": "blocked", "passed": False, "reason": "non_finite_rgb"}
    scale = max(float(np.ptp(observed)), 1.0)
    error = float(np.mean(np.abs(predicted - observed)) / scale)
    passed = error <= float(max_normalized_error)
    return {
        "status": "pass" if passed else "fail",
        "passed": bool(passed),
        "normalized_l1_error": error,
        "threshold": float(max_normalized_error),
        "claim_boundary": "RGB successor consistency only; not physical causality or planning success",
    }


def validate_matched_quality_record(record: dict[str, Any]) -> dict[str, Any]:
    """Require the declared matching factors before transfer-risk analysis."""

    diagnostics = record.get("diagnostics", {})
    factors = record.get("model_factors", {})
    missing = sorted(
        field for field in REQUIRED_MATCHED_FACTORS
        if field not in diagnostics and field not in factors
    )
    planning = record.get("heldout", {})
    if "planning_loss" not in planning and "planning_risk" not in planning:
        missing.append("heldout_planning_risk")
    if missing:
        return {"status": "blocked", "missing": missing}
    return {
        "status": "pass",
        "factors": sorted(REQUIRED_MATCHED_FACTORS),
        "heldout_risk_field": "planning_loss" if "planning_loss" in planning else "planning_risk",
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def heldout_planning_risk_prediction(
    diagnostic_values: Iterable[float],
    planning_risk: Iterable[float],
    *,
    minimum_pairs: int = 4,
) -> dict[str, Any]:
    """Report a predeclared rank association without fabricating support."""

    metric = np.asarray(list(diagnostic_values), dtype=np.float64)
    risk = np.asarray(list(planning_risk), dtype=np.float64)
    if len(metric) != len(risk) or len(metric) < int(minimum_pairs):
        return {
            "status": "insufficient_support",
            "n_pairs": int(min(len(metric), len(risk))),
            "minimum_pairs": int(minimum_pairs),
        }
    if not np.isfinite(metric).all() or not np.isfinite(risk).all():
        return {"status": "blocked", "reason": "non_finite_metric_or_risk"}
    metric_rank = _rank(metric)
    risk_rank = _rank(risk)
    denominator = float(np.linalg.norm(metric_rank - metric_rank.mean()) * np.linalg.norm(risk_rank - risk_rank.mean()))
    correlation = None if denominator <= 1e-12 else float(
        np.dot(metric_rank - metric_rank.mean(), risk_rank - risk_rank.mean()) / denominator
    )
    return {
        "status": "pass" if correlation is not None else "blocked",
        "n_pairs": int(len(metric)),
        "spearman_metric_vs_risk": correlation,
        "risk_direction": "lower_is_better",
        "independent_heldout_outcome_required": True,
    }


def terminal_pose_closure_gate(
    rows: Iterable[dict[str, Any]],
    *,
    max_xy_error: float = 8.0,
    max_angle_error: float = 0.20,
    minimum_hold_steps: int = 2,
) -> dict[str, Any]:
    """Admit a learned pose-closure policy only under a declared hold test."""

    rows = list(rows)
    if not rows:
        return {"status": "insufficient_support", "passed": False, "n_cases": 0}
    missing = []
    accepted = []
    for index, row in enumerate(rows):
        required = {"final_xy", "final_angle", "hold_steps", "runtime_successor_verification"}
        absent = sorted(required.difference(row))
        if absent:
            missing.append({"case": index, "fields": absent})
            continue
        case_pass = (
            float(row["final_xy"]) <= float(max_xy_error)
            and float(row["final_angle"]) <= float(max_angle_error)
            and int(row["hold_steps"]) >= int(minimum_hold_steps)
            and bool(row["runtime_successor_verification"])
        )
        accepted.append(case_pass)
    if missing:
        return {"status": "blocked", "passed": False, "missing": missing}
    rate = float(np.mean(accepted)) if accepted else 0.0
    return {
        "status": "pass" if rate == 1.0 else "fail",
        "passed": bool(rate == 1.0),
        "n_cases": len(accepted),
        "strict_case_pass_rate": rate,
        "thresholds": {
            "max_xy_error": float(max_xy_error),
            "max_angle_error": float(max_angle_error),
            "minimum_hold_steps": int(minimum_hold_steps),
        },
        "runtime_successor_verification_required": True,
    }


def protocol_completion_matrix() -> dict[str, Any]:
    """Return the executable acceptance requirements for the five open gates."""

    return {
        "finite_horizon_transfer_bound": {"status": "implemented_contract", "entrypoint": "theory_spectral_transfer.finite_horizon_transfer"},
        "fixed_quality_synthetic_intervention": {"status": "implemented_entrypoint", "entrypoint": "run_fixed_quality_synthetic_intervention.py"},
        "adaptive_event_memory": {"status": "implemented", "entrypoint": "run_adaptive_time_frequency_memory_audit.py"},
        "pure_rgb_successor_verification": {"status": "implemented_contract", "entrypoint": "rgb_successor_verification_gate"},
        "matched_cross_architecture_environment_transfer": {"status": "implemented_contract", "entrypoint": "run_matched_learned_transfer_benchmark.py"},
        "heldout_planning_risk_prediction": {"status": "implemented_contract", "entrypoint": "heldout_planning_risk_prediction"},
        "stable_pose_closure": {"status": "implemented_contract", "entrypoint": "terminal_pose_closure_gate"},
        "pusht_mechanism_protocol": {"status": "implemented_contract", "entrypoint": "oral_evidence_protocol.pusht_mechanism_protocol"},
        "claim_boundary": "Contracts and entrypoints are not positive scientific evidence until their declared held-out runs pass.",
    }
