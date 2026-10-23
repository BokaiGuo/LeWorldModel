"""Calibration-only support gate for prospective TwoRoom effect repair.

The gate answers a deliberately narrow C2 question: after the full effect
adapter has been fitted, can train-feature support and calibration-only paired
regret identify roots on which the adapter should be released instead of an
action-only fallback?  It is not a conformal guarantee and it never reads an
outcome root while choosing the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SCHEMA = "tworoom_support_gated_repair.v1"


@dataclass(frozen=True)
class FrozenSupportGate:
    mean: np.ndarray
    scale: np.ndarray
    reference: np.ndarray
    threshold: float
    authorized: bool

    def distance(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError("features do not match frozen support gate")
        standardized = (values - self.mean) / self.scale
        pairwise = np.sqrt(np.mean((standardized[:, None, :] - self.reference[None, :, :]) ** 2, axis=-1))
        return np.min(pairwise, axis=1)

    def release(self, features: np.ndarray) -> bool:
        # A root is released only when every candidate is within the calibrated
        # feature-support envelope.  This avoids certifying a chosen candidate
        # merely because a different candidate happened to have support.
        return bool(self.authorized and np.max(self.distance(features)) <= self.threshold)


def _fit_support(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(train_features, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("need at least two finite train feature rows")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    return mean, scale, (values - mean) / scale


def _root_distances(gate: FrozenSupportGate, features: np.ndarray, roots: np.ndarray) -> dict[int, float]:
    values = np.asarray(features, dtype=np.float64)
    root_ids = np.asarray(roots, dtype=np.int64)
    if values.ndim != 2 or root_ids.shape != (len(values),):
        raise ValueError("feature/root arrays are malformed")
    distances = gate.distance(values)
    return {
        int(root): float(np.max(distances[root_ids == root]))
        for root in sorted(set(root_ids.tolist()))
    }


def calibrate(
    *,
    train_features: np.ndarray,
    calibration_features: np.ndarray,
    calibration_roots: np.ndarray,
    adapter_regret: dict[int, float],
    fallback_regret: dict[int, float],
    minimum_release_coverage: float = 0.10,
) -> tuple[FrozenSupportGate, dict[str, Any]]:
    """Freeze a release threshold without reading any outcome-root labels."""

    if not 0.0 < minimum_release_coverage <= 1.0:
        raise ValueError("minimum_release_coverage must lie in (0, 1]")
    mean, scale, reference = _fit_support(train_features)
    provisional = FrozenSupportGate(mean, scale, reference, threshold=float("-inf"), authorized=False)
    support = _root_distances(provisional, calibration_features, calibration_roots)
    roots = sorted(support)
    if set(adapter_regret) != set(roots) or set(fallback_regret) != set(roots):
        raise ValueError("calibration regret keys must exactly match calibration roots")
    gains = {root: float(fallback_regret[root] - adapter_regret[root]) for root in roots}
    # Candidate thresholds are fully determined by pre-outcome support values.
    candidates = [float("-inf"), *sorted(set(support.values()))]
    scored: list[dict[str, float]] = []
    for threshold in candidates:
        released = [root for root in roots if support[root] <= threshold]
        utility = np.asarray([gains[root] if root in released else 0.0 for root in roots], dtype=np.float64)
        scored.append({
            "threshold": float(threshold),
            "coverage": float(len(released) / len(roots)),
            "mean_gain_over_fallback": float(utility.mean()),
            "released_root_count": float(len(released)),
        })
    eligible = [
        value for value in scored
        if value["coverage"] >= minimum_release_coverage and value["mean_gain_over_fallback"] > 0.0
    ]
    chosen = max(eligible, key=lambda value: (value["mean_gain_over_fallback"], value["coverage"], -value["threshold"])) if eligible else scored[0]
    authorized = bool(eligible)
    gate = FrozenSupportGate(mean, scale, reference, threshold=float(chosen["threshold"]), authorized=authorized)
    return gate, {
        "schema": SCHEMA,
        "status": "authorized_calibration_support_gate" if authorized else "rejected_calibration_support_gate",
        "outcome_labels_read": False,
        "minimum_release_coverage": float(minimum_release_coverage),
        "calibration_root_count": len(roots),
        "calibration_support": {str(root): support[root] for root in roots},
        "calibration_gain_over_fallback": {str(root): gains[root] for root in roots},
        "candidate_thresholds": scored,
        "selected": chosen,
        "claim_boundary": (
            "Calibration-only release gate. It requires a separate root-disjoint outcome evaluation "
            "and is not a coverage or safety certificate."
        ),
    }


def choose_scores(
    gate: FrozenSupportGate,
    root_features: np.ndarray,
    adapter_scores: np.ndarray,
    fallback_scores: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Return full-adapter scores only for a prospectively released root."""

    adapter = np.asarray(adapter_scores, dtype=np.float64)
    fallback = np.asarray(fallback_scores, dtype=np.float64)
    if adapter.shape != fallback.shape or adapter.ndim != 1 or len(adapter) != len(root_features):
        raise ValueError("candidate score/features shapes do not align")
    released = gate.release(root_features)
    return (adapter.copy() if released else fallback.copy()), released
