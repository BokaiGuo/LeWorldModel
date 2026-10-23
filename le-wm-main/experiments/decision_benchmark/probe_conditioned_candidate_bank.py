"""Construct a verified local PointMaze candidate bank after RGB probes.

This module encodes the missing intervention exposed by the fixed-bank active
probe panels.  A local RGB action--effect chart must not be asked to rank a
global bank whose actions lie outside the actions that identified it.  Instead,
it constructs a small candidate set *inside* the convex hull of the observed
probe actions.  The construction deliberately consumes only:

* executed probe actions;
* their RGB-decoded observed effects; and
* the current RGB-derived direction to the declared goal.

It has no environment-state, privileged oracle, future successor, or task
outcome input.  Downstream experiments must still establish on calibration
roots that the resulting bank has sufficient real oracle headroom before any
held-out outcome is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Callable

import numpy as np


SCHEMA = "pointmaze.probe_conditioned_candidate_bank.v1"


def _as_matrix(value: np.ndarray, *, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a non-empty finite [N,2] matrix")
    return matrix


def _canonical_digest(actions: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(actions, dtype=np.float64))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _cross(origin: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    left_delta = left - origin
    right_delta = right - origin
    return float(left_delta[0] * right_delta[1] - left_delta[1] * right_delta[0])


def action_hull_support_mask(
    probe_actions: np.ndarray,
    candidates: np.ndarray,
    *,
    tolerance: float = 1.0e-8,
) -> np.ndarray:
    """Return whether each candidate is inside the observed action hull."""

    probes = _as_matrix(probe_actions, label="probe_actions")
    values = _as_matrix(candidates, label="candidates")
    points = np.unique(probes, axis=0)
    if len(points) < 3:
        return np.zeros(len(values), dtype=bool)
    ordered = sorted(tuple(point) for point in points)
    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(
            np.asarray(lower[-2]), np.asarray(lower[-1]), np.asarray(point)
        ) <= tolerance:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(
            np.asarray(upper[-2]), np.asarray(upper[-1]), np.asarray(point)
        ) <= tolerance:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        return np.zeros(len(values), dtype=bool)
    return np.asarray([
        all(
            _cross(hull[index], hull[(index + 1) % len(hull)], candidate) >= -tolerance
            for index in range(len(hull))
        )
        for candidate in values
    ], dtype=bool)


def _support_points(probe_actions: np.ndarray) -> np.ndarray:
    """Build deterministic, strictly convex-combination candidate proposals."""

    probes = _as_matrix(probe_actions, label="probe_actions")
    weights: list[np.ndarray] = []
    count = len(probes)
    for index in range(count):
        value = np.zeros(count, dtype=np.float64)
        value[index] = 1.0
        weights.append(value)
    for left in range(count):
        for right in range(left + 1, count):
            for fraction in (0.25, 0.5, 0.75):
                value = np.zeros(count, dtype=np.float64)
                value[left], value[right] = 1.0 - fraction, fraction
                weights.append(value)
    centroid = np.full(count, 1.0 / count, dtype=np.float64)
    for index in range(count):
        vertex = np.zeros(count, dtype=np.float64)
        vertex[index] = 1.0
        for fraction in (0.25, 0.5, 0.75):
            weights.append((1.0 - fraction) * centroid + fraction * vertex)
    points = np.asarray(weights, dtype=np.float64) @ probes
    # Numeric duplicates are common when probes are symmetric.  Keep first
    # occurrence so ties are deterministic and traceable.
    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.allclose(point, prior, atol=1.0e-12, rtol=0.0) for prior in unique):
            unique.append(point)
    return np.asarray(unique, dtype=np.float64)


@dataclass(frozen=True)
class ProbeConditionedCandidateBank:
    """A local, RGB-only action bank plus its pre-outcome construction trace."""

    actions: np.ndarray
    predicted_effects: np.ndarray
    diagnostics: dict[str, float | int | str | bool]
    construction_inputs: str = (
        "RGB_observed_probe_actions_RGB_observed_effects_RGB_goal_direction_only"
    )

    def trace_record(self) -> dict[str, object]:
        """Return a JSON-safe contract suitable for immutable planning traces."""

        return {
            "schema": SCHEMA,
            "candidate_bank_digest": _canonical_digest(self.actions),
            "candidate_count": int(len(self.actions)),
            "candidate_actions": self.actions.tolist(),
            "predicted_effects": self.predicted_effects.tolist(),
            "construction_inputs": self.construction_inputs,
            "privileged_state_or_outcome_used": False,
            "diagnostics": self.diagnostics,
        }

    def trace_digest(self) -> str:
        encoded = json.dumps(self.trace_record(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def build_probe_conditioned_candidate_bank(
    probe_actions: np.ndarray,
    observed_effects: np.ndarray,
    goal_direction: np.ndarray,
    *,
    candidate_count: int = 16,
    ridge: float = 5.0e-2,
    min_predicted_progress: float = 1.0e-8,
) -> ProbeConditionedCandidateBank | None:
    """Build a support-preserving local candidate bank or return ``None``.

    ``None`` is the intended fail-closed result: insufficient rank, no
    goal-progressing action in the observed hull, or fewer than four distinct
    usable candidates means the caller must retain its declared fallback.
    """

    actions = _as_matrix(probe_actions, label="probe_actions")
    effects = _as_matrix(observed_effects, label="observed_effects")
    if actions.shape != effects.shape:
        raise ValueError("probe_actions and observed_effects must have identical shape")
    if int(candidate_count) < 4:
        raise ValueError("candidate_count must be at least four")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be positive and finite")
    direction = np.asarray(goal_direction, dtype=np.float64).reshape(-1)
    if direction.shape != (2,) or not np.isfinite(direction).all():
        raise ValueError("goal_direction must be finite [2]")
    direction_norm = float(np.linalg.norm(direction))
    # Reaching the current RGB waypoint during the charged probe schedule is
    # a normal closed-loop event, not malformed input.  A local action chart
    # has no defined *forward* direction in that case, so decline to replace
    # the fallback bank and let the ordinary planner re-observe/replan.
    if direction_norm <= 1.0e-12:
        return None
    direction = direction / direction_norm
    design = np.concatenate([np.ones((len(actions), 1)), actions], axis=1)
    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        return None
    weights = np.linalg.solve(
        design.T @ design + float(ridge) * np.eye(design.shape[1]),
        design.T @ effects,
    )
    proposals = _support_points(actions)
    proposal_effects = np.concatenate([np.ones((len(proposals), 1)), proposals], axis=1) @ weights
    predicted_progress = proposal_effects @ direction
    supported = action_hull_support_mask(actions, proposals)
    usable = supported & (predicted_progress > float(min_predicted_progress))
    if int(np.sum(usable)) < 4:
        return None
    usable_indices = np.flatnonzero(usable)
    # Favor predicted progress while retaining candidate diversity: ties are
    # settled by action norm then original deterministic proposal order.
    order = sorted(
        usable_indices.tolist(),
        key=lambda index: (
            -float(predicted_progress[index]),
            float(np.linalg.norm(proposals[index])),
            int(index),
        ),
    )
    selected = np.asarray(order[: int(candidate_count)], dtype=np.int64)
    chosen_actions = proposals[selected]
    chosen_effects = proposal_effects[selected]
    chosen_progress = predicted_progress[selected]
    diagnostics: dict[str, float | int | str | bool] = {
        "probe_count": int(len(actions)),
        "affine_design_rank": rank,
        "affine_feature_count": int(design.shape[1]),
        "proposal_count_before_goal_filter": int(len(proposals)),
        "candidate_count_after_goal_filter": int(len(usable_indices)),
        "candidate_count": int(len(chosen_actions)),
        "all_candidates_in_observed_action_hull": bool(np.all(action_hull_support_mask(actions, chosen_actions))),
        "minimum_predicted_goal_progress": float(np.min(chosen_progress)),
        "predicted_best_second_goal_progress_gap": (
            float(chosen_progress[0] - chosen_progress[1]) if len(chosen_progress) > 1 else 0.0
        ),
        "construction_status": "constructed_pre_outcome",
    }
    return ProbeConditionedCandidateBank(chosen_actions, chosen_effects, diagnostics)


def build_prior_anchored_residual_probe_conditioned_candidate_bank(
    probe_actions: np.ndarray,
    observed_residual_effects: np.ndarray,
    goal_direction: np.ndarray,
    *,
    baseline_effects: Callable[[np.ndarray], np.ndarray],
    candidate_count: int = 16,
    ridge: float = 1.0e-6,
    min_predicted_progress: float = 1.0e-8,
) -> ProbeConditionedCandidateBank | None:
    """Construct a local bank from a prior-anchored residual Jacobian.

    The affine V11 chart fitted raw sequential displacements.  In an inertial
    maze that aliases action sensitivity with the position/velocity drift that
    occurred between probes.  This builder instead receives RGB-observed
    effects *after* subtraction of a calibration-only dynamics prediction at
    each probe's own velocity.  Its zero-action intercept is therefore fixed
    by that dynamics prior rather than estimated from a changing trajectory.

    ``baseline_effects`` is evaluated only for candidate actions at the
    current RGB-derived velocity.  It is an explicit dependency so this
    module remains free of runtime state and simulator access.
    """

    actions = _as_matrix(probe_actions, label="probe_actions")
    residuals = _as_matrix(observed_residual_effects, label="observed_residual_effects")
    if actions.shape != residuals.shape:
        raise ValueError("probe_actions and observed_residual_effects must have identical shape")
    if int(candidate_count) < 4:
        raise ValueError("candidate_count must be at least four")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be positive and finite")
    direction = np.asarray(goal_direction, dtype=np.float64).reshape(-1)
    if direction.shape != (2,) or not np.isfinite(direction).all():
        raise ValueError("goal_direction must be finite [2]")
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        return None
    direction = direction / norm
    # A residual Jacobian is anchored at a zero-action prior; two
    # non-collinear information actions identify its two action coefficients.
    rank = int(np.linalg.matrix_rank(actions))
    if rank < 2:
        return None
    weights = np.linalg.solve(
        actions.T @ actions + float(ridge) * np.eye(2, dtype=np.float64),
        actions.T @ residuals,
    )
    anchored_actions = np.concatenate([np.zeros((1, 2), dtype=np.float64), actions], axis=0)
    proposals = _support_points(anchored_actions)
    baseline = _as_matrix(baseline_effects(proposals), label="baseline_effects")
    if baseline.shape != proposals.shape:
        raise ValueError("baseline_effects must return a finite [N,2] matrix")
    proposal_effects = baseline + proposals @ weights
    progress = proposal_effects @ direction
    supported = action_hull_support_mask(anchored_actions, proposals)
    usable = supported & (progress > float(min_predicted_progress))
    if int(np.sum(usable)) < 4:
        return None
    usable_indices = np.flatnonzero(usable)
    order = sorted(
        usable_indices.tolist(),
        key=lambda index: (-float(progress[index]), float(np.linalg.norm(proposals[index])), int(index)),
    )
    selected = np.asarray(order[: int(candidate_count)], dtype=np.int64)
    chosen_actions = proposals[selected]
    chosen_effects = proposal_effects[selected]
    chosen_progress = progress[selected]
    diagnostics: dict[str, float | int | str | bool] = {
        "probe_count": int(len(actions)),
        "residual_jacobian_design_rank": rank,
        "residual_jacobian_feature_count": 2,
        "zero_action_prior_anchor": True,
        "proposal_count_before_goal_filter": int(len(proposals)),
        "candidate_count_after_goal_filter": int(len(usable_indices)),
        "candidate_count": int(len(chosen_actions)),
        "all_candidates_in_observed_action_hull": bool(
            np.all(action_hull_support_mask(anchored_actions, chosen_actions))
        ),
        "minimum_predicted_goal_progress": float(np.min(chosen_progress)),
        "predicted_best_second_goal_progress_gap": (
            float(chosen_progress[0] - chosen_progress[1]) if len(chosen_progress) > 1 else 0.0
        ),
        "construction_status": "constructed_pre_outcome",
    }
    return ProbeConditionedCandidateBank(
        chosen_actions,
        chosen_effects,
        diagnostics,
        construction_inputs=(
            "RGB_observed_probe_actions_RGB_observed_effect_residuals_"
            "RGB_goal_direction_calibration_dynamics_only"
        ),
    )
