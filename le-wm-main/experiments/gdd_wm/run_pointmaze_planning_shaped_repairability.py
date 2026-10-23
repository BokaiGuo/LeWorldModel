"""Prospective root-level routing for a planning-shaped repair.

The repair is deliberately narrow: choose, before outcome execution, whether a
cost-only candidate scorer or a planning-shaped scorer should control one root.
It is not a GDD, D-JEPA, PlanWAM, or online-deployment claim.  Its purpose is
to test the missing link exposed by the planning-shaped controls: can lawful
pre-outcome evidence predict when the extra future-effect supervision lowers
real candidate-set regret?

Roles are root-disjoint.  Cross-fitted train roots create repair labels;
calibration roots freeze only the release threshold; outcome candidates are
executed only after both scorers and the threshold are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.gdd_wm.run_pointmaze_planning_shaped_control import (
    Objective,
    _action_summary,
    _calibration_terminal,
    _cost,
    _feature,
    _feature_ordered_action_sequence,
    _fresh_terminal,
    _group_rows,
    _mean_predictor,
    _sha256,
    _terminal_from_ref,
    _fit,
    action_representation_input_fields,
)
from experiments.gpu_preflight import require_cuda_allocation


SCHEMA = "pointmaze_planning_shaped_repairability.v0.2"
CONTROL_LABEL = "crossfit_visual_history_action_repairability_router"
COST_ONLY = Objective("cost_regression", effect_weight=0.0, rank_weight=0.0)
PLANNING_SHAPED = Objective("planning_shaped", effect_weight=0.25, rank_weight=0.25)
SUMMARY_ACTION_REPRESENTATION = "summary_mean_std_first_last"
ORDERED_ACTION_REPRESENTATION = "full_ordered_20x2"
V7_PREFLIGHT_SCHEMA = "pointmaze_planning_shaped_advantage_v7_static_preflight.v1"
V9_PREFLIGHT_SCHEMA = "pointmaze_planning_shaped_opportunity_router_v9_static_preflight.v1"
V7_ROUTER_KIND = "advantage_ridge"
ADVANTAGE_BOOTSTRAP_LOWER_ROUTER = "advantage_bootstrap_lower"
ADVANTAGE_ROUTER_KINDS = frozenset({V7_ROUTER_KIND, ADVANTAGE_BOOTSTRAP_LOWER_ROUTER})
BOOTSTRAP_ADVANTAGE_MEMBER_COUNT = 64
BOOTSTRAP_ADVANTAGE_LOWER_QUANTILE = 0.10
V7_ACTION_REPRESENTATION = "full_ordered_20x2"
SUPPORTED_SHARED_ADVANTAGE_CANDIDATE_DESIGNS = frozenset({
    "goal_balanced_interaction_pairs",
    "tail_robust_goal_balanced_interaction_pairs",
    "expanded_tail_robust_goal_balanced_interaction_pairs",
    "ultra_tail_robust_goal_balanced_interaction_pairs",
    "directional_tail_robust_goal_balanced_interaction_pairs",
})
V7_ROUTER_FEATURE_DIMENSION = 95
V7_ROUTER_REGULARIZATION_SELECTION = "train_root_only_four_fold_blocked_advantage_mse"
V7_POST_EXECUTION_ALIASING_CONTRACT = {
    "representation_key": "source_history_model",
    "radius_quantile": 0.25,
    "contrast_gap_threshold": 0.05,
    "max_reported_pairs": 64,
}
ADVANTAGE_RIDGE_L2_GRID = (0.25, 1.0, 4.0, 16.0, 64.0)
V9_ROUTER_FEATURE_DIMENSION = 15
V9_ROUTER_FEATURE_CONTRACT = "opportunity_compact_v1_predicted_score_action_delta"
V9_MIN_ROUTER_ROOT_SURPLUS = 16
# The opportunity stratum, not the full train split, is the fitting sample.
# It must therefore exceed the compact router dimension by the declared
# surplus.  The old 24-root threshold silently permitted a 95-feature ridge
# fit and was a direct source of high-variance routing.
V9_MIN_TRAIN_OPPORTUNITY_ROOTS = V9_ROUTER_FEATURE_DIMENSION + V9_MIN_ROUTER_ROOT_SURPLUS
V9_MIN_CALIBRATION_OPPORTUNITY_ROOTS_PER_ROLE = 8


def _feature_for_action_representation(
    action_representation: str,
) -> Callable[[dict[str, Any]], np.ndarray]:
    """Bind the declared action schema to the actual scorer input.

    V7 deliberately contains candidate pairs that collide under the legacy
    summary.  Merely recording ``full_ordered_20x2`` in a contract while
    continuing to call ``_feature`` would make those actions information-
    theoretically indistinguishable and turn a negative result into a setup
    artifact.  Older panels retain their summary default for reproducibility.
    """

    if action_representation == SUMMARY_ACTION_REPRESENTATION:
        return _feature
    if action_representation == ORDERED_ACTION_REPRESENTATION:
        return _feature_ordered_action_sequence
    raise ValueError(
        "action representation must be summary_mean_std_first_last or full_ordered_20x2"
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _root_fold(root: str) -> int:
    return int(hashlib.sha256(root.encode("utf-8")).hexdigest()[:8], 16) % 4


def _fit_ensemble(
    rows: list[dict[str, Any]], terminal: np.ndarray, objective: Objective, *,
    seeds: tuple[int, ...], width: int, steps: int, device: str,
    feature_fn: Callable[[dict[str, Any]], np.ndarray] = _feature,
    action_representation: str = "summary_mean_std_first_last",
) -> tuple[dict[str, Any], Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]:
    if len(rows) != len(terminal):
        raise ValueError("rows and terminal targets differ in length")
    features = np.stack([feature_fn(row) for row in rows])
    source = np.asarray([row["source_state_raw"] for row in rows], dtype=np.float64)
    effect, cost, groups = terminal - source, _cost(rows, terminal), _group_rows(rows)
    members = [
        _fit(features, effect, cost, groups, objective, width=width, steps=steps, device=device, seed=seed)
        for seed in seeds
    ]
    contract = {
        **members[0][0], "initialization_seeds": list(seeds), "ensemble_size": len(seeds),
        "objective": objective.name, "action_representation": str(action_representation),
        "input_fields": action_representation_input_fields(action_representation),
    }
    return contract, _mean_predictor([member[1] for member in members])


def _choice(group: list[dict[str, Any]], prediction: np.ndarray) -> int:
    return min(range(len(group)), key=lambda index: (float(prediction[index]), str(group[index]["candidate_id"])))


def _top_margin(prediction: np.ndarray) -> float:
    ordered = np.sort(np.asarray(prediction, dtype=np.float64))
    return float(ordered[1] - ordered[0]) if len(ordered) > 1 else 0.0


def _root_features(
    group: list[dict[str, Any]], cost_prediction: np.ndarray, planning_prediction: np.ndarray,
    cost_effect: np.ndarray | None = None, planning_effect: np.ndarray | None = None,
    *, action_representation: str = SUMMARY_ACTION_REPRESENTATION,
) -> np.ndarray:
    """Lawful, decision-time features; raw simulator state is intentionally absent.

    Optional effects contribute only frozen predicted-disagreement geometry, not
    terminal labels or privileged state.
    """

    cost_pick, plan_pick = _choice(group, cost_prediction), _choice(group, planning_prediction)
    history = np.asarray(group[0]["source_history_model"], dtype=np.float64).reshape(-1)
    latent = np.asarray(group[0]["source_latent_model"], dtype=np.float64).reshape(-1)
    goal = np.asarray(group[0]["goal_raw"], dtype=np.float64).reshape(-1)
    if not len(latent) or history.size % latent.size or goal.shape != (2,):
        raise ValueError("visual history, source latent, and declared goal are required")
    first, last = history[: latent.size], history[-latent.size :]
    if action_representation == SUMMARY_ACTION_REPRESENTATION:
        action_cost = _action_summary(group[cost_pick])
        action_plan = _action_summary(group[plan_pick])
        action_values = [float(np.linalg.norm(action_cost - action_plan))]
    elif action_representation == ORDERED_ACTION_REPRESENTATION:
        action_cost = np.asarray(group[cost_pick]["action_sequence"], dtype=np.float64)
        action_plan = np.asarray(group[plan_pick]["action_sequence"], dtype=np.float64)
        if (
            action_cost.shape != (20, 2) or action_plan.shape != (20, 2)
            or not np.isfinite(action_cost).all() or not np.isfinite(action_plan).all()
        ):
            raise ValueError("full_ordered_20x2 router inputs require finite [20,2] actions")
        # Preserve the individual ordered sequences, not merely their norm:
        # two interior pulse permutations can have identical summary and
        # distance while requiring different decisions.
        action_values = [
            float(np.linalg.norm(action_cost - action_plan)),
            *action_cost.reshape(-1).tolist(),
            *action_plan.reshape(-1).tolist(),
        ]
    else:
        raise ValueError(
            "action representation must be summary_mean_std_first_last or full_ordered_20x2"
        )
    values = [
        float(cost_pick != plan_pick),
        _top_margin(cost_prediction), _top_margin(planning_prediction),
        float(np.std(cost_prediction)), float(np.std(planning_prediction)),
        float(cost_prediction[plan_pick] - cost_prediction[cost_pick]),
        float(planning_prediction[cost_pick] - planning_prediction[plan_pick]),
        float(np.linalg.norm(last - first)), float(np.linalg.norm(last)),
        float(np.linalg.norm(goal)), *action_values,
    ]
    if (cost_effect is None) != (planning_effect is None):
        raise ValueError("cost and planning effect predictions must be supplied together")
    if cost_effect is not None and planning_effect is not None:
        cost_effect = np.asarray(cost_effect, dtype=np.float64)
        planning_effect = np.asarray(planning_effect, dtype=np.float64)
        if cost_effect.shape != (len(group), 4) or planning_effect.shape != (len(group), 4):
            raise ValueError("effect predictions must be [candidate_count, 4]")
        if not np.isfinite(cost_effect).all() or not np.isfinite(planning_effect).all():
            raise ValueError("effect predictions must be finite")
        values.extend((
            float(np.linalg.norm(cost_effect - planning_effect, axis=1).mean()),
            float(np.linalg.norm(cost_effect - cost_effect.mean(axis=0), axis=1).mean()),
            float(np.linalg.norm(planning_effect - planning_effect.mean(axis=0), axis=1).mean()),
            float(np.linalg.norm(planning_effect[plan_pick] - cost_effect[cost_pick])),
        ))
    return np.asarray(values, dtype=np.float64)


def _opportunity_compact_features(
    group: list[dict[str, Any]], cost_prediction: np.ndarray, planning_prediction: np.ndarray,
) -> np.ndarray:
    """Low-variance router features for V9's scorer-disagreement stratum.

    The ordered action sequence remains mandatory for both candidate scorers.
    The router has a different job: decide whether to release the planning arm
    *after* the scorers have emitted their candidate-specific predictions.
    Feeding 80 raw action coordinates to a ridge fit on roughly thirty
    disagreement roots made the original V9 proposal underdetermined.  This
    frozen 15-dimensional map retains the predictive margins and the selected
    actions' signed/temporal contrast without claiming to re-encode the full
    candidate sequence.
    """

    cost_pick, plan_pick = _choice(group, cost_prediction), _choice(group, planning_prediction)
    history = np.asarray(group[0]["source_history_model"], dtype=np.float64).reshape(-1)
    latent = np.asarray(group[0]["source_latent_model"], dtype=np.float64).reshape(-1)
    goal = np.asarray(group[0]["goal_raw"], dtype=np.float64).reshape(-1)
    if not len(latent) or history.size % latent.size or goal.shape != (2,):
        raise ValueError("visual history, source latent, and declared goal are required")
    cost_action = np.asarray(group[cost_pick].get("action_sequence"), dtype=np.float64)
    plan_action = np.asarray(group[plan_pick].get("action_sequence"), dtype=np.float64)
    if (
        cost_action.shape != (20, 2) or plan_action.shape != (20, 2)
        or not np.isfinite(cost_action).all() or not np.isfinite(plan_action).all()
    ):
        raise ValueError("V9 compact opportunity router requires finite [20,2] actions")
    first, last = history[: latent.size], history[-latent.size :]
    action_delta = cost_action - plan_action
    return np.asarray([
        float(cost_pick != plan_pick),
        _top_margin(cost_prediction), _top_margin(planning_prediction),
        float(np.std(cost_prediction)), float(np.std(planning_prediction)),
        float(cost_prediction[plan_pick] - cost_prediction[cost_pick]),
        float(planning_prediction[cost_pick] - planning_prediction[plan_pick]),
        float(np.linalg.norm(last - first)), float(np.linalg.norm(last)), float(np.linalg.norm(goal)),
        *action_delta.mean(axis=0).tolist(),
        float(np.square(cost_action).sum() - np.square(plan_action).sum()),
        float(np.linalg.norm(action_delta)),
        float(np.linalg.norm(action_delta[:10].mean(axis=0) - action_delta[10:].mean(axis=0))),
    ], dtype=np.float64)


def _regret(cost: np.ndarray, pick: int) -> float:
    return float(cost[pick] - np.min(cost))


def _fit_logistic(features: np.ndarray, labels: np.ndarray, *, l2: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Deterministic, standardized L2 logistic router with a legal constant fallback."""

    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    if len(np.unique(labels)) < 2:
        probability = float(np.mean(labels))
        return lambda values: np.full(len(values), probability, dtype=np.float64)
    x = (features - mean) / scale
    design = np.c_[np.ones(len(x)), x]
    beta = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(800):
        logits = np.clip(design @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probability - labels) / len(labels)
        gradient[1:] += l2 * beta[1:] / len(labels)
        hessian = design.T @ (design * (probability * (1.0 - probability))[:, None]) / len(labels)
        hessian[1:, 1:] += np.eye(len(beta) - 1) * l2 / len(labels)
        try:
            step = np.linalg.solve(hessian + np.eye(len(beta)) * 1.0e-8, gradient)
        except np.linalg.LinAlgError:
            step = gradient
        beta -= np.clip(step, -1.0, 1.0)
        if float(np.max(np.abs(step))) < 1.0e-8:
            break

    def predict(values: np.ndarray) -> np.ndarray:
        design_eval = np.c_[np.ones(len(values)), (values - mean) / scale]
        return 1.0 / (1.0 + np.exp(-np.clip(design_eval @ beta, -30.0, 30.0)))

    return predict


def _fit_ridge_advantage(features: np.ndarray, advantage: np.ndarray, *, l2: float = 2.0) -> Callable[[np.ndarray], np.ndarray]:
    """Predict cross-fitted planning-minus-cost *utility*, not merely its sign.

    ``advantage`` is ``cost_regret - planning_regret``: positive values mean
    that releasing the planning-shaped scorer lowers oracle candidate-bank
    regret.  This is deliberately a small, deterministic linear router.  It
    does not inspect calibration/outcome labels and avoids converting a rare,
    high-value repair event into the same binary target as a negligible tie.
    """

    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(advantage, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or len(values) != len(target) or not len(values):
        raise ValueError("advantage router needs non-empty aligned feature and target arrays")
    if not np.isfinite(values).all() or not np.isfinite(target).all() or l2 < 0.0:
        raise ValueError("advantage router requires finite values and non-negative L2")
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    design = np.c_[np.ones(len(values)), (values - mean) / scale]
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    penalty[0, 0] = 0.0
    try:
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(design.T @ design + penalty) @ design.T @ target

    def predict(evaluation: np.ndarray) -> np.ndarray:
        checked = np.asarray(evaluation, dtype=np.float64)
        if checked.ndim != 2 or checked.shape[1] != values.shape[1] or not np.isfinite(checked).all():
            raise ValueError("advantage-router evaluation features must be finite and feature-compatible")
        return np.asarray(np.c_[np.ones(len(checked)), (checked - mean) / scale] @ beta, dtype=np.float64)

    return predict


def _router_fold(root: str) -> int:
    """Independent root fold for router regularization selection.

    Scorer cross-fitting already prevents a root's terminal from training its
    own scorer.  This separate namespace prevents the advantage router from
    choosing its regularization on the same root labels it is evaluated on.
    """

    return int(hashlib.sha256(f"advantage-router/{root}".encode("utf-8")).hexdigest()[:8], 16) % 4


def _fit_selected_ridge_advantage(
    roots: list[str], features: np.ndarray, advantage: np.ndarray,
    *, l2_grid: tuple[float, ...] = ADVANTAGE_RIDGE_L2_GRID,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Select ridge regularization with train-root-only blocked validation.

    V7's full ordered action representation intentionally has many more
    features than the old action summary.  Keeping the historical fixed ridge
    penalty would turn a null routing result into an arbitrary regularization
    choice.  This selector sees only cross-fitted train-root advantages and
    retains an explicit diagnostic record for later calibration/outcome audit.
    """

    ordered_roots = [str(root) for root in roots]
    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(advantage, dtype=np.float64).reshape(-1)
    if len(ordered_roots) != len(set(ordered_roots)):
        raise ValueError("advantage router roots must be unique")
    if values.ndim != 2 or len(values) != len(ordered_roots) or len(target) != len(ordered_roots):
        raise ValueError("advantage router roots, features, and targets must align")
    if len(ordered_roots) < 8 or not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError("advantage router needs at least eight finite train-root labels")
    grid = tuple(float(value) for value in l2_grid)
    if not grid or any(not np.isfinite(value) or value < 0.0 for value in grid):
        raise ValueError("advantage ridge grid must contain finite non-negative penalties")
    folds = np.asarray([_router_fold(root) for root in ordered_roots], dtype=np.int64)
    if len(np.unique(folds)) < 2:
        raise ValueError("advantage router regularization selection needs at least two populated folds")

    candidates: list[dict[str, Any]] = []
    for l2 in grid:
        prediction = np.full(len(target), np.nan, dtype=np.float64)
        for fold in sorted(set(int(value) for value in folds)):
            held = folds == fold
            fit = ~held
            if not held.any() or not fit.any():
                raise ValueError("advantage router regularization fold is empty")
            model = _fit_ridge_advantage(values[fit], target[fit], l2=l2)
            prediction[held] = model(values[held])
        residual = prediction - target
        candidates.append({
            "l2": l2,
            "root_blocked_mse": float(np.mean(np.square(residual))),
            "root_blocked_mae": float(np.mean(np.abs(residual))),
            "root_blocked_sign_accuracy": float(np.mean((prediction >= 0.0) == (target >= 0.0))),
            "root_blocked_predicted_positive_coverage": float(np.mean(prediction >= 0.0)),
        })
    # Prefer the most regularized model only when predictive error is exactly
    # tied; that deterministic tie-break reduces variance without reading
    # calibration or outcome labels.
    selected = min(candidates, key=lambda row: (row["root_blocked_mse"], row["root_blocked_mae"], -row["l2"]))
    return _fit_ridge_advantage(values, target, l2=float(selected["l2"])), {
        "selection_protocol": "train_root_only_four_fold_blocked_advantage_mse",
        "root_count": len(ordered_roots), "feature_dimension": int(values.shape[1]),
        "fold_counts": {str(fold): int(np.sum(folds == fold)) for fold in sorted(set(int(value) for value in folds))},
        "candidate_l2": list(grid), "candidates": candidates, "selected": selected,
        "claim_boundary": "Train-root-only router regularization diagnostics. They do not authorize a repair route; independent calibration still freezes and certifies release.",
    }


def _fit_bootstrap_lower_advantage(
    roots: list[str], features: np.ndarray, advantage: np.ndarray, *, seed: int,
    member_count: int = BOOTSTRAP_ADVANTAGE_MEMBER_COUNT,
    lower_quantile: float = BOOTSTRAP_ADVANTAGE_LOWER_QUANTILE,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Return a conservative train-root-bootstrap advantage score.

    V9 exposed a concrete failure: a point estimate that looked ordered on
    calibration reversed on untouched roots.  The ensemble is fixed from
    cross-fitted *train-root* advantages only, then exports its lower
    prediction quantile.  Calibration still selects the sole release
    threshold; this is an uncertainty-aware score, not a conditional
    guarantee or a retrospective V9 adjustment.
    """

    ordered_roots = [str(root) for root in roots]
    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(advantage, dtype=np.float64).reshape(-1)
    if (
        values.ndim != 2 or len(values) != len(ordered_roots) or len(target) != len(ordered_roots)
        or len(ordered_roots) != len(set(ordered_roots))
    ):
        raise ValueError("bootstrap advantage router requires unique, aligned train roots")
    if member_count < 2 or not 0.0 < lower_quantile < 0.5:
        raise ValueError("bootstrap router requires at least two members and a lower quantile in (0, 0.5)")
    # Preserve V7/V9's train-root-only blocked regularization selection.  Each
    # bootstrap member uses the selected penalty and one coherent resample of
    # root features plus their paired cross-fitted advantages.
    _base_router, selection = _fit_selected_ridge_advantage(ordered_roots, values, target)
    selected_l2 = float(selection["selected"]["l2"])
    rng = np.random.default_rng(int(seed))
    members = []
    for _ in range(int(member_count)):
        indices = rng.integers(0, len(values), size=len(values))
        members.append(_fit_ridge_advantage(values[indices], target[indices], l2=selected_l2))

    def predict(evaluation: np.ndarray) -> np.ndarray:
        checked = np.asarray(evaluation, dtype=np.float64)
        if checked.ndim != 2 or checked.shape[1] != values.shape[1] or not np.isfinite(checked).all():
            raise ValueError("bootstrap advantage-router evaluation features must be finite and feature-compatible")
        prediction = np.stack([member(checked) for member in members], axis=0)
        return np.asarray(np.quantile(prediction, lower_quantile, axis=0), dtype=np.float64)

    return predict, {
        **selection,
        "uncertainty_transform": "train_root_bootstrap_lower_advantage_quantile",
        "bootstrap_member_count": int(member_count),
        "bootstrap_lower_quantile": float(lower_quantile),
        "bootstrap_seed": int(seed),
        "claim_boundary": (
            "Train-root-only bootstrap score construction. Independent calibration still freezes "
            "the release threshold and certification; bootstrap dispersion is not a conditional guarantee."
        ),
    }


def _routing_score(row: dict[str, Any]) -> float:
    """Read a named router score while retaining V5 logistic-artifact support."""

    value = row.get("router_score_pre_outcome", row.get("repair_probability_pre_outcome"))
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("each router row requires a finite pre-outcome routing score") from error
    if not np.isfinite(score):
        raise ValueError("each router row requires a finite pre-outcome routing score")
    return score


def _summary(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(rows.values())
    repair = np.asarray([row["planning_beats_cost"] for row in values], dtype=bool)
    release = np.asarray([row["released_planning"] for row in values], dtype=bool)
    return {
        "root_count": len(values),
        "mean_regret": {
            key: float(np.mean([row[f"{key}_regret"] for row in values]))
            for key in ("cost", "planning", "routed")
        },
        "planning_release_coverage": float(release.mean()),
        "planning_repair_prevalence": float(repair.mean()),
        "release_precision": None if not release.any() else float(repair[release].mean()),
        "release_recall": None if not repair.any() else float(release[repair].mean()),
    }


def _calibration_partition(rows: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Separate threshold search from its prospective release certificate."""

    select = {root: row for root, row in rows.items() if _root_fold(root) % 2 == 0}
    certify = {root: row for root, row in rows.items() if _root_fold(root) % 2 == 1}
    if not select or not certify:
        raise ValueError("calibration split must populate both selector and certification roots")
    return select, certify


def _policy_regret(rows: dict[str, dict[str, Any]], policy: str) -> np.ndarray:
    if policy not in {"cost", "planning", "routed"}:
        raise ValueError(f"unknown policy: {policy}")
    return np.asarray([row[f"{policy}_regret"] for row in rows.values()], dtype=np.float64)


def _upper_bootstrap_delta(candidate: np.ndarray, incumbent: np.ndarray, *, seed: int, draws: int = 4096) -> float:
    """One-sided paired upper confidence bound on candidate minus incumbent regret."""

    if candidate.shape != incumbent.shape or not len(candidate):
        raise ValueError("paired regret arrays must be non-empty and conformable")
    delta = candidate - incumbent
    rng = np.random.default_rng(int(seed))
    resamples = rng.integers(0, len(delta), size=(int(draws), len(delta)))
    return float(np.quantile(delta[resamples].mean(axis=1), 0.95))


def _calibrate_policy(
    rows: dict[str, dict[str, Any]], *, seed: int, minimum_release_coverage: float = 0.10,
    opportunity_gated: bool = False, minimum_opportunity_roots_per_role: int = 1,
) -> dict[str, Any]:
    """Freeze a threshold, then certify it on disjoint calibration roots.

    Threshold search is deliberately *not* allowed to reuse certification roots.
    This makes a failed certificate a safe fixed-arm fallback instead of a
    threshold-search win that is silently deployed.
    """

    selector, certification = _calibration_partition(rows)
    def eligible(partition: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if not opportunity_gated:
            return partition
        result = {
            root: row for root, row in partition.items()
            if bool(row.get("routing_eligible", False))
        }
        if len(result) < int(minimum_opportunity_roots_per_role):
            raise ValueError(
                "opportunity-gated calibration lacks the predeclared minimum "
                "number of scorer-disagreement roots"
            )
        return result

    selector_scope, certification_scope = eligible(selector), eligible(certification)
    cost_mean, planning_mean = (float(_policy_regret(selector_scope, arm).mean()) for arm in ("cost", "planning"))
    # V9's scientific comparison is to the direct cost fallback.  Even if the
    # planning arm wins on the opportunity stratum, it has not earned control
    # over roots where the arms pick the same action.  Keeping `cost` as the
    # fixed incumbent makes both the certificate and the all-root consequence
    # answer that deployable question directly.
    # A rejected route must fail closed to the protocol's fixed cost arm.
    # Selecting planning after seeing selector-root labels would make a failed
    # certificate a data-adaptive arm switch, contradicting the declared V7
    # deployment contract and obscuring whether the prospective router added
    # value.  The already-executed V7 artifact is retained unchanged; this
    # governs only future panels.
    incumbent = "cost"
    scores = [_routing_score(row) for row in selector_scope.values()]
    # Score values are model-specific.  Only selector-root values and a
    # no-release reference are eligible as release boundaries.
    thresholds = sorted({float("inf"), *scores})
    candidates: list[tuple[float, dict[str, dict[str, Any]]]] = []
    for threshold in thresholds:
        evaluated = {
            root: {
                **row,
                "released_planning": bool((not opportunity_gated or row.get("routing_eligible", False)) and _routing_score(row) >= threshold),
                "routed_regret": row["planning_regret"] if (not opportunity_gated or row.get("routing_eligible", False)) and _routing_score(row) >= threshold else row["cost_regret"],
            }
            for root, row in selector.items()
        }
        coverage_rows = eligible(evaluated)
        if _summary(coverage_rows)["planning_release_coverage"] >= float(minimum_release_coverage):
            candidates.append((threshold, evaluated))
    if not candidates:
        raise ValueError("no threshold meets minimum release coverage on selector roots")
    threshold, selected = min(candidates, key=lambda item: (_policy_regret(item[1], "routed").mean(), -item[0]))
    certified = {
        root: {
            **row,
            "released_planning": bool((not opportunity_gated or row.get("routing_eligible", False)) and _routing_score(row) >= threshold),
            "routed_regret": row["planning_regret"] if (not opportunity_gated or row.get("routing_eligible", False)) and _routing_score(row) >= threshold else row["cost_regret"],
        }
        for root, row in certification.items()
    }
    certified_scope = eligible(certified)
    candidate, baseline = _policy_regret(certified_scope, "routed"), _policy_regret(certified_scope, incumbent)
    mean_delta = float((candidate - baseline).mean())
    upper_delta = _upper_bootstrap_delta(candidate, baseline, seed=int(seed) + 113)
    authorized = bool(mean_delta < 0.0 and upper_delta <= 0.0)
    all_candidate, all_baseline = _policy_regret(certified, "routed"), _policy_regret(certified, incumbent)
    all_mean_delta = float((all_candidate - all_baseline).mean())
    all_upper_delta = _upper_bootstrap_delta(all_candidate, all_baseline, seed=int(seed) + 127)
    if opportunity_gated:
        authorized = bool(authorized and all_mean_delta < 0.0 and all_upper_delta <= 0.0)
    return {
        "selector_root_count": len(selector), "certification_root_count": len(certification),
        "opportunity_gated": bool(opportunity_gated),
        "selector_opportunity_root_count": len(selector_scope),
        "certification_opportunity_root_count": len(certification_scope),
        "minimum_opportunity_roots_per_role": int(minimum_opportunity_roots_per_role),
        "fixed_incumbent": incumbent, "threshold": float(threshold),
        "minimum_release_coverage": float(minimum_release_coverage),
        "selector_summary": _summary(selected), "certification_summary": _summary(certified),
        "certification_mean_delta_vs_incumbent": mean_delta,
        "certification_upper_95_delta_vs_incumbent": upper_delta,
        "certification_all_root_mean_delta_vs_incumbent": all_mean_delta,
        "certification_all_root_upper_95_delta_vs_incumbent": all_upper_delta,
        "routing_authorized": authorized,
        "deployment_policy": "routed" if authorized else incumbent,
    }


def _deployment_summary(rows: dict[str, dict[str, Any]], policy: str) -> dict[str, Any]:
    """Summarize the sole policy actually authorized before outcome execution."""

    if policy not in {"cost", "planning", "routed"}:
        raise ValueError(f"unknown deployment policy: {policy}")
    deployed = {
        root: {
            **row,
            "released_planning": row["released_planning"] if policy == "routed" else policy == "planning",
            "routed_regret": row[f"{policy}_regret"],
        }
        for root, row in rows.items()
    }
    return _summary(deployed)


def _evaluate(
    groups: dict[str, list[dict[str, Any]]], costs: dict[str, np.ndarray], cost_predictor: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    planning_predictor: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], router: Callable[[np.ndarray], np.ndarray], threshold: float,
    *, router_kind: str, action_representation: str, opportunity_gated: bool = False,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    feature_fn = _feature_for_action_representation(action_representation)
    for root, group in groups.items():
        # The scorer must receive exactly the same declared action schema at
        # evaluation as it did at fit time.  In particular, V7 deliberately
        # contains candidates which collide under the legacy summary but differ
        # in their ordered interior pulses.
        features = np.stack([feature_fn(row) for row in group])
        cost_prediction, cost_effect = cost_predictor(features)
        planning_prediction, planning_effect = planning_predictor(features)
        router_features = (
            _opportunity_compact_features(group, cost_prediction, planning_prediction)
            if opportunity_gated else _root_features(
                group, cost_prediction, planning_prediction, cost_effect, planning_effect,
                action_representation=action_representation,
            )
        )
        score = float(router(router_features[None, :])[0])
        cost_pick, planning_pick = _choice(group, cost_prediction), _choice(group, planning_prediction)
        routing_eligible = bool(cost_pick != planning_pick)
        released = bool(score >= threshold and (routing_eligible or not opportunity_gated))
        selected = planning_pick if released else cost_pick
        rows[root] = {
            "router_kind": str(router_kind), "router_score_pre_outcome": score,
            # Keep this legacy field only for V5 logistic compatibility.  It
            # is intentionally absent for an advantage score, which is not a
            # calibrated probability.
            **({"repair_probability_pre_outcome": score} if router_kind == "logistic" else {}),
            "released_planning": released,
            "candidate_disagreement": routing_eligible,
            "routing_eligible": routing_eligible if opportunity_gated else True,
            "cost_candidate_id": str(group[cost_pick]["candidate_id"]),
            "planning_candidate_id": str(group[planning_pick]["candidate_id"]),
            "routed_candidate_id": str(group[selected]["candidate_id"]),
            "cost_regret": _regret(costs[root], cost_pick),
            "planning_regret": _regret(costs[root], planning_pick),
            "routed_regret": _regret(costs[root], selected),
            "planning_beats_cost": bool(_regret(costs[root], planning_pick) + 1.0e-10 < _regret(costs[root], cost_pick)),
        }
    return rows


def _cost_map(rows: list[dict[str, Any]], terminal: np.ndarray) -> dict[str, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, terminal):
        grouped[str(row["instance_id"])].append(float(np.linalg.norm(value[:2] - np.asarray(row["goal_raw"], dtype=np.float64))))
    return {root: np.asarray(value, dtype=np.float64) for root, value in grouped.items()}


def _post_execution_labels(rows: list[dict[str, Any]], terminal: np.ndarray) -> dict[str, dict[str, Any]]:
    """Serialize realized labels only after the frozen outcome execution point."""

    if len(rows) != len(terminal):
        raise ValueError("post-execution labels require aligned rows and terminals")
    labels: dict[str, dict[str, Any]] = {}
    for row, value in zip(rows, terminal):
        root, candidate = str(row["instance_id"]), str(row["candidate_id"])
        key = f"{root}/{candidate}"
        if key in labels:
            raise ValueError(f"duplicate post-execution label identity: {key}")
        terminal_state = np.asarray(value, dtype=np.float64).reshape(-1)
        if terminal_state.shape != (4,) or not np.isfinite(terminal_state).all():
            raise ValueError("post-execution terminal state must be finite [4]")
        labels[key] = {
            "instance_id": root, "candidate_id": row["candidate_id"],
            "realized_terminal_state": terminal_state.tolist(),
            "real_task_cost": float(np.linalg.norm(terminal_state[:2] - np.asarray(row["goal_raw"], dtype=np.float64))),
        }
    return labels


def _validate_root_partition(
    train_rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]], outcome_rows: list[dict[str, Any]],
) -> dict[str, int]:
    """Fail closed unless the three experimental roles are root-disjoint.

    The runner receives records assembled by an upstream panel builder.  It
    must still enforce this at the execution boundary: an overlapping root
    would let the router or its release threshold observe the same initial
    condition later reported as an outcome consequence.
    """

    partitions = {
        "train": {str(row["instance_id"]) for row in train_rows},
        "calibration": {str(row["instance_id"]) for row in calibration_rows},
        "outcome": {str(row["instance_id"]) for row in outcome_rows},
    }
    overlaps = {
        "train_calibration": sorted(partitions["train"] & partitions["calibration"]),
        "train_outcome": sorted(partitions["train"] & partitions["outcome"]),
        "calibration_outcome": sorted(partitions["calibration"] & partitions["outcome"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"train, calibration, and outcome roots must be disjoint: {overlaps}")
    return {name: len(values) for name, values in partitions.items()}


def _validate_shared_candidate_support(
    source: dict[str, Any], context: dict[str, Any], *,
    router_kind: str = V7_ROUTER_KIND, action_representation: str = V7_ACTION_REPRESENTATION,
    opportunity_gated: bool = False,
) -> dict[str, Any]:
    """Bind V7 execution to its shared-candidate support ledger.

    A previous PointMaze package intentionally held all action sequences out of
    training.  That is valid for action-generalization evaluation, but it
    invalidates calibration of a repairability router whose features include
    candidate action geometry.  V7 is the separate in-support protocol: this
    check prevents a manual runner invocation from silently substituting the
    former regime.
    """

    metadata = context.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("shared-candidate support requires context metadata")
    if metadata.get("candidate_generalization_regime") != "shared_candidate_in_support":
        raise ValueError("V7 requires candidate_generalization_regime=shared_candidate_in_support")
    if metadata.get("candidate_design") not in SUPPORTED_SHARED_ADVANTAGE_CANDIDATE_DESIGNS:
        raise ValueError("shared-support advantage routing requires a declared headroom-screened interaction candidate design")
    if metadata.get("scorer_candidate_bank_protocol") != "held_exact_shared_bank":
        raise ValueError("V7 requires scorer_candidate_bank_protocol=held_exact_shared_bank")
    if router_kind not in ADVANTAGE_ROUTER_KINDS or action_representation != V7_ACTION_REPRESENTATION:
        raise ValueError("shared-support advantage routing requires advantage_ridge or advantage_bootstrap_lower with full_ordered_20x2 action representation")
    train_rows = source.get("records")
    held_rows = context.get("records")
    if not isinstance(train_rows, list) or not isinstance(held_rows, list):
        raise ValueError("shared-candidate support requires source/context record arrays")
    train_hashes = {str(row.get("candidate_action_hash", "")) for row in train_rows}
    held_hashes = {str(row.get("candidate_action_hash", "")) for row in held_rows}
    train_root_count = len({str(row.get("instance_id", "")) for row in train_rows if str(row.get("split", "")) == "train"})
    if "" in train_hashes or "" in held_hashes or held_hashes != train_hashes:
        raise ValueError("V7 scorer training and held candidate action banks must be exactly identical")
    ledger = metadata.get("frozen_preflight_ledger")
    expected_schema = V9_PREFLIGHT_SCHEMA if opportunity_gated else V7_PREFLIGHT_SCHEMA
    if not isinstance(ledger, dict) or ledger.get("schema") != expected_schema or ledger.get("status") != "passed_static_preflight":
        raise ValueError("shared-support advantage routing requires the matching passed static-preflight provenance receipt")
    ledger_path = ledger.get("path")
    if not isinstance(ledger_path, str) or not Path(ledger_path).is_file():
        raise ValueError("V7 static-preflight provenance receipt lacks a readable ledger path")
    frozen = _load(Path(ledger_path))
    bank = frozen.get("candidate_bank")
    router_dimension = V9_ROUTER_FEATURE_DIMENSION if opportunity_gated else V7_ROUTER_FEATURE_DIMENSION
    router_contract = V9_ROUTER_FEATURE_CONTRACT if opportunity_gated else "full_ordered_20x2_root_features_plus_frozen_effect_disagreement_v1"
    minimum_surplus = V9_MIN_ROUTER_ROOT_SURPLUS if opportunity_gated else 16
    if (
        not isinstance(bank, dict) or bank.get("shared_candidate_in_support") is not True
        or bank.get("held_action_hash_count") != len(held_hashes)
        or bank.get("held_train_action_hash_overlap_count") != len(held_hashes)
        or bank.get("scorer_candidate_bank_protocol") != "held_exact_shared_bank"
        or bank.get("scorer_training_action_hash_count") != len(held_hashes)
        or bank.get("oracle_headroom_contract") != "design_root_terminal_goal_cost_v1"
        or float(bank.get("design_best_second_cost_gap_q10", -1.0)) < 0.001
        or float(bank.get("design_best_second_cost_gap_ge_0p003_fraction", -1.0)) < 0.50
        or float(bank.get("design_candidate_cost_spread_q10", -1.0)) < 0.05
        or bank.get("router_feature_contract") != router_contract
        or bank.get("router_feature_dimension") != router_dimension
        or bank.get("router_train_root_count") != train_root_count
        or bank.get("minimum_router_train_root_surplus", -1) < minimum_surplus
        or bank.get("router_train_root_count", 0) < bank.get("router_feature_dimension", 0) + bank.get("minimum_router_train_root_surplus", 0)
        or bank.get("router_kind") != router_kind
        or bank.get("router_action_representation") != action_representation
        or bank.get("router_regularization_selection") != V7_ROUTER_REGULARIZATION_SELECTION
        or frozen.get("post_execution_aliasing_contract") != V7_POST_EXECUTION_ALIASING_CONTRACT
    ):
        raise ValueError("shared-support static-preflight ledger lacks support, headroom, capacity, execution-interface, or aliasing-contract evidence")
    if opportunity_gated and (
        bank.get("router_training_scope") != "scorer_disagreement_only"
        or bank.get("opportunity_gate") != "cost_planning_candidate_disagreement"
        or bank.get("minimum_router_opportunity_root_count") != V9_MIN_TRAIN_OPPORTUNITY_ROOTS
        or bank.get("minimum_calibration_opportunity_roots_per_role") != V9_MIN_CALIBRATION_OPPORTUNITY_ROOTS_PER_ROLE
    ):
        raise ValueError("V9 static-preflight ledger lacks the declared opportunity-gating contract")
    return {
        "verified": True, "held_action_hash_count": len(held_hashes),
        "train_action_hash_count": len(train_hashes), "held_train_action_hash_overlap_count": len(held_hashes),
        "ledger_path": str(Path(ledger_path).resolve()),
        "router_feature_dimension": int(bank["router_feature_dimension"]), "router_train_root_count": train_root_count,
        "router_feature_contract": str(bank["router_feature_contract"]),
        "router_kind": router_kind, "action_representation": action_representation,
        "opportunity_gated": bool(opportunity_gated),
    }


def run(*, source_dataset: Path, source_arrays: Path, context: Path, calibration_labels: Path, output: Path,
        device: str = "cuda", seed: int = 202610305, width: int = 32, steps: int = 400, ensemble_size: int = 4,
        router_kind: str = "logistic", action_representation: str = SUMMARY_ACTION_REPRESENTATION,
        require_shared_candidate_support: bool = False, opportunity_gated: bool = False) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be positive")
    if router_kind not in {"logistic", *ADVANTAGE_ROUTER_KINDS}:
        raise ValueError("router_kind must be logistic, advantage_ridge, or advantage_bootstrap_lower")
    feature_fn = _feature_for_action_representation(action_representation)
    source, context_payload, calibration = _load(source_dataset), _load(context), _load(calibration_labels)
    shared_support = _validate_shared_candidate_support(
        source, context_payload, router_kind=router_kind, action_representation=action_representation,
        opportunity_gated=opportunity_gated,
    ) if require_shared_candidate_support else None
    train_rows = [dict(row) for row in source.get("records", []) if str(row.get("split")) == "train"]
    calibration_rows = [dict(row) for row in context_payload.get("records", []) if str(row.get("split")) == "calibration"]
    outcome_rows = [dict(row) for row in context_payload.get("records", []) if str(row.get("split")) == "outcome"]
    if not train_rows or not calibration_rows or not outcome_rows:
        raise ValueError("source/context require non-empty train, calibration, and outcome records")
    root_counts = _validate_root_partition(train_rows, calibration_rows, outcome_rows)
    cuda = require_cuda_allocation(device)
    with np.load(source_arrays, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}
    train_terminal = np.stack([_terminal_from_ref(row, arrays) for row in train_rows])
    train_groups = _group_rows(train_rows)
    train_by_root = {root: [row for row in train_rows if str(row["instance_id"]) == root] for root in train_groups}
    terminal_by_root = {root: train_terminal[[index for index, row in enumerate(train_rows) if str(row["instance_id"]) == root]] for root in train_groups}
    seeds = tuple(int(seed) + offset for offset in range(int(ensemble_size)))

    # Cross-fit labels answer the prospective question on roots excluded from
    # each scorer fit.  Neither calibration nor outcome terminals appear here.
    cross_features, cross_repair_labels, cross_advantages, cross_opportunities = {}, {}, {}, {}
    for fold in range(4):
        held = [root for root in train_groups if _root_fold(root) == fold]
        fit = [root for root in train_groups if _root_fold(root) != fold]
        if not held or not fit:
            raise ValueError("cross-fitting produced an empty fold")
        fit_rows = [row for root in fit for row in train_by_root[root]]
        fit_terminal = np.concatenate([terminal_by_root[root] for root in fit])
        _cost_contract, cost_predictor = _fit_ensemble(
            fit_rows, fit_terminal, COST_ONLY, seeds=seeds, width=width, steps=steps,
            device=device, feature_fn=feature_fn, action_representation=action_representation,
        )
        _plan_contract, planning_predictor = _fit_ensemble(
            fit_rows, fit_terminal, PLANNING_SHAPED, seeds=seeds, width=width, steps=steps,
            device=device, feature_fn=feature_fn, action_representation=action_representation,
        )
        for root in held:
            group, terminal = train_by_root[root], terminal_by_root[root]
            # Cross-fit predictions are part of the router's training data,
            # so this must use the same action representation as both scorers.
            features = np.stack([feature_fn(row) for row in group])
            cost_prediction, cost_effect = cost_predictor(features)
            planning_prediction, planning_effect = planning_predictor(features)
            cost = _cost(group, terminal)
            cross_features[root] = (
                _opportunity_compact_features(group, cost_prediction, planning_prediction)
                if opportunity_gated else _root_features(
                    group, cost_prediction, planning_prediction, cost_effect, planning_effect,
                    action_representation=action_representation,
                )
            )
            cost_regret = _regret(cost, _choice(group, cost_prediction))
            planning_regret = _regret(cost, _choice(group, planning_prediction))
            cross_opportunities[root] = bool(
                _choice(group, cost_prediction) != _choice(group, planning_prediction)
            )
            cross_repair_labels[root] = float(planning_regret + 1.0e-10 < cost_regret)
            cross_advantages[root] = float(cost_regret - planning_regret)

    cost_contract, cost_predictor = _fit_ensemble(
        train_rows, train_terminal, COST_ONLY, seeds=seeds, width=width, steps=steps,
        device=device, feature_fn=feature_fn, action_representation=action_representation,
    )
    planning_contract, planning_predictor = _fit_ensemble(
        train_rows, train_terminal, PLANNING_SHAPED, seeds=seeds, width=width, steps=steps,
        device=device, feature_fn=feature_fn, action_representation=action_representation,
    )
    ordered_cross_roots = sorted(cross_features)
    router_train_roots = [
        root for root in ordered_cross_roots
        if not opportunity_gated or cross_opportunities[root]
    ]
    if opportunity_gated and len(router_train_roots) < V9_MIN_TRAIN_OPPORTUNITY_ROOTS:
        raise ValueError(
            "opportunity-gated router lacks the predeclared minimum number of "
            "cross-fitted scorer-disagreement train roots"
        )
    router_features = np.stack([cross_features[root] for root in router_train_roots])
    if shared_support is not None and int(router_features.shape[1]) != int(shared_support["router_feature_dimension"]):
        raise RuntimeError("realized router feature dimension differs from the frozen static-preflight contract")
    router_selection: dict[str, Any]
    if router_kind == "logistic":
        router = _fit_logistic(router_features, np.asarray([cross_repair_labels[root] for root in router_train_roots], dtype=np.float64))
        router_selection = {
            "selection_protocol": "fixed_l2_logistic_legacy_control",
            "root_count": len(router_train_roots), "feature_dimension": int(router_features.shape[1]),
            "claim_boundary": "Legacy router control; no advantage-ridge regularization selection is performed.",
        }
    elif router_kind == V7_ROUTER_KIND:
        router, router_selection = _fit_selected_ridge_advantage(
            router_train_roots, router_features,
            np.asarray([cross_advantages[root] for root in router_train_roots], dtype=np.float64),
        )
    else:
        router, router_selection = _fit_bootstrap_lower_advantage(
            router_train_roots, router_features,
            np.asarray([cross_advantages[root] for root in router_train_roots], dtype=np.float64),
            seed=int(seed) + 919,
        )
    if shared_support is not None and router_selection.get("selection_protocol") != V7_ROUTER_REGULARIZATION_SELECTION:
        raise RuntimeError("V7 router regularization selection differs from the frozen static-preflight contract")

    calibration_terminal = _calibration_terminal(calibration_rows, calibration)
    calibration_groups = _group_rows(calibration_rows)
    calibration_cost = _cost_map(calibration_rows, calibration_terminal)
    calibration_grouped = {root: [row for row in calibration_rows if str(row["instance_id"]) == root] for root in calibration_groups}
    # Calibration selects the only legal threshold from the router's own
    # score scale.  The provisional zero boundary is not a policy decision.
    raw_calibration = _evaluate(
        calibration_grouped, calibration_cost, cost_predictor, planning_predictor, router,
        threshold=0.0, router_kind=router_kind, action_representation=action_representation,
        opportunity_gated=opportunity_gated,
    )
    calibration_policy = _calibrate_policy(
        raw_calibration, seed=seed, opportunity_gated=opportunity_gated,
        minimum_opportunity_roots_per_role=(V9_MIN_CALIBRATION_OPPORTUNITY_ROOTS_PER_ROLE if opportunity_gated else 1),
    )
    threshold = float(calibration_policy["threshold"])

    # The sole outcome materialization happens after all cross-fit models,
    # router, feature contract, and threshold are frozen.
    outcome_terminal = _fresh_terminal(outcome_rows)
    outcome_groups = _group_rows(outcome_rows)
    outcome_cost = _cost_map(outcome_rows, outcome_terminal)
    post_execution_outcome_labels = _post_execution_labels(outcome_rows, outcome_terminal)
    outcome_grouped = {root: [row for row in outcome_rows if str(row["instance_id"]) == root] for root in outcome_groups}
    outcome = _evaluate(
        outcome_grouped, outcome_cost, cost_predictor, planning_predictor, router, threshold,
        router_kind=router_kind, action_representation=action_representation,
        opportunity_gated=opportunity_gated,
    )
    deployment_summary = _deployment_summary(outcome, str(calibration_policy["deployment_policy"]))
    report = {
        "schema": SCHEMA, "status": "completed_execution_grounded_repairability_control",
        "claim_boundary": "A PointMaze visual-history/action root-level repairability control; not GDD, PlanWAM, D-JEPA, online deployment, or universal evidence.",
        "control_contract": {
            "label": CONTROL_LABEL, "input_is_visual_history_goal_action_and_frozen_predictions_only": True,
            "router_uses_frozen_effect_disagreement_geometry": not opportunity_gated,
            "router_feature_contract": (
                V9_ROUTER_FEATURE_CONTRACT if opportunity_gated
                else "full_ordered_20x2_root_features_plus_frozen_effect_disagreement_v1"
            ),
            "router_kind": router_kind,
            "opportunity_gated": bool(opportunity_gated),
            "opportunity_gate": "cost_planning_candidate_disagreement" if opportunity_gated else "none",
            "action_representation": action_representation,
            "action_input_fields": action_representation_input_fields(action_representation),
            "source_state_raw_forbidden_in_router": True, "crossfit_train_labels_only": True,
            "outcome_execution_deferred_until_router_threshold_and_certificate_frozen": True,
            "outcome_cost_used_for_router_or_threshold_selection": False,
            "shared_candidate_support_verified": shared_support,
        },
        "split_isolation": {
            "train_root_count": root_counts["train"], "calibration_root_count": root_counts["calibration"],
            "outcome_root_count": root_counts["outcome"], "root_disjoint": True,
        },
        "repair_arms": {"cost_only": cost_contract, "planning_shaped": planning_contract},
        "crossfit": {
            "fold_count": 4, "repair_prevalence": float(np.mean(list(cross_repair_labels.values()))),
            "mean_cost_minus_planning_regret": float(np.mean(list(cross_advantages.values()))),
            "feature_dimension": int(len(next(iter(cross_features.values())))),
            "scorer_disagreement_root_count": int(sum(cross_opportunities.values())),
            "scorer_disagreement_rate": float(np.mean(list(cross_opportunities.values()))),
            "router_training_root_count": len(router_train_roots),
            "router_training_scope": "scorer_disagreement_only" if opportunity_gated else "all_crossfit_roots",
        },
        "router_train_only_selection": router_selection,
        "calibration": {"routing": calibration_policy, "per_root": raw_calibration},
        "outcome": {
            "router_summary": _summary(outcome), "authorized_deployment_summary": deployment_summary,
            "authorized_deployment_policy": calibration_policy["deployment_policy"], "per_root": outcome,
        },
        "post_execution_outcome_labels": post_execution_outcome_labels,
        "cuda_preflight": cuda,
        "provenance": {"source_dataset": str(source_dataset.resolve()), "source_dataset_sha256": _sha256(source_dataset), "source_arrays": str(source_arrays.resolve()), "source_arrays_sha256": _sha256(source_arrays), "context": str(context.resolve()), "context_sha256": _sha256(context), "calibration_labels": str(calibration_labels.resolve()), "calibration_labels_sha256": _sha256(calibration_labels)},
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--source-arrays", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--calibration-labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--seed", type=int, default=202610305)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument(
        "--router-kind",
        choices=("logistic", "advantage_ridge", "advantage_bootstrap_lower"),
        default="logistic",
    )
    parser.add_argument(
        "--action-representation",
        choices=(SUMMARY_ACTION_REPRESENTATION, ORDERED_ACTION_REPRESENTATION),
        default=SUMMARY_ACTION_REPRESENTATION,
    )
    parser.add_argument("--require-shared-candidate-support", action="store_true")
    parser.add_argument(
        "--opportunity-gated", action="store_true",
        help=(
            "train and certify the router only where the two frozen scorers select "
            "different candidates; non-opportunity roots fail closed to cost"
        ),
    )
    result = run(**vars(parser.parse_args()))
    print(json.dumps({"status": result["status"], "crossfit": result["crossfit"], "outcome": result["outcome"]["authorized_deployment_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
