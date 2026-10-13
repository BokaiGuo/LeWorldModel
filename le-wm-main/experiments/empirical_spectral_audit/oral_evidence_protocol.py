"""Executable P3--P6 evidence gates for the learned-model panel.

The functions here validate protocol sufficiency and implement the two-stage
selector/nested-risk estimands.  They intentionally return gated statuses when
the requested architecture/seed/environment panel is not present.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


REQUIRED_RECORD_FIELDS = {
    "architecture", "environment", "seed", "training_stage", "latent_dimension",
    "prediction_mse", "atm_action_inverse_risk", "predictor_spectrum", "task_metric",
    "planning_risk",
}


def validate_learned_checkpoint_panel(
    records: Iterable[dict[str, Any]],
    *,
    minimum_architectures: int = 3,
    minimum_seeds: int = 5,
    minimum_environments: int = 4,
    minimum_snapshots: int = 6,
    horizons: tuple[int, ...] = (1, 2, 4, 8, 16),
) -> dict[str, Any]:
    """Validate the declared cross-model panel without pooling failures."""

    rows = list(records)
    missing = []
    for index, row in enumerate(rows):
        absent = sorted(REQUIRED_RECORD_FIELDS.difference(row))
        if absent:
            missing.append({"index": index, "fields": absent})
    architectures = sorted({str(row.get("architecture")) for row in rows if "architecture" in row})
    seeds = sorted({str(row.get("seed")) for row in rows if "seed" in row})
    environments = sorted({str(row.get("environment")) for row in rows if "environment" in row})
    snapshot_keys = {(str(row.get("architecture")), str(row.get("seed"))) for row in rows}
    snapshot_counts = {
        f"{architecture}/{seed}": sum(
            str(row.get("architecture")) == architecture and str(row.get("seed")) == seed
            for row in rows
        )
        for architecture, seed in sorted(snapshot_keys)
    }
    required = {
        "architectures": len(architectures) >= minimum_architectures,
        "seeds": len(seeds) >= minimum_seeds,
        "environments": len(environments) >= minimum_environments,
        "snapshots_per_architecture_seed": bool(snapshot_counts) and min(snapshot_counts.values()) >= minimum_snapshots,
        "required_horizons": all(
            all(int(horizon) in row.get("horizons", horizons) for horizon in horizons)
            for row in rows
        ) if rows else False,
        "record_fields": not missing,
    }
    passed = bool(rows) and all(required.values())
    return {
        "status": "pass" if passed else "blocked",
        "passed": passed,
        "counts": {
            "records": len(rows),
            "architectures": len(architectures),
            "seeds": len(seeds),
            "environments": len(environments),
            "snapshot_counts": snapshot_counts,
        },
        "required": required,
        "missing": missing,
        "horizons": [int(value) for value in horizons],
    }


def _design(rows: list[dict[str, Any]], fields: tuple[str, ...], environments: list[str] | None = None) -> np.ndarray:
    values = []
    for row in rows:
        values.append([1.0] + [float(row[field]) for field in fields])
    if environments:
        values = [item + [float(row["environment"] == env) for env in environments[1:]] for item, row in zip(values, rows)]
    return np.asarray(values, dtype=np.float64)


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    ridge = 1e-8 * max(float(np.trace(train_x.T @ train_x)) / max(train_x.shape[1], 1), 1e-12)
    coefficient = np.linalg.solve(train_x.T @ train_x + ridge * np.eye(train_x.shape[1]), train_x.T @ train_y)
    return test_x @ coefficient


def nested_incremental_risk_audit(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Fit the predeclared MSE+ATM versus geometry nested models.

    The returned incremental score is leave-one-environment-out, so a row is
    never used both to fit and score the planning-risk consequence.
    """

    rows = list(records)
    environments = sorted({str(row.get("environment")) for row in rows})
    if len(rows) < 6 or len(environments) < 2:
        return {"status": "insufficient_support", "n_records": len(rows), "environment_count": len(environments)}
    base_fields = ("prediction_mse", "atm_action_inverse_risk")
    geometry_fields = ("alignment", "commutator", "schur", "transient")
    base_errors = []
    full_errors = []
    folds = []
    for heldout in environments:
        train = [row for row in rows if str(row["environment"]) != heldout]
        test = [row for row in rows if str(row["environment"]) == heldout]
        if len(train) < 3 or not test:
            continue
        train_x = _design(train, base_fields, environments)
        test_x = _design(test, base_fields, environments)
        train_full = _design(train, base_fields + geometry_fields, environments)
        test_full = _design(test, base_fields + geometry_fields, environments)
        train_y = np.asarray([float(row["planning_risk"]) for row in train])
        test_y = np.asarray([float(row["planning_risk"]) for row in test])
        base_pred = _fit_predict(train_x, train_y, test_x)
        full_pred = _fit_predict(train_full, train_y, test_full)
        base_mse = float(np.mean((base_pred - test_y) ** 2))
        full_mse = float(np.mean((full_pred - test_y) ** 2))
        base_errors.append(base_mse)
        full_errors.append(full_mse)
        folds.append({"heldout_environment": heldout, "base_mse": base_mse, "full_mse": full_mse})
    if not folds:
        return {"status": "insufficient_support", "n_records": len(rows), "environment_count": len(environments)}
    return {
        "status": "pass" if np.mean(full_errors) < np.mean(base_errors) else "fail_or_partial",
        "n_folds": len(folds),
        "base_loeo_mse": float(np.mean(base_errors)),
        "full_loeo_mse": float(np.mean(full_errors)),
        "incremental_geometry_mse_gain": float(np.mean(base_errors) - np.mean(full_errors)),
        "folds": folds,
        "model_specification": "risk ~ MSE + ATM + geometry + environment; leave-one-environment-out",
    }


def two_stage_selector(
    records: Iterable[dict[str, Any]],
    *,
    mse_tolerance: float = 0.05,
    geometry_key: str = "alignment",
) -> dict[str, Any]:
    """Select within MSE-matched strata using a train/calibration metric."""

    rows = list(records)
    if not rows:
        return {"status": "insufficient_support", "reason": "no_records"}
    minimum = min(float(row["prediction_mse"]) for row in rows)
    candidates = [row for row in rows if float(row["prediction_mse"]) <= minimum + float(mse_tolerance)]
    if len(candidates) < 2:
        return {"status": "blocked", "reason": "no_mse_matched_candidates", "candidate_count": len(candidates)}
    if any(geometry_key not in row for row in candidates):
        return {"status": "blocked", "reason": "geometry_missing", "candidate_count": len(candidates)}
    selected = min(candidates, key=lambda row: float(row[geometry_key]))
    risks = [float(row["planning_risk"]) for row in candidates]
    oracle = min(risks)
    return {
        "status": "pass",
        "candidate_count": len(candidates),
        "geometry_key": geometry_key,
        "selected_checkpoint": selected.get("checkpoint", selected.get("candidate")),
        "selected_risk": float(selected["planning_risk"]),
        "oracle_risk": oracle,
        "regret": float(float(selected["planning_risk"]) - oracle),
        "selector": "MSE-filtered-geometry",
        "test_outcome_used_for_selection": False,
    }


def pusht_mechanism_protocol() -> dict[str, Any]:
    """Return the frozen mechanism-level PushT protocol contract."""

    return {
        "stages": ["history", "mode_estimation", "mode_conditioned_effect", "short_pulse", "rgb_reobservation", "recovery"],
        "required_metrics": ["velocity_prediction", "contact_mode_accuracy", "action_ranking", "counterfactual_effect_error", "contact_retention", "block_xy", "angle_error", "terminal_success"],
        "runtime_constraints": ["rgb_only", "shared_candidate_bank", "real_predicted_transition_gate", "receding_horizon", "no_privileged_successor_verification", "fresh_initial_states", "unseen_seeds"],
        "claim_boundary": "PushT is a mechanism experiment; mode accuracy without angle closure is not task success.",
    }
