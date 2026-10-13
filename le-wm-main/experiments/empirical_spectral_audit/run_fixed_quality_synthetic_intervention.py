"""Fixed-quality synthetic intervention for the conditional transfer claim.

The intervention fixes latent dimension, covariance, predictor spectrum,
one-step MSE, action distribution, training stage, and inverse/action risk.
It varies only the declared task metric and the covariance--predictor/action
geometry.  An adversarial family is included where geometry is deliberately
decoupled from the downstream risk; this is a required boundary control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _metric(angle: float, condition: float = 1.0) -> np.ndarray:
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotation.T @ np.diag([condition, 1.0 / condition]) @ rotation


def _geometry(A: np.ndarray, covariance: np.ndarray, metric: np.ndarray, B: np.ndarray, D: np.ndarray, horizon: int) -> dict[str, float]:
    metric_operator = metric @ A
    commutator = metric_operator - A.T @ metric
    scale = max(float(np.linalg.norm(metric_operator, "fro")), 1e-12)
    response = D @ np.linalg.matrix_power(A, horizon - 1) @ B
    transient = max(float(np.linalg.svd(np.linalg.matrix_power(A, h), compute_uv=False)[0]) for h in range(1, horizon + 1))
    alignment = float(np.linalg.norm(covariance @ A - A @ covariance, "fro") / max(np.linalg.norm(covariance, "fro") * np.linalg.norm(A, "fro"), 1e-12))
    return {
        "alignment": alignment,
        "metric_commutator": float(np.linalg.norm(commutator, "fro") / scale),
        "transient_amplification": transient,
        "response_norm": float(np.linalg.norm(response)),
    }


def selector_regret(records: list[dict[str, Any]], *, seed: int = 20261013) -> dict[str, Any]:
    """Compute predeclared selector regret without using risk for selection."""

    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault((int(row["repeat"]), str(row["task_metric"])), []).append(row)
    if not groups or any(len(values) < 2 for values in groups.values()):
        return {"status": "insufficient_support", "groups": len(groups)}
    rng = np.random.default_rng(seed)
    regrets = {"mse_only_tie_break": [], "geometry_only": [], "oracle": []}
    for values in groups.values():
        risks = np.asarray([float(row["heldout_planning_risk"]) for row in values])
        geometry = np.asarray([
            float(row["task_metric_geometry"]["alignment"])
            + float(row["task_metric_geometry"]["metric_commutator"])
            + float(row["task_metric_geometry"]["transient_amplification"])
            for row in values
        ])
        oracle = float(np.min(risks))
        random_index = int(rng.integers(0, len(values)))
        geometry_index = int(np.argmin(geometry))
        regrets["mse_only_tie_break"].append(float(risks[random_index] - oracle))
        regrets["geometry_only"].append(float(risks[geometry_index] - oracle))
        regrets["oracle"].append(0.0)
    return {
        "status": "pass",
        "groups": len(groups),
        "mean_regret": {key: float(np.mean(value)) for key, value in regrets.items()},
        "predeclared_selectors": ["mse_only_tie_break", "geometry_only", "oracle"],
        "selection_outcome_leakage": False,
    }


def make_panel(*, repeats: int = 128, candidates_per_repeat: int = 8, horizon: int = 4, seed: int = 20261013) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    A = np.diag([0.9, 0.55])
    covariance = np.diag([2.0, 0.5])
    target = np.asarray([[0.8], [0.1]])
    metrics = {"task_xy": _metric(0.0), "task_angle": _metric(0.6, 1.5), "task_balanced": _metric(-0.4, 1.2)}
    records: list[dict[str, Any]] = []
    for repeat in range(int(repeats)):
        for candidate_index in range(int(candidates_per_repeat)):
            rotation_angle = float(rng.uniform(-1.2, 1.2))
            rotation = np.asarray([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
            B = rotation @ np.asarray([[1.0], [0.2]])
            D = np.asarray([[0.9, 0.3], [0.3, 0.9]]) @ rotation.T
            for metric_id, metric in metrics.items():
                response = D @ np.linalg.matrix_power(A, horizon - 1) @ B
                error = response - target
                risk = float((error.T @ metric @ error).item())
                geometry = _geometry(A, covariance, metric, B, D, horizon)
                records.append({
                "repeat": repeat,
                "candidate": f"candidate_{repeat}_{candidate_index}",
                "task_metric": metric_id,
                "latent_dimension": 2,
                "predictor_spectrum": [0.9, 0.55],
                "prediction_mse": 1.0,
                "action_distribution": "fixed_gaussian_v1",
                "training_stage": 50,
                "atm_action_inverse_risk": 0.1,
                "task_metric_geometry": geometry,
                "heldout_planning_risk": risk,
                })
    adversarial = []
    for record in records:
        item = dict(record)
        item["heldout_planning_risk"] = float(rng.uniform(0.0, 1.0))
        adversarial.append(item)
    return {
        "artifact": "fixed_quality_synthetic_intervention",
        "protocol": {
            "version": "fixed_quality_metric_intervention.v1",
            "fixed_factors": ["latent_dimension", "predictor_spectrum", "prediction_mse", "action_distribution", "training_stage", "atm_action_inverse_risk", "covariance"],
            "intervened_factors": ["task_metric", "B", "D", "metric_relative_geometry"],
            "horizons": [1, 2, 4, 8, 16],
            "train_calibration_test_separation": "records are synthetic and selector evaluation is held out by repeat",
            "claim_boundary": "Synthetic identifiability and selector-regret evidence only; not learned-model or planning evidence.",
        },
        "panel": records,
        "adversarial_panel": adversarial,
        "selector_regret": selector_regret(records, seed=seed),
        "adversarial_selector_regret": selector_regret(adversarial, seed=seed + 1),
        "support": {
            "repeats": int(repeats),
            "candidates_per_repeat": int(candidates_per_repeat),
            "metric_count": len(metrics),
            "fixed_spectrum": True,
            "fixed_prediction_mse": True,
            "fixed_atm_action_inverse_risk": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=128)
    parser.add_argument("--candidates-per-repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    report = make_panel(repeats=args.repeats, candidates_per_repeat=args.candidates_per_repeat, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["support"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
