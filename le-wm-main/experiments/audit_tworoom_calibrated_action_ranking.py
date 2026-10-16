"""Calibration-only action-utility audit for the frozen TwoRoom ranking.

This is an exploratory follow-up to ``audit_tworoom_shared_candidate_ranking``.
It fits a small, frozen-after-calibration utility map from the model's shared
candidate costs and generic action features to privileged replay distance.
Calibration roots and evaluation roots are disjoint.  The simulator labels
are used only for calibration/evaluation and are never available at runtime.

The result is therefore evidence about whether a lightweight action-utility
adapter can repair the ranking bottleneck; it is not evidence for universal
planning or for a causal certificate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _directions() -> np.ndarray:
    values: list[tuple[float, float]] = [(0.0, 0.0)]
    for magnitude in (0.25, 0.50, 0.75):
        values.extend([
            (magnitude, 0.0), (-magnitude, 0.0),
            (0.0, magnitude), (0.0, -magnitude),
        ])
    values.extend([(0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)])
    return np.asarray(values, dtype=np.float64)


def _features(model_cost: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Build state-conditioned but low-capacity utility features.

    Cost normalization is performed independently for each root because only
    the ordering and relative scale inside a candidate bank are meaningful at
    deployment.  Action terms encode support geometry without action IDs.
    """
    centered = model_cost - model_cost.mean()
    scale = model_cost.std()
    z = centered / max(float(scale), 1e-8)
    dx, dy = actions.T
    base = np.column_stack([
        z,
        z * z,
        dx,
        dy,
        dx * dx,
        dy * dy,
        dx * dy,
        z * dx,
        z * dy,
        z * dx * dx,
        z * dy * dy,
        np.ones_like(z),
    ])
    return base


def _fit_ridge(features: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    scale = np.std(features, axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = features / scale
    design = np.column_stack([normalized, np.ones(len(normalized))])
    regularizer = np.eye(design.shape[1], dtype=np.float64)
    regularizer[-1, -1] = 0.0
    return np.linalg.solve(
        design.T @ design + float(ridge) * regularizer,
        design.T @ target,
    ), scale


def _predict(features: np.ndarray, fit: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    weights, scale = fit
    design = np.column_stack([features / scale, np.ones(len(features))])
    return design @ weights


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    out = np.empty(len(values), dtype=np.int64)
    out[order] = np.arange(len(values))
    return out


def _evaluate(rows: list[dict], indices: np.ndarray, actions: np.ndarray, fit=None) -> dict:
    top1 = []
    top4 = []
    regrets = []
    ranks = []
    for index in indices.tolist():
        row = rows[index]
        model_cost = np.asarray(row["model_cost"], dtype=np.float64)
        true_cost = np.asarray(row["true_cost"], dtype=np.float64)
        if fit is None:
            predicted_cost = model_cost
        else:
            predicted_cost = _predict(_features(model_cost, actions), fit)
        predicted_best = int(np.argmin(predicted_cost))
        true_order = np.argsort(true_cost, kind="stable")
        top1.append(int(predicted_best == int(true_order[0])))
        top4.append(int(predicted_best in set(true_order[:4].tolist())))
        ranks.append(int(np.flatnonzero(true_order == predicted_best)[0] + 1))
        regrets.append(float(true_cost[predicted_best] - true_cost[true_order[0]]))
    return {
        "n": int(len(indices)),
        "top1_accuracy": float(np.mean(top1)),
        "top4_accuracy": float(np.mean(top4)),
        "mean_true_rank_of_predicted_best": float(np.mean(ranks)),
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "top1_hits": int(np.sum(top1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-roots", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1.0)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = sorted(payload["roots"], key=lambda row: int(row["seed"]))
    if len(rows) <= args.calibration_roots:
        raise ValueError("need at least one held-out root")
    actions = _directions()
    calibration_indices = np.arange(args.calibration_roots, dtype=np.int64)
    test_indices = np.arange(args.calibration_roots, len(rows), dtype=np.int64)

    calibration_features = np.concatenate([
        _features(np.asarray(rows[i]["model_cost"], dtype=np.float64), actions)
        for i in calibration_indices
    ], axis=0)
    calibration_target = np.concatenate([
        np.asarray(rows[i]["true_cost"], dtype=np.float64)
        for i in calibration_indices
    ], axis=0)
    fit = _fit_ridge(calibration_features, calibration_target, args.ridge)
    calibration_result = _evaluate(rows, calibration_indices, actions, fit)
    test_raw = _evaluate(rows, test_indices, actions, fit=None)
    test_calibrated = _evaluate(rows, test_indices, actions, fit=fit)

    output = {
        "schema": "tworoom_calibrated_action_ranking.v1",
        "status": "exploratory_calibration_only",
        "protocol": {
            "input_audit": str(args.input),
            "candidate_count": int(len(actions)),
            "calibration_roots": [int(rows[i]["seed"]) for i in calibration_indices],
            "held_out_roots": [int(rows[i]["seed"]) for i in test_indices],
            "features": "within-bank normalized model cost plus action geometry and interactions",
            "ridge": float(args.ridge),
            "simulator_labels": "offline calibration/evaluation only",
            "runtime_privileged_state": False,
        },
        "results": {
            "calibration_fitted": calibration_result,
            "held_out_raw_model": test_raw,
            "held_out_calibrated": test_calibrated,
            "delta_calibrated_minus_raw_top1": float(
                test_calibrated["top1_accuracy"] - test_raw["top1_accuracy"]
            ),
            "delta_calibrated_minus_raw_regret": float(
                test_calibrated["mean_regret"] - test_raw["mean_regret"]
            ),
        },
        "claim_boundary": "Exploratory root-disjoint utility calibration; requires a fresh preregistered split and live planning validation before any control claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output["results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
