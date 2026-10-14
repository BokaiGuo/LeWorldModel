"""Calibrate a task-coordinate chart before estimated goal-frame selection.

This is a predeclared diagnostic for the remaining failure case where RGB
successor prediction improves but estimated goal-frame candidate ranking does
not.  A source-latent -> source-XY chart is fit on train roots only.  The
chart family (linear or quadratic) is selected using calibration-root source
coordinate error, with calibration task regret reported separately.  Outcome
roots are never used for fitting or chart selection.

The experiment is an offline task-chart adapter.  It is not evidence for
causal dynamics, D-JEPA preference validation, or closed-loop RGB planning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gdd_wm.audit_transfer_sufficient_feature_families import (
    _action_summary,
    _evaluate,
    _fit_ridge,
    _predict,
)


CHARTS = ("linear", "quadratic")


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("candidate dataset must contain records")
    return payload


def _rows(dataset: dict[str, Any], arrays_path: Path) -> list[dict[str, Any]]:
    arrays = np.load(arrays_path, allow_pickle=False)
    output = []
    for record in dataset["records"]:
        key = str(record["predicted_latent_ref"]).split("::", 1)[-1]
        if key not in arrays.files:
            raise ValueError(f"missing predicted latent array {key}")
        row = dict(record)
        row["_predicted"] = np.asarray(arrays[key], dtype=np.float64)
        if row["_predicted"].ndim != 1 or not bool(np.all(np.isfinite(row["_predicted"]))):
            raise ValueError(f"invalid predicted latent array {key}")
        output.append(row)
    return output


def _unique_root_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(str(row["instance_id"]), row)
    return list(unique.values())


def _chart_features(latents: np.ndarray, chart: str) -> np.ndarray:
    values = np.asarray(latents, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("latents must be rank-2")
    if chart == "linear":
        return values
    if chart == "quadratic":
        terms = [values, values * values]
        pair_terms = [
            (values[:, left] * values[:, right])[:, None]
            for left in range(values.shape[1])
            for right in range(left + 1, values.shape[1])
        ]
        return np.concatenate(terms + pair_terms, axis=1)
    raise ValueError(f"unknown chart {chart!r}")


def _fit_chart(rows: list[dict[str, Any]], chart: str, alpha: float) -> dict[str, np.ndarray]:
    roots = _unique_root_rows(rows)
    latents = np.asarray([row["source_latent_model"] for row in roots], dtype=np.float64)
    targets = np.asarray([row["source_state_raw"][:2] for row in roots], dtype=np.float64)
    return _fit_ridge(_chart_features(latents, chart), targets, alpha)


def _estimate_positions(rows: list[dict[str, Any]], fit: dict[str, np.ndarray], chart: str) -> dict[str, np.ndarray]:
    latents = np.asarray([row["source_latent_model"] for row in rows], dtype=np.float64)
    positions = _predict(_chart_features(latents, chart), fit)
    return {str(row["instance_id"]): positions[index] for index, row in enumerate(rows)}


def _goalframe_action(row: dict[str, Any], source_position: np.ndarray) -> np.ndarray:
    source = np.asarray(source_position, dtype=np.float64).reshape(-1)
    goal = np.asarray(row["goal_raw"], dtype=np.float64).reshape(-1)
    direct = goal[:2] - source[:2]
    norm = float(np.linalg.norm(direct))
    if norm <= 1.0e-8:
        raise ValueError("source and goal must be distinct")
    direct = direct / norm
    perpendicular = np.asarray([-direct[1], direct[0]], dtype=np.float64)
    actions = np.asarray(row["action_sequence"], dtype=np.float64)
    canonical = np.stack([actions @ direct, actions @ perpendicular], axis=1)
    return _action_summary(canonical)


def _features(row: dict[str, Any], position: np.ndarray) -> np.ndarray:
    source = np.asarray(row["source_latent_model"], dtype=np.float64)
    predicted = np.asarray(row["_predicted"], dtype=np.float64)
    return np.concatenate([source, predicted - source, _goalframe_action(row, position)])


def _position_rmse(rows: list[dict[str, Any]], positions: dict[str, np.ndarray]) -> float:
    roots = _unique_root_rows(rows)
    errors = [
        np.asarray(positions[str(row["instance_id"])]) - np.asarray(row["source_state_raw"][:2])
        for row in roots
    ]
    return float(np.sqrt(np.mean(np.square(np.asarray(errors, dtype=np.float64)))))


def _calibration_key(result: dict[str, Any]) -> tuple[float, float]:
    regret = result["candidate_regret"]["calibration"]["mean_regret"]
    consistency = result["execution_consistency"]["calibration"]["pairs"]["execution_consistency"]
    return (float("inf") if regret is None else float(regret), -(float("-inf") if consistency is None else float(consistency)))


def audit(
    candidate_dataset: Path,
    arrays_path: Path,
    *,
    alpha: float = 1.0e-2,
    margin: float = 1.0e-3,
) -> dict[str, Any]:
    dataset = _load_report(candidate_dataset)
    rows = _rows(dataset, arrays_path)
    train = [row for row in rows if str(row["split"]) == "train"]
    calibration = [row for row in rows if str(row["split"]) == "calibration"]
    outcome = [row for row in rows if str(row["split"]) == "outcome"]
    if not train or not calibration or not outcome:
        raise ValueError("train, calibration, and outcome rows are required")
    train_roots = {str(row["instance_id"]) for row in train}
    if train_roots & ({str(row["instance_id"]) for row in calibration} | {str(row["instance_id"]) for row in outcome}):
        raise ValueError("root-disjoint splits are required")

    results: dict[str, Any] = {}
    for chart in CHARTS:
        fit = _fit_chart(train, chart, alpha)
        positions = _estimate_positions(rows, fit, chart)
        features = np.asarray([_features(row, positions[str(row["instance_id"])]) for row in rows], dtype=np.float64)
        train_mask = np.asarray([str(row["split"]) == "train" for row in rows])
        targets = np.asarray([float(row["real_task_cost"]) for row in rows], dtype=np.float64)
        readout = _fit_ridge(features[train_mask], targets[train_mask], alpha)
        scores = _predict(features, readout)
        chart_rows = [row for row in rows if str(row["split"]) == "train"]
        train_position_rmse = _position_rmse(chart_rows, positions)
        calibration_position_rmse = _position_rmse(calibration, positions)
        outcome_position_rmse = _position_rmse(outcome, positions)
        results[chart] = {
            "chart_feature_dimension": int(_chart_features(np.asarray([train[0]["source_latent_model"]]), chart).shape[1]),
            "train_position_rmse": train_position_rmse,
            "calibration_position_rmse": calibration_position_rmse,
            "outcome_position_rmse_diagnostic": outcome_position_rmse,
            "train_cost_fit_rmse": float(np.sqrt(np.mean((_predict(features[train_mask], readout) - targets[train_mask]) ** 2))),
            **_evaluate(rows, scores, margin),
        }

    selected = min(CHARTS, key=lambda chart: _calibration_key(results[chart]))
    return {
        "schema": "gdd_wm_task_chart_calibration.v0.1",
        "status": "completed_diagnostic",
        "protocol": {
            "candidate_dataset": str(candidate_dataset.resolve()),
            "arrays": str(arrays_path.resolve()),
            "chart_candidates": list(CHARTS),
            "chart_fit_split": "train",
            "chart_selection_split": "calibration",
            "task_readout_fit_split": "train",
            "evaluation_split": "outcome",
            "selection_rule": "minimum calibration candidate regret, then maximum calibration pair consistency",
            "outcome_used_for_fit_or_chart_selection": False,
            "ridge_alpha": float(alpha),
            "pair_margin": float(margin),
        },
        "selected_chart": selected,
        "results": results,
        "claim_boundary": (
            "offline train-only latent task-chart calibration and calibrated candidate selection; "
            "not D-JEPA preference validation, physical causality, or closed-loop RGB planning"
        ),
        "input_metadata": dataset.get("metadata", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", required=True, type=Path)
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=1.0e-2)
    parser.add_argument("--margin", type=float, default=1.0e-3)
    args = parser.parse_args()
    report = audit(args.candidate_dataset, args.arrays, alpha=args.alpha, margin=args.margin)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected_chart": report["selected_chart"], "results": report["results"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
