"""Mechanism decomposition for an independent planning-gain failure.

This diagnostic consumes one explicitly supplied checkpoint record and never
selects a checkpoint or a formal metric from outcome rows.  It separates
horizon ranking, task-readout calibration, candidate-bank coverage, and
outcome heterogeneity.  The output is diagnostic evidence; it is not a new
confirmatory result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_independent_pointmaze_planning_gain import _payload_rows, _resolve
from train_pointmaze_learned_adapter_panel import HORIZONS, _task_direction, _task_readout_with_center


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.ptp(left) <= 1e-12 or np.ptp(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _task_arrays(rows: list[dict[str, Any]], decoder: dict[int, np.ndarray], direction: np.ndarray, latent_center: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    predicted: dict[int, np.ndarray] = {}
    actual: dict[int, np.ndarray] = {}
    for horizon in HORIZONS:
        predicted_latent = np.asarray([
            row["latent_by_horizon"][str(horizon)]["predicted"] for row in rows
        ], dtype=np.float64)
        predicted[horizon] = (predicted_latent - latent_center[horizon]) @ decoder[horizon] @ direction
        actual[horizon] = np.asarray([
            (np.asarray(row["latent_by_horizon"][str(horizon)]["target_state_raw"], dtype=np.float64)[:2]
             - np.asarray(row["source_state_raw"], dtype=np.float64)[:2]) @ direction
            for row in rows
        ], dtype=np.float64)
    return predicted, actual


def diagnose(record_path: Path) -> dict[str, Any]:
    outer = json.loads(record_path.read_text())
    record = outer.get("record", outer)
    protocol = record["protocol"]
    fit_path = _resolve(str(record["provenance"]["fit_export"]), record_path.parent)
    outcome_path = _resolve(str(record["provenance"]["outcome_export"]), record_path.parent)
    fit_rows = _payload_rows(fit_path)
    outcome_rows = _payload_rows(outcome_path)
    calibration_ids = {int(value) for value in protocol.get("calibration_episode_ids", [])}
    train_rows = [row for row in fit_rows if int(row["episode_id"]) not in calibration_ids]
    calibration_rows = [row for row in fit_rows if int(row["episode_id"]) in calibration_ids]
    train_states = np.asarray([row["source_state_raw"] for row in train_rows], dtype=np.float64)
    train_target_states_h1 = np.asarray([
        row["latent_by_horizon"]["1"]["target_state_raw"] for row in train_rows
    ], dtype=np.float64)
    direction = _task_direction(train_states, train_target_states_h1)
    train_decoder = {}
    train_center = {}
    for horizon in HORIZONS:
        train_decoder[horizon], train_center[horizon] = _task_readout_with_center(
            np.asarray([row["latent_by_horizon"][str(horizon)]["target"] for row in train_rows], dtype=np.float64),
            train_states,
            np.asarray([row["latent_by_horizon"][str(horizon)]["target_state_raw"] for row in train_rows], dtype=np.float64),
        )
    calibration_decoder = None
    calibration_center = None
    if len(calibration_rows) >= 3:
        calibration_decoder = {}
        calibration_center = {}
        calibration_states = np.asarray([row["source_state_raw"] for row in calibration_rows], dtype=np.float64)
        for horizon in HORIZONS:
            calibration_decoder[horizon], calibration_center[horizon] = _task_readout_with_center(
                np.asarray([row["latent_by_horizon"][str(horizon)]["target"] for row in calibration_rows], dtype=np.float64),
                calibration_states,
                np.asarray([row["latent_by_horizon"][str(horizon)]["target_state_raw"] for row in calibration_rows], dtype=np.float64),
            )
    predicted, actual = _task_arrays(outcome_rows, train_decoder, direction, train_center)
    calibration_predicted = None
    if calibration_decoder is not None:
        calibration_predicted, _ = _task_arrays(outcome_rows, calibration_decoder, direction, calibration_center)
    predicted_matrix = np.stack([predicted[h] for h in HORIZONS], axis=1)
    actual_matrix = np.stack([actual[h] for h in HORIZONS], axis=1)
    selected = np.argmax(predicted_matrix, axis=1)
    actual_best = np.argmax(actual_matrix, axis=1)
    oracle = np.max(actual_matrix, axis=1)
    selected_loss = oracle - actual_matrix[np.arange(len(actual_matrix)), selected]
    fixed = {}
    for index, horizon in enumerate(HORIZONS):
        fixed_loss = oracle - actual_matrix[:, index]
        fixed[str(horizon)] = {
            "mean_loss": float(fixed_loss.mean()),
            "mean_gain_vs_diagnostic": float(fixed_loss.mean() - selected_loss.mean()),
            "selection_count": int(np.sum(selected == index)),
            "actual_best_count": int(np.sum(actual_best == index)),
        }
    horizon_metrics = {}
    for horizon in HORIZONS:
        target = actual[horizon]
        item = {
            "predicted_progress_mean": float(predicted[horizon].mean()),
            "actual_progress_mean": float(target.mean()),
            "prediction_task_pearson": _corr(predicted[horizon], target),
            "prediction_task_spearman": _corr(_rank(predicted[horizon]), _rank(target)),
            "heldout_latent_mse": float(np.mean([
                np.mean((np.asarray(row["latent_by_horizon"][str(horizon)]["predicted"], dtype=np.float64)
                         - np.asarray(row["latent_by_horizon"][str(horizon)]["target"], dtype=np.float64)) ** 2)
                for row in outcome_rows
            ])),
        }
        if calibration_predicted is not None:
            item["calibration_decoder_task_pearson"] = _corr(calibration_predicted[horizon], target)
            item["calibration_decoder_task_spearman"] = _corr(_rank(calibration_predicted[horizon]), _rank(target))
        horizon_metrics[str(horizon)] = item
    episode_gain = np.asarray([fixed["1"]["mean_gain_vs_diagnostic"]])
    return {
        "schema": "independent_planning_failure_diagnostic.v1",
        "record": str(record_path.resolve()),
        "protocol": {
            "checkpoint_selection": "explicit input path",
            "outcome_used_for_selection": False,
            "formal_metric_unchanged": True,
            "task_direction_source": "train-only h=1 source-to-target displacement",
            "diagnostic_only": True,
        },
        "split_sizes": {"train": len(train_rows), "calibration": len(calibration_rows), "outcome": len(outcome_rows)},
        "selected_horizon_counts": {str(h): int(np.sum(selected == index)) for index, h in enumerate(HORIZONS)},
        "actual_best_horizon_counts": {str(h): int(np.sum(actual_best == index)) for index, h in enumerate(HORIZONS)},
        "diagnostic_loss_mean": float(selected_loss.mean()),
        "oracle_progress_mean": float(oracle.mean()),
        "fixed_horizon": fixed,
        "horizon_metrics": horizon_metrics,
        "outcome_gain_distribution": {
            "baseline_horizon_1_minus_diagnostic_mean": float(selected_loss.size and fixed["1"]["mean_gain_vs_diagnostic"]),
            "per_episode_gain_min": float(np.min(oracle - actual_matrix[:, 0] - selected_loss)),
            "per_episode_gain_max": float(np.max(oracle - actual_matrix[:, 0] - selected_loss)),
            "per_episode_gain_positive_fraction": float(np.mean((oracle - actual_matrix[:, 0] - selected_loss) > 0.0)),
        },
        "hypothesis_readout": {
            "train_decoder_available": True,
            "calibration_decoder_available": calibration_decoder is not None,
            "interpretation": "A calibration-only decoder comparison is diagnostic; it cannot replace the frozen formal protocol post hoc.",
        },
        "claim_boundary": "Mechanism decomposition only; no checkpoint, metric, or planner is selected from outcome rows.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"diagnostic_loss_mean": report["diagnostic_loss_mean"], "horizon_metrics": report["horizon_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
