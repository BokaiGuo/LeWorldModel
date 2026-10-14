"""Test train-only RGB successor calibration before candidate selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gdd_wm.audit_transfer_sufficient_feature_families import (
    _evaluate,
    _features,
    _fit_ridge,
    _predict,
)


FAMILIES = (
    "source_plus_delta",
    "source_delta_action",
    "source_delta_estimated_goalframe_action",
)


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("renderer_consistency", "realized_rgb_latents_available", "pca_projected_successor_comparison")
    if not all(bool(payload.get("gates", {}).get(name)) for name in required):
        raise ValueError(f"RGB report gates are not all passed: {path}")
    return payload


def _report_rows(report: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["instance_id"]), int(row["candidate_id"])): row
        for row in report.get("rows", [])
        if str(row.get("split")) in {"train", "calibration", "outcome"}
    }


def _fit_position_probe(rows: list[dict[str, Any]], alpha: float) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    unique = {}
    for row in rows:
        unique.setdefault(str(row["instance_id"]), row)
    features = np.asarray([row["source_latent_model"] for row in unique.values()], dtype=np.float64)
    targets = np.asarray([np.asarray(row["source_state_raw"], dtype=np.float64)[:2] for row in unique.values()])
    fit = _fit_ridge(features, targets, alpha)
    positions = {
        str(row["instance_id"]): _predict(np.asarray([row["source_latent_model"]], dtype=np.float64), fit)[0]
        for row in rows
    }
    return fit, positions


def audit(
    candidate_dataset: Path,
    arrays_path: Path,
    train_report_path: Path,
    calibration_report_path: Path,
    outcome_report_path: Path,
    *,
    alpha: float = 1.0e-2,
    margin: float = 1.0e-3,
) -> dict[str, Any]:
    dataset = json.loads(candidate_dataset.read_text(encoding="utf-8"))
    records = [dict(row) for row in dataset["records"]]
    arrays = np.load(arrays_path, allow_pickle=False)
    train_report = _load_report(train_report_path)
    calibration_report = _load_report(calibration_report_path)
    outcome_report = _load_report(outcome_report_path)
    report_maps = {
        "train": _report_rows(train_report),
        "calibration": _report_rows(calibration_report),
        "outcome": _report_rows(outcome_report),
    }
    rows = []
    for row in records:
        key = str(row["predicted_latent_ref"]).split("::", 1)[-1]
        if key not in arrays.files:
            raise ValueError(f"missing predicted array {key}")
        item = dict(row)
        item["_predicted"] = np.asarray(arrays[key], dtype=np.float64)
        rows.append(item)
    train_rows = [row for row in rows if str(row["split"]) == "train"]
    if not train_rows:
        raise ValueError("train rows are required")

    # Fit the successor calibration only on train-root realized RGB references.
    train_pairs = []
    train_realized = []
    for row in train_rows:
        key = (str(row["instance_id"]), int(row["candidate_id"]))
        rgb = report_maps["train"].get(key)
        if rgb is None:
            raise ValueError(f"missing train realized RGB row {key}")
        train_pairs.append(np.asarray(rgb["predicted_terminal_projection"], dtype=np.float64))
        train_realized.append(np.asarray(rgb["realized_terminal_projection"], dtype=np.float64))
    successor_fit = _fit_ridge(np.asarray(train_pairs), np.asarray(train_realized), alpha)
    calibrated = {
        id(row): _predict(np.asarray([row["_predicted"]]), successor_fit)[0]
        for row in rows
    }
    uncalibrated = {id(row): row["_predicted"] for row in rows}
    position_fit, _ = _fit_position_probe(train_rows, alpha)
    estimated_positions = {
        str(row["instance_id"]): _predict(
            np.asarray([row["source_latent_model"]], dtype=np.float64), position_fit
        )[0]
        for row in rows
    }

    results = {}
    target = np.asarray([float(row["real_task_cost"]) for row in rows], dtype=np.float64)
    train_mask = np.asarray([str(row["split"]) == "train" for row in rows])
    for calibration_name, predictions in (
        ("uncalibrated", uncalibrated),
        ("rgb_successor_calibrated", calibrated),
    ):
        for family in FAMILIES:
            features = np.asarray(
                [
                    _features(
                        row,
                        predictions[id(row)],
                        family,
                        estimated_source_position=estimated_positions[str(row["instance_id"])],
                    )
                    for row in rows
                ],
                dtype=np.float64,
            )
            fit = _fit_ridge(features[train_mask], target[train_mask], alpha)
            scores = _predict(features, fit)
            results[f"{calibration_name}::{family}"] = {
                "feature_dimension": int(features.shape[1]),
                "train_cost_fit_rmse": float(np.sqrt(np.mean((_predict(features[train_mask], fit) - target[train_mask]) ** 2))),
                **_evaluate(rows, scores, margin),
            }

    successor_train_error = np.asarray(train_pairs) - np.asarray(train_realized)
    report = {
        "schema": "gdd_wm_rgb_calibrated_candidate_selection.v0.1",
        "status": "completed_diagnostic",
        "protocol": {
            "candidate_dataset": str(candidate_dataset.resolve()),
            "arrays": str(arrays_path.resolve()),
            "train_rgb_report": str(train_report_path.resolve()),
            "calibration_rgb_report": str(calibration_report_path.resolve()),
            "outcome_rgb_report": str(outcome_report_path.resolve()),
            "successor_calibration_fit_split": "train",
            "task_readout_fit_split": "train",
            "evaluation_splits": ["calibration", "outcome"],
            "outcome_used_for_fit_or_selection": False,
            "ridge_alpha": float(alpha),
            "pair_margin": float(margin),
        },
        "successor_calibration": {
            "train_rows": len(train_pairs),
            "train_rmse": float(np.sqrt(np.mean(successor_train_error ** 2))),
            "source_to_realized_rmse": float(
                np.sqrt(np.mean((np.asarray([row["source_projection"] for row in train_report["rows"]]) - np.asarray(train_realized)) ** 2))
            ) if train_report.get("rows") else None,
            "position_probe_fit_split": "train",
            "position_probe_train_rmse": float(
                np.sqrt(np.mean((_predict(np.asarray([row["source_latent_model"] for row in train_rows]), position_fit) - np.asarray([row["source_state_raw"][:2] for row in train_rows])) ** 2))
            ),
        },
        "results": results,
        "claim_boundary": (
            "train-only realized-RGB successor calibration followed by offline candidate selection; "
            "not D-JEPA preference validation, physical causality, or closed-loop planning"
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--outcome-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0e-2)
    parser.add_argument("--margin", type=float, default=1.0e-3)
    args = parser.parse_args()
    report = audit(
        args.candidate_dataset,
        args.arrays,
        args.train_report,
        args.calibration_report,
        args.outcome_report,
        alpha=args.alpha,
        margin=args.margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "protocol": report["protocol"], "successor_calibration": report["successor_calibration"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
