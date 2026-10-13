"""Recompute PointMaze held-out planning fields after the readout bug fix.

This script does not retrain adapters, alter diagnostics, choose checkpoints,
or inspect outcome losses for selection.  It only replaces the old planning
readout evaluation with horizon-specific readouts and train-only latent
centers, preserving the frozen exports and every original record field.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_independent_pointmaze_planning_gain import _payload_rows, _resolve
from train_pointmaze_learned_adapter_panel import HORIZONS, _task_direction, _task_readout_with_center


def _repair(record_path: Path) -> dict[str, Any]:
    payload = json.loads(record_path.read_text())
    record = copy.deepcopy(payload["record"])
    protocol = record["protocol"]
    provenance = record["provenance"]
    fit_path = _resolve(str(provenance["fit_export"]), record_path.parent)
    outcome_path = _resolve(str(provenance["outcome_export"]), record_path.parent)
    fit_rows = _payload_rows(fit_path)
    outcome_rows = _payload_rows(outcome_path)
    calibration_ids = {int(value) for value in protocol.get("calibration_episode_ids", [])}
    train_rows = [row for row in fit_rows if int(row["episode_id"]) not in calibration_ids]
    train_states = np.asarray([row["source_state_raw"] for row in train_rows], dtype=np.float64)
    train_target_states_h1 = np.asarray([
        row["latent_by_horizon"]["1"]["target_state_raw"] for row in train_rows
    ], dtype=np.float64)
    direction = _task_direction(train_states, train_target_states_h1)
    decoders = {}
    centers = {}
    for horizon in HORIZONS:
        decoders[horizon], centers[horizon] = _task_readout_with_center(
            np.asarray([row["latent_by_horizon"][str(horizon)]["target"] for row in train_rows], dtype=np.float64),
            train_states,
            np.asarray([row["latent_by_horizon"][str(horizon)]["target_state_raw"] for row in train_rows], dtype=np.float64),
        )
    predicted = {}
    actual = {}
    for horizon in HORIZONS:
        predicted_latent = np.asarray([
            row["latent_by_horizon"][str(horizon)]["predicted"] for row in outcome_rows
        ], dtype=np.float64)
        predicted[horizon] = (predicted_latent - centers[horizon]) @ decoders[horizon] @ direction
        actual[horizon] = np.asarray([
            (np.asarray(row["latent_by_horizon"][str(horizon)]["target_state_raw"], dtype=np.float64)[:2]
             - np.asarray(row["source_state_raw"], dtype=np.float64)[:2]) @ direction
            for row in outcome_rows
        ], dtype=np.float64)
    predicted_matrix = np.stack([predicted[horizon] for horizon in HORIZONS], axis=1)
    actual_matrix = np.stack([actual[horizon] for horizon in HORIZONS], axis=1)
    selected = np.argmax(predicted_matrix, axis=1)
    chosen = actual_matrix[np.arange(len(actual_matrix)), selected]
    oracle = np.max(actual_matrix, axis=1)
    planning_risk = float(np.mean(oracle - chosen))
    record["heldout"]["planning_risk"] = planning_risk
    record["heldout"]["planning_details"] = {
        "candidate_horizons": list(HORIZONS),
        "selected_horizon_counts": {str(h): int(np.sum(selected == index)) for index, h in enumerate(HORIZONS)},
        "mean_chosen_progress": float(np.mean(chosen)),
        "mean_oracle_progress": float(np.mean(oracle)),
        "direction": direction.tolist(),
        "task_readout_source": "train rows only, horizon-specific",
        "task_direction_source": "train-only h=1 source-to-target displacement",
        "prediction_center_source": "train target latent mean only, horizon-specific",
        "evaluation_repair": "replaced outcome-batch centering and shared-horizon readout; no model retraining",
    }
    record["protocol"].update({
        "task_readout_source": "train rows only, horizon-specific",
        "prediction_center_source": "train target latent mean only, horizon-specific",
        "planning_evaluation_repair": "20261013_readout_center_and_horizon_alignment",
        "outcome_used_for_selection": False,
    })
    return {"record_schema": "real_checkpoint_selection.v1", "record": record, "protocol": record["protocol"]}


def repair(input_root: Path, output_root: Path) -> dict[str, Any]:
    paths = sorted(input_root.rglob("checkpoint_record.json"))
    if not paths:
        raise ValueError(f"no checkpoint records under {input_root}")
    output_paths = []
    failures = []
    for path in paths:
        try:
            relative = path.relative_to(input_root)
            target = output_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(_repair(path), indent=2, sort_keys=True) + "\n")
            output_paths.append(str(target))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            failures.append({"path": str(path), "error": str(exc)})
    manifest = {
        "schema": "pointmaze_readout_repair_panel.v1",
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "record_count": len(output_paths),
        "failure_count": len(failures),
        "records": output_paths,
        "protocol": {
            "retrained": False,
            "diagnostic_metrics_changed": False,
            "heldout_outcome_used_for_selection": False,
            "repair": "train-only horizon-specific task readout and train-only latent center",
        },
        "failures": failures,
        "claim_boundary": "Evaluation repair artifact; scientific claims remain subject to independent matched and planning gates.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = repair(args.input_root, args.output_root)
    print(json.dumps({key: report[key] for key in ("record_count", "failure_count", "output_root")}, indent=2))
    if report["failure_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
