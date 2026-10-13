"""Build a frozen baseline-vs-diagnostic planning consequence artifact.

The selected checkpoint is supplied explicitly; this runner never searches
snapshots using outcome loss.  The baseline is a predeclared fixed-horizon
controller and the diagnostic controller selects from the already frozen
candidate bank using predictions only.  True successor states are used solely
by the offline evaluator to score the paired consequence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.empirical_spectral_audit.train_pointmaze_learned_adapter_panel import (
    HORIZONS,
    _task_direction,
    _task_readout_with_center,
)


def _resolve(path_value: str, base: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidates = (base / path, PROJECT_ROOT / path, Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def _payload_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("enriched_records")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} has no enriched_records")
    return rows


def _select_candidate_indices(
    predicted_matrix: np.ndarray,
    permission_details: dict[str, Any],
    baseline_horizon: int,
) -> np.ndarray:
    """Apply the frozen permission decision without consulting outcomes."""
    permission = permission_details.get("planning_permission")
    baseline_index = HORIZONS.index(baseline_horizon)
    if permission == "rejected_fail_closed":
        return np.full(predicted_matrix.shape[0], baseline_index, dtype=np.int64)
    allowed_horizons = permission_details.get("candidate_horizons_allowed", list(HORIZONS))
    allowed_indices = [HORIZONS.index(int(horizon)) for horizon in allowed_horizons if int(horizon) in HORIZONS]
    if not allowed_indices:
        return np.full(predicted_matrix.shape[0], baseline_index, dtype=np.int64)
    masked = np.full_like(predicted_matrix, -np.inf)
    masked[:, allowed_indices] = predicted_matrix[:, allowed_indices]
    return np.argmax(masked, axis=1)


def build_artifact(record_path: Path, *, baseline_horizon: int = 1) -> dict[str, Any]:
    outer = json.loads(record_path.read_text())
    record = outer.get("record", outer)
    protocol = record.get("protocol", {})
    provenance = record.get("provenance", {})
    fit_path = _resolve(str(provenance["fit_export"]), record_path.parent)
    outcome_path = _resolve(str(provenance["outcome_export"]), record_path.parent)
    fit_rows = _payload_rows(fit_path)
    outcome_rows = _payload_rows(outcome_path)
    calibration_ids = {int(value) for value in protocol.get("calibration_episode_ids", [])}
    train_rows = [row for row in fit_rows if int(row["episode_id"]) not in calibration_ids]
    if not train_rows or not outcome_rows:
        raise ValueError("train and outcome rows are required")
    if baseline_horizon not in HORIZONS:
        raise ValueError(f"baseline_horizon must be one of {HORIZONS}")
    train_states = np.asarray([row["source_state_raw"] for row in train_rows], dtype=np.float64)
    train_target_states = np.asarray([
        row["latent_by_horizon"]["4"]["target_state_raw"] for row in train_rows
    ], dtype=np.float64)
    train_target_states_h1 = np.asarray([
        row["latent_by_horizon"]["1"]["target_state_raw"] for row in train_rows
    ], dtype=np.float64)
    direction = _task_direction(train_states, train_target_states_h1)
    decoders = {}
    latent_centers = {}
    target_centers = {}
    for horizon in HORIZONS:
        train_targets = np.asarray([
            row["latent_by_horizon"][str(horizon)]["target"] for row in train_rows
        ], dtype=np.float64)
        train_target_states_h = np.asarray([
            row["latent_by_horizon"][str(horizon)]["target_state_raw"] for row in train_rows
        ], dtype=np.float64)
        decoders[horizon], latent_centers[horizon], target_centers[horizon] = _task_readout_with_center(
            train_targets, train_states, train_target_states_h
        )
    predicted = {}
    actual = {}
    for horizon in HORIZONS:
        predicted_latent = np.asarray([
            row["latent_by_horizon"][str(horizon)]["predicted"] for row in outcome_rows
        ], dtype=np.float64)
        predicted[horizon] = (
            (predicted_latent - latent_centers[horizon]) @ decoders[horizon]
            + target_centers[horizon]
        ) @ direction
        actual[horizon] = np.asarray([
            (np.asarray(row["latent_by_horizon"][str(horizon)]["target_state_raw"], dtype=np.float64)[:2]
             - np.asarray(row["source_state_raw"], dtype=np.float64)[:2]) @ direction
            for row in outcome_rows
        ], dtype=np.float64)
    predicted_matrix = np.stack([predicted[horizon] for horizon in HORIZONS], axis=1)
    actual_matrix = np.stack([actual[horizon] for horizon in HORIZONS], axis=1)
    permission_details = record.get("heldout", {}).get("planning_details_conservative", {})
    permission = permission_details.get("planning_permission")
    selected = _select_candidate_indices(predicted_matrix, permission_details, baseline_horizon)
    oracle = np.max(actual_matrix, axis=1)
    baseline_index = HORIZONS.index(baseline_horizon)
    rows = []
    for index, row in enumerate(outcome_rows):
        rows.append({
            "environment": "PointMaze",
            "episode_id": str(row["episode_id"]),
            "baseline_loss": float(oracle[index] - actual_matrix[index, baseline_index]),
            "diagnostic_loss": float(oracle[index] - actual_matrix[index, selected[index]]),
            "selected_horizon": int(HORIZONS[selected[index]]),
            "baseline_horizon": int(baseline_horizon),
            "protocol": {
                "metric_selection_frozen": True,
                "selector_weights_frozen": True,
                "test_planning_used_for_selection": False,
                "runtime_state_access": False,
                "privileged_successor_verification": False,
                "fresh_planning_episodes": True,
                "planner_frozen": True,
                "independent_outcome_split": True,
                "environment": "PointMaze",
                "metric_fit_split": "train rows only",
                "task_direction_source": "train-only h=1 source-to-target displacement",
                "task_readout_source": "train rows only, horizon-specific",
                "prediction_center_source": "train target latent mean only, horizon-specific",
                "target_displacement_intercept_source": "train target displacement mean only, horizon-specific",
                "outcome_split": "episode-disjoint outcome rows",
                "baseline_rule": f"fixed horizon {baseline_horizon}; frozen before outcome scoring",
                "diagnostic_rule": "argmax predicted task progress over fixed candidate bank",
                "planning_permission": permission or "legacy_record_without_permission_gate",
                "permission_allowed_horizons": permission_details.get("candidate_horizons_allowed", list(HORIZONS)),
                "candidate_bank": list(HORIZONS),
                "physical_state_role": "offline evaluator only",
            },
        })
    gains = np.asarray([row["baseline_loss"] - row["diagnostic_loss"] for row in rows])
    return {
        "schema": "independent_pointmaze_planning_gain_artifact.v1",
        "records": rows,
        "summary": {
            "environment": "PointMaze",
            "baseline_horizon": int(baseline_horizon),
            "candidate_bank": list(HORIZONS),
            "outcome_pair_count": len(rows),
            "mean_gain": float(gains.mean()),
            "diagnostic_loss_mean": float(np.mean([row["diagnostic_loss"] for row in rows])),
            "baseline_loss_mean": float(np.mean([row["baseline_loss"] for row in rows])),
        },
        "provenance": {
            "record": str(record_path.resolve()),
            "fit_export": str(fit_path),
            "outcome_export": str(outcome_path),
            "outcome_used_for_selection": False,
            "checkpoint_selected_by": "explicit CLI record path; no outcome search",
        },
        "claim_boundary": (
            "Paired held-out PointMaze planning consequence from a pre-frozen "
            "baseline and prediction-only diagnostic controller. This is not a "
            "cross-environment claim and does not establish universal visual planning."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-horizon", type=int, default=1)
    args = parser.parse_args()
    report = build_artifact(args.record, baseline_horizon=args.baseline_horizon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
