"""Evaluate a train-derived indexed canonical action bank.

The fresh candidate banks are declared shared in goal-relative coordinates,
but their serialized action sequences are root-frame controls.  Reconstructing
the goal frame from a test latent therefore adds an unnecessary and noisy
latent-to-XY adapter.  This audit reconstructs each candidate's canonical
action summary once from a train-root representative and uses only the
candidate id lookup on calibration/outcome roots.

The test-root source state is used only for a post-hoc shared-bank consistency
diagnostic; it is never used to construct test features, fit the task readout,
or select a variant.
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


def _load(dataset_path: Path, arrays_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    arrays = np.load(arrays_path, allow_pickle=False)
    rows = []
    for record in dataset.get("records", []):
        key = str(record["predicted_latent_ref"]).split("::", 1)[-1]
        if key not in arrays.files:
            raise ValueError(f"missing predicted latent array {key}")
        row = dict(record)
        row["_predicted"] = np.asarray(arrays[key], dtype=np.float64)
        if row["_predicted"].ndim != 1 or not bool(np.all(np.isfinite(row["_predicted"]))):
            raise ValueError(f"invalid predicted latent array {key}")
        rows.append(row)
    if not rows:
        raise ValueError("candidate dataset has no records")
    return dict(dataset.get("metadata", {})), rows


def _canonical_summary(row: dict[str, Any]) -> np.ndarray:
    source = np.asarray(row["source_state_raw"], dtype=np.float64).reshape(-1)
    goal = np.asarray(row["goal_raw"], dtype=np.float64).reshape(-1)
    if source.size < 2 or goal.size < 2:
        raise ValueError("source state and goal must contain XY")
    direction = goal[:2] - source[:2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-8:
        raise ValueError("source and goal must be distinct")
    direction = direction / norm
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    actions = np.asarray(row["action_sequence"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("action_sequence must have shape [time, 2]")
    canonical = np.stack([actions @ direction, actions @ perpendicular], axis=1)
    return _action_summary(canonical)


def _fit_indexed_bank(train_rows: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    bank: dict[int, np.ndarray] = {}
    for row in train_rows:
        candidate = int(row["candidate_id"])
        summary = _canonical_summary(row)
        previous = bank.get(candidate)
        if previous is not None and not np.allclose(previous, summary, atol=1.0e-7, rtol=0.0):
            raise ValueError(f"train candidate {candidate} is not a shared canonical action")
        bank[candidate] = summary
    if not bank:
        raise ValueError("train roots do not cover a canonical action bank")
    return bank


def _shared_bank_diagnostic(rows: list[dict[str, Any]], bank: dict[int, np.ndarray]) -> dict[str, Any]:
    deviations = []
    missing = []
    for row in rows:
        candidate = int(row["candidate_id"])
        if candidate not in bank:
            missing.append(candidate)
            continue
        deviations.append(float(np.linalg.norm(_canonical_summary(row) - bank[candidate])))
    values = np.asarray(deviations, dtype=np.float64)
    return {
        "rows_checked": int(values.size),
        "missing_candidate_ids": sorted(set(missing)),
        "max_l2_deviation": None if values.size == 0 else float(np.max(values)),
        "mean_l2_deviation": None if values.size == 0 else float(np.mean(values)),
        "shared_within_tolerance": bool(values.size > 0 and not missing and np.max(values) <= 1.0e-7),
    }


def _features(row: dict[str, Any], bank: dict[int, np.ndarray]) -> np.ndarray:
    candidate = int(row["candidate_id"])
    if candidate not in bank:
        raise ValueError(f"candidate {candidate} is absent from train-derived bank")
    source = np.asarray(row["source_latent_model"], dtype=np.float64)
    predicted = np.asarray(row["_predicted"], dtype=np.float64)
    return np.concatenate([source, predicted - source, bank[candidate]])


def audit(
    candidate_dataset: Path,
    arrays_path: Path,
    *,
    alpha: float = 1.0e-2,
    margin: float = 1.0e-3,
) -> dict[str, Any]:
    metadata, rows = _load(candidate_dataset, arrays_path)
    train = [row for row in rows if str(row["split"]) == "train"]
    calibration = [row for row in rows if str(row["split"]) == "calibration"]
    outcome = [row for row in rows if str(row["split"]) == "outcome"]
    if not train or not calibration or not outcome:
        raise ValueError("train, calibration, and outcome splits are required")
    train_roots = {str(row["instance_id"]) for row in train}
    if train_roots & ({str(row["instance_id"]) for row in calibration} | {str(row["instance_id"]) for row in outcome}):
        raise ValueError("candidate roots must be disjoint across splits")
    bank = _fit_indexed_bank(train)
    features = np.asarray([_features(row, bank) for row in rows], dtype=np.float64)
    train_mask = np.asarray([str(row["split"]) == "train" for row in rows])
    targets = np.asarray([float(row["real_task_cost"]) for row in rows], dtype=np.float64)
    readout = _fit_ridge(features[train_mask], targets[train_mask], alpha)
    scores = _predict(features, readout)
    evaluation = _evaluate(rows, scores, margin)
    return {
        "schema": "gdd_wm_indexed_canonical_action_geometry.v0.1",
        "status": "completed_diagnostic",
        "protocol": {
            "candidate_dataset": str(candidate_dataset.resolve()),
            "arrays": str(arrays_path.resolve()),
            "bank_fit_split": "train",
            "task_readout_fit_split": "train",
            "evaluation_splits": ["calibration", "outcome"],
            "test_action_features": "train_derived_candidate_id_lookup",
            "outcome_used_for_fit_or_feature_selection": False,
            "ridge_alpha": float(alpha),
            "pair_margin": float(margin),
        },
        "bank": {
            "candidate_count": len(bank),
            "candidate_ids": sorted(bank),
            "train_derived_shared_bank": True,
            "all_root_shared_bank_diagnostic": _shared_bank_diagnostic(rows, bank),
        },
        "task_readout": {
            "feature_dimension": int(features.shape[1]),
            "train_cost_fit_rmse": float(np.sqrt(np.mean((_predict(features[train_mask], readout) - targets[train_mask]) ** 2))),
            **evaluation,
        },
        "claim_boundary": (
            "offline train-derived indexed canonical action geometry with simulator-labeled task readout; "
            "not D-JEPA preference validation, physical causality, or closed-loop RGB planning"
        ),
        "input_metadata": metadata,
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
    print(json.dumps({"bank": report["bank"], "task_readout": report["task_readout"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
