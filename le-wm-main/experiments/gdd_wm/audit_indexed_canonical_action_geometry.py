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


SHORTCUT_CONTROL_FAMILIES = (
    "full_source_delta_bank",
    "bank_only",
    "no_dynamics_source_bank",
    "source_plus_delta",
    "shuffled_prediction_full",
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


def _bank_summary(row: dict[str, Any], coordinate_frame: str) -> np.ndarray:
    if coordinate_frame == "goal_relative":
        return _canonical_summary(row)
    if coordinate_frame == "world_fixed":
        return _action_summary(row["action_sequence"])
    raise ValueError(
        "indexed action audit requires goal_relative or world_fixed bank, got "
        f"{coordinate_frame}"
    )


def _fit_indexed_bank(
    train_rows: list[dict[str, Any]],
    coordinate_frame: str = "goal_relative",
) -> dict[int, np.ndarray]:
    bank: dict[int, np.ndarray] = {}
    for row in train_rows:
        candidate = int(row["candidate_id"])
        summary = _bank_summary(row, coordinate_frame)
        previous = bank.get(candidate)
        if previous is not None and not np.allclose(previous, summary, atol=1.0e-7, rtol=0.0):
            raise ValueError(f"train candidate {candidate} is not a shared canonical action")
        bank[candidate] = summary
    if not bank:
        raise ValueError("train roots do not cover a canonical action bank")
    return bank


def _shared_bank_diagnostic(
    rows: list[dict[str, Any]],
    bank: dict[int, np.ndarray],
    coordinate_frame: str = "goal_relative",
) -> dict[str, Any]:
    deviations = []
    missing = []
    for row in rows:
        candidate = int(row["candidate_id"])
        if candidate not in bank:
            missing.append(candidate)
            continue
        deviations.append(float(np.linalg.norm(_bank_summary(row, coordinate_frame) - bank[candidate])))
    values = np.asarray(deviations, dtype=np.float64)
    return {
        "rows_checked": int(values.size),
        "missing_candidate_ids": sorted(set(missing)),
        "max_l2_deviation": None if values.size == 0 else float(np.max(values)),
        "mean_l2_deviation": None if values.size == 0 else float(np.mean(values)),
        "shared_within_tolerance": bool(values.size > 0 and not missing and np.max(values) <= 1.0e-7),
    }


def _features(
    row: dict[str, Any],
    bank: dict[int, np.ndarray],
    *,
    family: str = "full_source_delta_bank",
    predicted_override: np.ndarray | None = None,
) -> np.ndarray:
    candidate = int(row["candidate_id"])
    if candidate not in bank:
        raise ValueError(f"candidate {candidate} is absent from train-derived bank")
    source = np.asarray(row["source_latent_model"], dtype=np.float64)
    predicted = np.asarray(
        row["_predicted"] if predicted_override is None else predicted_override,
        dtype=np.float64,
    )
    if source.shape != predicted.shape:
        raise ValueError("source and predicted latent dimensions do not match")
    action = bank[candidate]
    if family in {"full_source_delta_bank", "shuffled_prediction_full"}:
        return np.concatenate([source, predicted - source, action])
    if family == "bank_only":
        return action.copy()
    if family == "no_dynamics_source_bank":
        return np.concatenate([source, action])
    if family == "source_plus_delta":
        return np.concatenate([source, predicted - source])
    raise ValueError(f"unknown shortcut-control family {family}")


def _cyclic_prediction_shuffle(rows: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    """Break candidate/prediction coupling without crossing roots or splits."""

    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        key = (str(row["split"]), str(row["instance_id"]))
        grouped.setdefault(key, []).append((index, row))
    shuffled: dict[int, np.ndarray] = {}
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda item: int(item[1]["candidate_id"]))
        if len(ordered) < 2:
            raise ValueError(f"prediction shuffle needs at least two candidates in {key}")
        predictions = [np.asarray(row["_predicted"], dtype=np.float64) for _, row in ordered]
        for position, (index, _row) in enumerate(ordered):
            shuffled[index] = predictions[(position + 1) % len(predictions)]
    if len(shuffled) != len(rows):
        raise RuntimeError("prediction shuffle did not cover every candidate row")
    return shuffled


def _fit_and_evaluate(
    rows: list[dict[str, Any]],
    features: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    alpha: float,
    margin: float,
) -> dict[str, Any]:
    fit = _fit_ridge(features[train_mask], targets[train_mask], alpha)
    train_predictions = _predict(features[train_mask], fit)
    return {
        "feature_dimension": int(features.shape[1]),
        "train_cost_fit_rmse": float(
            np.sqrt(np.mean((train_predictions - targets[train_mask]) ** 2))
        ),
        **_evaluate(rows, _predict(features, fit), margin),
    }


def _unseen_candidate_audit(
    rows: list[dict[str, Any]],
    bank: dict[int, np.ndarray],
    targets: np.ndarray,
    alpha: float,
    margin: float,
) -> dict[str, Any]:
    """Evaluate continuous action features on candidate IDs absent from fit."""

    candidate_ids = sorted(bank)
    if len(candidate_ids) < 8:
        raise ValueError("unseen-candidate audit requires at least eight candidates")
    heldout_ids = [candidate for index, candidate in enumerate(candidate_ids) if index % 4 == 3]
    fit_ids = [candidate for candidate in candidate_ids if candidate not in set(heldout_ids)]
    fit_mask = np.asarray([
        str(row["split"]) == "train" and int(row["candidate_id"]) in fit_ids
        for row in rows
    ])
    evaluation_mask = np.asarray([
        str(row["split"]) in {"calibration", "outcome"}
        and int(row["candidate_id"]) in heldout_ids
        for row in rows
    ])
    evaluation_rows = [row for row, keep in zip(rows, evaluation_mask) if keep]
    evaluation_targets = targets[evaluation_mask]
    families = {}
    for family in ("full_source_delta_bank", "bank_only", "no_dynamics_source_bank", "source_plus_delta"):
        features = np.asarray([_features(row, bank, family=family) for row in rows], dtype=np.float64)
        fit = _fit_ridge(features[fit_mask], targets[fit_mask], alpha)
        evaluation_features = features[evaluation_mask]
        families[family] = {
            "feature_dimension": int(features.shape[1]),
            "fit_rows": int(np.sum(fit_mask)),
            "evaluation_rows": int(np.sum(evaluation_mask)),
            **_evaluate(evaluation_rows, _predict(evaluation_features, fit), margin),
        }
    return {
        "partition_rule": "sorted_candidate_index_mod_4_equals_3_is_unseen",
        "fit_candidate_ids": fit_ids,
        "unseen_candidate_ids": heldout_ids,
        "fit_split": "train",
        "evaluation_splits": ["calibration", "outcome"],
        "outcome_used_for_partition_or_fit": False,
        "feature_families": families,
        "evaluation_target_count": int(evaluation_targets.size),
        "claim_boundary": (
            "held-out candidate-ID/action-template interpolation within the declared canonical bank; "
            "not adaptive CEM or arbitrary continuous-action generalization"
        ),
    }


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
    bank_coordinate_frame = str(metadata.get("candidate_bank_coordinate_frame", "goal_relative"))
    if not bool(metadata.get("shared_candidate_bank", False)):
        raise ValueError("indexed action audit requires metadata.shared_candidate_bank=true")
    bank = _fit_indexed_bank(train, bank_coordinate_frame)
    train_mask = np.asarray([str(row["split"]) == "train" for row in rows])
    targets = np.asarray([float(row["real_task_cost"]) for row in rows], dtype=np.float64)
    shuffled_predictions = _cyclic_prediction_shuffle(rows)
    shortcut_controls: dict[str, Any] = {}
    for family in SHORTCUT_CONTROL_FAMILIES:
        features = np.asarray(
            [
                _features(
                    row,
                    bank,
                    family=family,
                    predicted_override=(
                        shuffled_predictions[index]
                        if family == "shuffled_prediction_full"
                        else None
                    ),
                )
                for index, row in enumerate(rows)
            ],
            dtype=np.float64,
        )
        shortcut_controls[family] = _fit_and_evaluate(
            rows, features, targets, train_mask, alpha, margin,
        )
    evaluation = shortcut_controls["full_source_delta_bank"]
    return {
        "schema": "gdd_wm_indexed_canonical_action_geometry.v0.2",
        "status": "completed_diagnostic",
        "protocol": {
            "candidate_dataset": str(candidate_dataset.resolve()),
            "arrays": str(arrays_path.resolve()),
            "bank_fit_split": "train",
            "task_readout_fit_split": "train",
            "evaluation_splits": ["calibration", "outcome"],
            "test_action_features": "train_derived_candidate_id_lookup",
            "candidate_bank_coordinate_frame": bank_coordinate_frame,
            "outcome_used_for_fit_or_feature_selection": False,
            "ridge_alpha": float(alpha),
            "pair_margin": float(margin),
        },
        "bank": {
            "candidate_count": len(bank),
            "candidate_ids": sorted(bank),
            "train_derived_shared_bank": True,
            "all_root_shared_bank_diagnostic": _shared_bank_diagnostic(
                rows, bank, bank_coordinate_frame,
            ),
        },
        "task_readout": {
            **evaluation,
        },
        "shortcut_controls": shortcut_controls,
        "prediction_shuffle": {
            "scheme": "within_split_within_root_candidate_id_cyclic_shift_by_one",
            "cross_root_mixing": False,
            "cross_split_mixing": False,
        },
        "unseen_candidate_audit": _unseen_candidate_audit(
            rows, bank, targets, alpha, margin,
        ),
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
