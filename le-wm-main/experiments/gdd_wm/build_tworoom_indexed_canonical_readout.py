"""Build a train-only indexed candidate readout for live TwoRoom planning.

The candidate identity is the only action geometry used outside the train
roots.  The script fits a ridge readout on
``[source_latent, predicted_delta, train-derived raw-action-summary]`` and
serializes a runtime artifact.  Calibration and outcome labels are checked
for provenance but never read for fitting, model selection, or tolerance
selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gdd_wm.audit_transfer_sufficient_feature_families import (
    _action_summary,
    _fit_ridge,
    _predict,
)


def _load_rows(dataset_path: Path, arrays_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or not isinstance(dataset.get("records"), list):
        raise ValueError("candidate dataset must contain a records list")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        rows: list[dict[str, Any]] = []
        for record in dataset["records"]:
            row = dict(record)
            key = str(row["predicted_latent_ref"]).split("::", 1)[-1]
            if key not in arrays.files:
                raise ValueError(f"missing predicted latent array {key}")
            predicted = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
            source = np.asarray(row["source_latent_model"], dtype=np.float64).reshape(-1)
            if predicted.shape != source.shape or not np.all(np.isfinite(predicted)):
                raise ValueError(f"invalid source/predicted latent pair for {key}")
            row["_predicted"] = predicted
            rows.append(row)
    if not rows:
        raise ValueError("candidate dataset has no records")
    return dict(dataset.get("metadata", {})), rows


def _raw_summary(row: dict[str, Any]) -> np.ndarray:
    actions = np.asarray(row["action_sequence"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 2 or actions.shape[0] == 0:
        raise ValueError("TwoRoom action_sequence must have shape [time, 2]")
    return _action_summary(actions)


def _fit_bank(train_rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    bank: dict[str, np.ndarray] = {}
    for row in train_rows:
        candidate = str(row["candidate_id"])
        summary = _raw_summary(row)
        previous = bank.get(candidate)
        if previous is not None and not np.allclose(previous, summary, atol=1.0e-7, rtol=0.0):
            raise ValueError(f"train candidate {candidate} is not a shared raw action")
        bank[candidate] = summary
    if not bank:
        raise ValueError("train roots do not cover a candidate bank")
    return bank


def _features(
    row: dict[str, Any], bank: dict[str, np.ndarray], goal_conditioning: str,
) -> np.ndarray:
    candidate = str(row["candidate_id"])
    if candidate not in bank:
        raise ValueError(f"candidate {candidate} is absent from train-derived bank")
    source = np.asarray(row["source_latent_model"], dtype=np.float64)
    predicted = np.asarray(row["_predicted"], dtype=np.float64)
    features = np.concatenate([source, predicted - source, bank[candidate]])
    if goal_conditioning == "none":
        return features
    if goal_conditioning != "latent_relative":
        raise ValueError(f"unsupported goal_conditioning {goal_conditioning!r}")
    goal = np.asarray(row.get("goal_latent_model"), dtype=np.float64).reshape(-1)
    if goal.shape != source.shape:
        raise ValueError(
            "latent_relative readout requires goal_latent_model matching source latent"
        )
    return np.concatenate([features, goal - source, predicted - goal])


def _root_regrets(
    rows: list[dict[str, Any]], predictions: np.ndarray, mask: np.ndarray,
) -> dict[str, float]:
    """Return one candidate-selection regret per root.

    This is deliberately a calibration-only selector diagnostic.  Each root is
    one independent candidate set; averaging rows would overweight roots with
    more records and would not match the online ranking problem.
    """
    grouped: dict[str, list[int]] = {}
    for index in np.flatnonzero(mask):
        grouped.setdefault(str(rows[int(index)]["instance_id"]), []).append(int(index))
    regrets: dict[str, float] = {}
    for root, indices in sorted(grouped.items()):
        selected = indices[int(np.argmin(predictions[indices]))]
        costs = np.asarray([float(rows[index]["real_task_cost"]) for index in indices])
        regrets[root] = float(
            float(rows[selected]["real_task_cost"]) - float(np.min(costs))
        )
    return regrets


def _bootstrap_ci(values: np.ndarray, *, seed: int, draws: int = 4000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be a non-empty finite vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def build(
    candidate_dataset: Path,
    arrays_path: Path,
    *,
    alpha: float = 1.0e-2,
    support_tolerance: float = 1.0e-4,
    goal_conditioning: str = "none",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata, rows = _load_rows(candidate_dataset, arrays_path)
    train = [row for row in rows if str(row["split"]) == "train"]
    calibration = [row for row in rows if str(row["split"]) == "calibration"]
    outcome = [row for row in rows if str(row["split"]) == "outcome"]
    if not train or not calibration or not outcome:
        raise ValueError("train, calibration, and outcome splits are required")
    train_roots = {str(row["instance_id"]) for row in train}
    for split_rows in (calibration, outcome):
        if train_roots & {str(row["instance_id"]) for row in split_rows}:
            raise ValueError("train/calibration/outcome roots must be disjoint")
    if support_tolerance <= 0 or not np.isfinite(support_tolerance):
        raise ValueError("support_tolerance must be positive and finite")
    if goal_conditioning not in ("none", "latent_relative"):
        raise ValueError(f"unsupported goal_conditioning {goal_conditioning!r}")

    bank = _fit_bank(train)
    features = np.asarray(
        [_features(row, bank, goal_conditioning) for row in rows], dtype=np.float64,
    )
    action_features = np.asarray(
        [bank[str(row["candidate_id"])] for row in rows], dtype=np.float64,
    )
    train_mask = np.asarray([str(row["split"]) == "train" for row in rows])
    calibration_mask = np.asarray([str(row["split"]) == "calibration" for row in rows])
    targets = np.asarray([float(row["real_task_cost"]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(targets[train_mask])):
        raise ValueError("train task costs must be finite")
    fit = _fit_ridge(features[train_mask], targets[train_mask], alpha)
    train_prediction = _predict(features[train_mask], fit)
    action_fit = _fit_ridge(action_features[train_mask], targets[train_mask], alpha)
    indexed_scores = _predict(features, fit)
    action_scores = _predict(action_features, action_fit)
    indexed_calibration = _root_regrets(rows, indexed_scores, calibration_mask)
    action_calibration = _root_regrets(rows, action_scores, calibration_mask)
    common_roots = sorted(set(indexed_calibration) & set(action_calibration))
    if not common_roots:
        raise ValueError("calibration split has no complete roots for safe gating")
    regret_delta = np.asarray(
        [indexed_calibration[root] - action_calibration[root] for root in common_roots],
        dtype=np.float64,
    )
    regret_delta_ci = _bootstrap_ci(regret_delta, seed=20261016)
    gate_enabled = bool(regret_delta_ci[1] < 0.0)
    candidate_ids = sorted(bank)
    summaries = np.asarray([bank[candidate] for candidate in candidate_ids], dtype=np.float64)
    arrays = {
        "coefficient": fit["coefficients"].astype(np.float32),
        "feature_mean": fit["mean"].astype(np.float32),
        "feature_scale": fit["scale"].astype(np.float32),
        "candidate_ids": np.asarray(candidate_ids),
        "candidate_action_summaries": summaries.astype(np.float32),
        "support_tolerance": np.asarray([support_tolerance], dtype=np.float32),
        "action_frame": np.asarray(["raw_control"]),
        "feature_family": np.asarray(["source_delta_indexed_train_action"]),
        "cost_mode": np.asarray(["bank_rank_affine"]),
        "goal_conditioning": np.asarray([goal_conditioning]),
        "candidate_bank_hash": np.asarray([
            str(metadata.get("candidate_bank_hash", "unknown"))
        ]),
        "ridge_alpha": np.asarray([alpha], dtype=np.float32),
        "gate_enabled": np.asarray([int(gate_enabled)], dtype=np.int8),
    }
    report = {
        "schema": "tworoom_indexed_canonical_readout.v0.1",
        "status": "completed_train_only_fit",
        "protocol": {
            "candidate_dataset": str(candidate_dataset.resolve()),
            "arrays": str(arrays_path.resolve()),
            "fit_split": "train",
            "evaluation_splits": ["calibration", "outcome"],
            "root_disjoint": True,
            "outcome_used_for_fit_or_selection": False,
            "candidate_identity": "train_derived_candidate_id_lookup",
            "action_frame": "raw_control",
            "feature_family": "source_delta_indexed_train_action",
            "goal_conditioning": goal_conditioning,
            "cost_mode": "bank_rank_affine",
            "ridge_alpha": float(alpha),
            "support_tolerance": float(support_tolerance),
            "safe_gate_rule": (
                "enable indexed readout only when calibration root-macro regret "
                "delta versus action-only has bootstrap CI95 upper bound < 0"
            ),
        },
        "rows": {
            "train": len(train),
            "calibration": len(calibration),
            "outcome": len(outcome),
        },
        "roots": {
            "train": len({str(row["instance_id"]) for row in train}),
            "calibration": len({str(row["instance_id"]) for row in calibration}),
            "outcome": len({str(row["instance_id"]) for row in outcome}),
        },
        "candidate_bank": {
            "count": len(candidate_ids),
            "candidate_ids": candidate_ids,
            "hash": str(metadata.get("candidate_bank_hash", "unknown")),
            "train_shared_raw_action": True,
        },
        "fit": {
            "feature_dimension": int(features.shape[1]),
            "train_cost_fit_rmse": float(np.sqrt(np.mean((train_prediction - targets[train_mask]) ** 2))),
        },
        "selection": {
            "baseline": "action_only_train_ridge",
            "calibration_root_count": len(common_roots),
            "indexed_mean_regret": float(np.mean([indexed_calibration[root] for root in common_roots])),
            "action_only_mean_regret": float(np.mean([action_calibration[root] for root in common_roots])),
            "indexed_minus_action_only_regret_delta": {
                "mean": float(np.mean(regret_delta)),
                "ci95": regret_delta_ci,
            },
            "gate_enabled": gate_enabled,
            "fallback": not gate_enabled,
            "fallback_target": "frozen_base_cost",
            "outcome_used_for_selection": False,
        },
        "runtime_contract": {
            "bank_mask_required": True,
            "unsupported_gaussian_behavior": "frozen_base_cost",
            "unknown_bank_action_behavior": "frozen_base_cost",
            "outcome_labels_loaded_at_runtime": False,
            "gate_disabled_behavior": "frozen_base_cost",
        },
        "claim_boundary": (
            "train-only indexed candidate readout for an RGB-compatible live intervention; "
            "not a universal world-model or causal-dynamics certificate"
        ),
        "input_metadata": metadata,
    }
    return report, arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", required=True, type=Path)
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=1.0e-2)
    parser.add_argument("--support-tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--goal-conditioning", choices=("none", "latent_relative"), default="none",
    )
    args = parser.parse_args()
    report, arrays = build(
        args.candidate_dataset,
        args.arrays,
        alpha=args.alpha,
        support_tolerance=args.support_tolerance,
        goal_conditioning=args.goal_conditioning,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    report["artifact"] = str(args.output.with_suffix(".npz").resolve())
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
