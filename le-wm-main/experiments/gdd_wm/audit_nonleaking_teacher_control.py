"""Audit a train-only predicted-latent task-readout control.

This is deliberately not a D-JEPA teacher.  It is a diagnostic control for
the current GDD-WM negative result: fit a fixed ridge readout on train-root
predicted latents and evaluate its candidate ranking on calibration/outcome
roots.  No evaluation cost is used to fit, select, or tune the readout.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("candidate dataset must be a JSON object with records")
    return dict(payload.get("metadata", {})), list(payload["records"])


def _sign(value: float, margin: float) -> int:
    if abs(float(value)) <= margin:
        return 0
    return 1 if value > 0.0 else -1


def _pair_metrics(rows: list[dict[str, Any]], margin: float) -> dict[str, Any]:
    total = comparable = consistent = inversions = 0
    for left_index in range(len(rows)):
        for right_index in range(left_index + 1, len(rows)):
            total += 1
            left, right = rows[left_index], rows[right_index]
            teacher_sign = _sign(
                float(left["teacher_score"]) - float(right["teacher_score"]), margin
            )
            real_sign = _sign(
                float(right["real_task_cost"]) - float(left["real_task_cost"]), margin
            )
            if teacher_sign == 0 or real_sign == 0:
                continue
            comparable += 1
            if teacher_sign == real_sign:
                consistent += 1
            else:
                inversions += 1
    return {
        "pair_count": total,
        "comparable_pair_count": comparable,
        "execution_consistent_pair_count": consistent,
        "execution_inversion_count": inversions,
        "execution_consistency": None if comparable == 0 else consistent / comparable,
    }


def _fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> dict[str, Any]:
    mean = np.mean(features, axis=0)
    scale = np.maximum(np.std(features, axis=0), 1.0e-6)
    standardized = (features - mean) / scale
    design = np.concatenate([np.ones((len(features), 1)), standardized], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + float(alpha) * penalty,
        design.T @ target,
    )
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def _predict(features: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    standardized = (features - fit["mean"]) / fit["scale"]
    design = np.concatenate([np.ones((len(features), 1)), standardized], axis=1)
    return design @ fit["coefficients"]


def audit(
    candidate_dataset: Path,
    arrays_path: Path,
    *,
    alpha: float = 1.0e-2,
    margin: float = 1.0e-3,
) -> dict[str, Any]:
    metadata, records = _load(candidate_dataset)
    arrays = np.load(arrays_path, allow_pickle=False)
    rows: list[dict[str, Any]] = []
    missing_keys: list[str] = []
    for row in records:
        ref = str(row.get("predicted_latent_ref", ""))
        key = ref.split("::", 1)[-1]
        if key not in arrays.files:
            missing_keys.append(key)
            continue
        value = np.asarray(arrays[key], dtype=np.float64)
        if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid predicted latent array: {key}")
        item = dict(row)
        item["_features"] = value
        rows.append(item)
    if missing_keys:
        raise ValueError(f"missing predicted latent keys: {missing_keys[:5]}")
    if not rows:
        raise ValueError("no rows")
    dimensions = {int(row["_features"].size) for row in rows}
    if len(dimensions) != 1:
        raise ValueError(f"predicted latent dimensions are heterogeneous: {dimensions}")
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    if margin < 0.0 or not np.isfinite(margin):
        raise ValueError("margin must be finite and non-negative")

    train = [row for row in rows if str(row["split"]) == "train"]
    evaluation = [row for row in rows if str(row["split"]) != "train"]
    if not train or not evaluation:
        raise ValueError("both train and held-out rows are required")
    train_roots = {str(row["instance_id"]) for row in train}
    heldout_roots = {str(row["instance_id"]) for row in evaluation}
    if train_roots & heldout_roots:
        raise ValueError("train and held-out roots overlap")

    train_x = np.asarray([row["_features"] for row in train], dtype=np.float64)
    train_y = np.asarray([float(row["real_task_cost"]) for row in train], dtype=np.float64)
    fit = _fit_ridge(train_x, train_y, alpha)
    all_x = np.asarray([row["_features"] for row in rows], dtype=np.float64)
    predicted_cost = _predict(all_x, fit)
    for row, cost in zip(rows, predicted_cost):
        row["teacher_score"] = float(-cost)

    by_split: dict[str, dict[str, Any]] = {}
    by_split_regret: dict[str, dict[str, Any]] = {}
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if str(row["split"]) == split]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in split_rows:
            groups[str(row["instance_id"])].append(row)
        by_split[split] = {
            "rows": len(split_rows),
            "root_count": len(groups),
            "pairs": _aggregate_pairs(groups.values(), margin),
        }
        regrets = []
        for root, group in groups.items():
            selected = min(group, key=lambda row: -float(row["teacher_score"]))
            oracle = min(group, key=lambda row: float(row["real_task_cost"]))
            regrets.append({
                "instance_id": root,
                "selected_candidate_id": int(selected["candidate_id"]),
                "oracle_candidate_id": int(oracle["candidate_id"]),
                "selected_cost": float(selected["real_task_cost"]),
                "oracle_cost": float(oracle["real_task_cost"]),
                "regret": float(selected["real_task_cost"] - oracle["real_task_cost"]),
            })
        by_split_regret[split] = {
            "root_count": len(regrets),
            "mean_regret": None if not regrets else float(np.mean([r["regret"] for r in regrets])),
            "zero_regret_root_fraction": None
            if not regrets
            else float(np.mean([r["regret"] <= margin for r in regrets])),
            "roots": regrets,
        }

    # ``records`` are not required to be split-sorted; recompute on the exact
    # train matrix rather than slicing the global prediction vector.
    train_pred = _predict(train_x, fit)
    report = {
        "schema": "gdd_wm_nonleaking_teacher_control.v0.1",
        "status": "completed_diagnostic",
        "claim_boundary": (
            "train-only predicted-latent readout control; not a D-JEPA teacher, "
            "not D1 preference-stability evidence, and not planning evidence"
        ),
        "protocol": {
            "candidate_dataset": str(candidate_dataset.resolve()),
            "arrays": str(arrays_path.resolve()),
            "fit_split": "train",
            "evaluation_splits": ["calibration", "outcome"],
            "model": "standardized_ridge_predicted_latent_to_real_task_cost",
            "ridge_alpha": float(alpha),
            "pair_margin": float(margin),
            "outcome_used_for_fit_or_selection": False,
            "teacher_model_id": "train_only_predicted_latent_ridge_v1",
        },
        "summary": {
            "record_count": len(rows),
            "latent_dimension": next(iter(dimensions)),
            "train_root_count": len(train_roots),
            "heldout_root_count": len(heldout_roots),
            "train_cost_fit_rmse": float(np.sqrt(np.mean((train_pred - train_y) ** 2))),
            "teacher_score_exact_negative_real_cost": bool(
                np.allclose(predicted_cost + np.asarray([float(row["real_task_cost"]) for row in rows]), 0.0)
            ),
        },
        "execution_consistency": by_split,
        "heldout_candidate_regret": by_split_regret,
        "decision": {
            "nonleaking_control_available": True,
            "d1_preference_stability": "not_estimable_from_single_context_per_root",
            "d1_ready": False,
            "d2_ready": False,
        },
        "input_metadata": metadata,
    }
    return report


def _aggregate_pairs(groups: Any, margin: float) -> dict[str, Any]:
    totals = {
        "pair_count": 0,
        "comparable_pair_count": 0,
        "execution_consistent_pair_count": 0,
        "execution_inversion_count": 0,
    }
    for group in groups:
        metrics = _pair_metrics(list(group), margin)
        for key in totals:
            totals[key] += int(metrics[key])
    comparable = totals["comparable_pair_count"]
    totals["execution_consistency"] = None if comparable == 0 else (
        totals["execution_consistent_pair_count"] / comparable
    )
    return totals


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
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
