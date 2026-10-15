"""Compare predeclared train-only feature families for task transfer.

This experiment is a diagnostic control, not a D-JEPA teacher or a planning
run.  Every family is fit on train roots only and evaluated without choosing a
family from calibration/outcome performance.  The comparison asks whether the
current failure is caused by omitting source context/action summaries from an
otherwise usable predicted latent.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FEATURE_FAMILIES = (
    "predicted_terminal",
    "source_plus_terminal",
    "source_plus_delta",
    "source_delta_action",
    "source_delta_estimated_goalframe_action",
    "action_summary",
)


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


def _aggregate_pairs(groups: Iterable[list[dict[str, Any]]], margin: float) -> dict[str, Any]:
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


def _fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
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


def _predict(features: np.ndarray, fit: dict[str, np.ndarray]) -> np.ndarray:
    standardized = (features - fit["mean"]) / fit["scale"]
    design = np.concatenate([np.ones((len(features), 1)), standardized], axis=1)
    return design @ fit["coefficients"]


def _action_summary(action_sequence: Any) -> np.ndarray:
    actions = np.asarray(action_sequence, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] == 0 or actions.shape[0] == 0:
        raise ValueError("action_sequence must be a non-empty rank-2 array")
    midpoint = max(1, actions.shape[0] // 2)
    return np.concatenate(
        [
            np.mean(actions, axis=0),
            np.std(actions, axis=0),
            actions[0],
            actions[-1],
            np.mean(actions[:midpoint], axis=0),
            np.mean(actions[midpoint:], axis=0),
        ]
    )


def _goalframe_action_summary(action_sequence: Any, estimated_source_position: np.ndarray, goal: Any) -> np.ndarray:
    source = np.asarray(estimated_source_position, dtype=np.float64).reshape(-1)
    goal = np.asarray(goal, dtype=np.float64).reshape(-1)
    if source.size < 2 or goal.size < 2:
        raise ValueError("estimated goal-frame action requires two-dimensional source and goal")
    direct = goal[:2] - source[:2]
    norm = float(np.linalg.norm(direct))
    if norm <= 1.0e-8:
        raise ValueError("estimated goal-frame action requires nonzero source-goal displacement")
    direct = direct / norm
    perpendicular = np.asarray([-direct[1], direct[0]], dtype=np.float64)
    actions = np.asarray(action_sequence, dtype=np.float64)
    canonical = np.stack([actions @ direct, actions @ perpendicular], axis=1)
    return _action_summary(canonical)


def _features(
    row: dict[str, Any],
    predicted: np.ndarray,
    family: str,
    estimated_source_position: np.ndarray | None = None,
) -> np.ndarray:
    source = np.asarray(row["source_latent_model"], dtype=np.float64)
    if source.shape != predicted.shape:
        raise ValueError("source and predicted latent dimensions do not match")
    delta = predicted - source
    action = _action_summary(row["action_sequence"])
    if family == "predicted_terminal":
        return predicted
    if family == "source_plus_terminal":
        return np.concatenate([source, predicted])
    if family == "source_plus_delta":
        return np.concatenate([source, delta])
    if family == "source_delta_action":
        return np.concatenate([source, delta, action])
    if family == "source_delta_estimated_goalframe_action":
        if estimated_source_position is None:
            raise ValueError("estimated source position is required for estimated goal-frame action")
        estimated_action = _goalframe_action_summary(
            row["action_sequence"], estimated_source_position, row["goal_raw"]
        )
        return np.concatenate([source, delta, estimated_action])
    if family == "action_summary":
        return action
    raise ValueError(f"unknown feature family: {family}")


def _evaluate(rows: list[dict[str, Any]], scores: np.ndarray, margin: float) -> dict[str, Any]:
    scored = []
    for row, predicted_cost in zip(rows, scores):
        item = dict(row)
        item["teacher_score"] = float(-predicted_cost)
        scored.append(item)
    by_split: dict[str, Any] = {}
    by_split_regret: dict[str, Any] = {}
    root_metrics: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for split in sorted({str(row["split"]) for row in scored}):
        split_rows = [row for row in scored if str(row["split"]) == split]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in split_rows:
            groups[str(row["instance_id"])].append(row)
        regrets = []
        root_metrics[split] = {}
        for root, group in groups.items():
            selected = min(group, key=lambda row: -float(row["teacher_score"]))
            oracle = min(group, key=lambda row: float(row["real_task_cost"]))
            regret = float(selected["real_task_cost"] - oracle["real_task_cost"])
            pair_metrics = _pair_metrics(group, margin)
            regrets.append(regret)
            root_metrics[split][root] = {
                "regret": regret,
                "execution_consistency": pair_metrics["execution_consistency"],
                "comparable_pair_count": pair_metrics["comparable_pair_count"],
            }
        by_split[split] = {
            "rows": len(split_rows),
            "root_count": len(groups),
            "pairs": _aggregate_pairs(groups.values(), margin),
        }
        by_split_regret[split] = {
            "root_count": len(regrets),
            "mean_regret": None if not regrets else float(np.mean(regrets)),
            "zero_regret_root_fraction": None
            if not regrets
            else float(np.mean(np.asarray(regrets) <= margin)),
        }
    return {
        "execution_consistency": by_split,
        "candidate_regret": by_split_regret,
        "root_metrics": root_metrics,
    }


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
    for row in records:
        key = str(row.get("predicted_latent_ref", "")).split("::", 1)[-1]
        if key not in arrays.files:
            raise ValueError(f"missing predicted latent key: {key}")
        predicted = np.asarray(arrays[key], dtype=np.float64)
        if predicted.ndim != 1 or predicted.size == 0 or not np.all(np.isfinite(predicted)):
            raise ValueError(f"invalid predicted latent array: {key}")
        item = dict(row)
        item["_predicted"] = predicted
        rows.append(item)
    train = [row for row in rows if str(row["split"]) == "train"]
    heldout = [row for row in rows if str(row["split"]) != "train"]
    train_roots = {str(row["instance_id"]) for row in train}
    heldout_roots = {str(row["instance_id"]) for row in heldout}
    if not train or not heldout or train_roots & heldout_roots:
        raise ValueError("train/held-out roots must be non-empty and disjoint")
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    results: dict[str, Any] = {}
    target = np.asarray([float(row["real_task_cost"]) for row in rows], dtype=np.float64)
    train_position_rows = {}
    for row in train:
        train_position_rows.setdefault(str(row["instance_id"]), row)
    position_features = np.asarray(
        [np.asarray(row["source_latent_model"], dtype=np.float64) for row in train_position_rows.values()],
        dtype=np.float64,
    )
    position_targets = np.asarray(
        [np.asarray(row["source_state_raw"], dtype=np.float64)[:2] for row in train_position_rows.values()],
        dtype=np.float64,
    )
    position_fit = _fit_ridge(position_features, position_targets, alpha)
    estimated_positions = {
        str(row["instance_id"]): _predict(
            np.asarray([np.asarray(row["source_latent_model"], dtype=np.float64)]), position_fit
        )[0]
        for row in rows
    }
    position_probe_train_rmse = float(
        np.sqrt(np.mean((_predict(position_features, position_fit) - position_targets) ** 2))
    )
    all_position_features = np.asarray(
        [np.asarray(row["source_latent_model"], dtype=np.float64) for row in rows], dtype=np.float64
    )
    all_position_targets = np.asarray(
        [np.asarray(row["source_state_raw"], dtype=np.float64)[:2] for row in rows], dtype=np.float64
    )
    position_probe_heldout_rmse = float(
        np.sqrt(np.mean((_predict(all_position_features, position_fit) - all_position_targets) ** 2))
    )
    for family in FEATURE_FAMILIES:
        all_features = np.asarray(
            [
                _features(
                    row,
                    row["_predicted"],
                    family,
                    estimated_source_position=estimated_positions[str(row["instance_id"])],
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        train_features = all_features[[str(row["split"]) == "train" for row in rows]]
        fit = _fit_ridge(
            train_features,
            target[[str(row["split"]) == "train" for row in rows]],
            alpha,
        )
        predicted_cost = _predict(all_features, fit)
        evaluation = _evaluate(rows, predicted_cost, margin)
        train_predicted = _predict(train_features, fit)
        results[family] = {
            "feature_dimension": int(all_features.shape[1]),
            "train_cost_fit_rmse": float(
                np.sqrt(np.mean((train_predicted - target[[str(row["split"]) == "train" for row in rows]]) ** 2))
            ),
            **evaluation,
        }
    return {
        "schema": "gdd_wm_transfer_sufficient_feature_families.v0.1",
        "status": "completed_diagnostic",
        "claim_boundary": (
            "predeclared train-only feature-family control; not a D-JEPA teacher, "
            "not preference-stability evidence, and not closed-loop planning"
        ),
        "protocol": {
            "candidate_dataset": str(candidate_dataset.resolve()),
            "arrays": str(arrays_path.resolve()),
            "feature_family_order": list(FEATURE_FAMILIES),
            "fit_split": "train",
            "evaluation_splits": ["calibration", "outcome"],
            "ridge_alpha": float(alpha),
            "pair_margin": float(margin),
            "outcome_used_for_fit_or_feature_selection": False,
            "estimated_goalframe_pose_probe_fit_split": "train",
            "estimated_goalframe_pose_probe_target": "source_state_raw_xy",
            "estimated_goalframe_pose_probe_train_rmse": position_probe_train_rmse,
            "estimated_goalframe_pose_probe_all_root_rmse": position_probe_heldout_rmse,
        },
        "summary": {
            "record_count": len(rows),
            "train_root_count": len(train_roots),
            "heldout_root_count": len(heldout_roots),
            "latent_dimension": int(rows[0]["_predicted"].size),
        },
        "feature_families": results,
        "decision": {
            "family_selection": "not_performed",
            "d1_ready": False,
            "d2_ready": False,
        },
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
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
