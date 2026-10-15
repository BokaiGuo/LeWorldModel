"""Audit root-centered and interaction-aware task-transfer geometry.

The earlier controls fit absolute task cost, which can spend capacity on
root-level difficulty instead of candidate-level action effect.  This audit
freezes a small set of relative-effect models: train-root-centered targets,
source/delta/action interactions, and one absolute baseline.  No held-out
root is used for fitting or variant selection.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VARIANTS = (
    "absolute_predicted_terminal",
    "root_centered_predicted_terminal",
    "root_centered_source_delta",
    "root_centered_source_delta_action",
    "root_centered_source_delta_action_interaction",
    "root_centered_source_delta_action_interaction_lr2",
    "root_centered_source_delta_action_interaction_lr4",
    "root_centered_source_delta_action_interaction_lr8",
    "root_centered_source_delta_action_interaction_lr16",
    "root_centered_source_delta_action_pairwise",
    "root_centered_source_delta_action_interaction_lr4_pairwise",
    "root_centered_source_delta_action_interaction_lr8_pairwise",
    "root_centered_source_delta_goalframe_action",
    "root_centered_source_delta_goalframe_action_pairwise",
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


def _pairs(groups: Iterable[list[dict[str, Any]]], margin: float) -> dict[str, Any]:
    totals = {"pair_count": 0, "comparable_pair_count": 0, "consistent": 0, "inversions": 0}
    for group in groups:
        for left_index in range(len(group)):
            for right_index in range(left_index + 1, len(group)):
                totals["pair_count"] += 1
                left, right = group[left_index], group[right_index]
                pred = _sign(float(left["teacher_score"]) - float(right["teacher_score"]), margin)
                real = _sign(float(right["real_task_cost"]) - float(left["real_task_cost"]), margin)
                if pred == 0 or real == 0:
                    continue
                totals["comparable_pair_count"] += 1
                if pred == real:
                    totals["consistent"] += 1
                else:
                    totals["inversions"] += 1
    comparable = totals["comparable_pair_count"]
    return {
        "pair_count": totals["pair_count"],
        "comparable_pair_count": comparable,
        "execution_consistent_pair_count": totals["consistent"],
        "execution_inversion_count": totals["inversions"],
        "execution_consistency": None if comparable == 0 else totals["consistent"] / comparable,
    }


def _fit(features: np.ndarray, target: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.maximum(np.std(features, axis=0), 1.0e-6)
    standardized = (features - mean) / scale
    design = np.concatenate([np.ones((len(features), 1)), standardized], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + alpha * penalty, design.T @ target)
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def _predict(features: np.ndarray, fit: dict[str, np.ndarray]) -> np.ndarray:
    standardized = (features - fit["mean"]) / fit["scale"]
    design = np.concatenate([np.ones((len(features), 1)), standardized], axis=1)
    return design @ fit["coefficients"]


def _fit_pairwise(
    features: np.ndarray,
    target: np.ndarray,
    groups: dict[str, list[int]],
    alpha: float,
) -> dict[str, np.ndarray]:
    """Fit candidate-effect differences using train roots only.

    The objective is invariant to root difficulty and directly targets the
    ordering used by candidate selection.  Root identities are never used
    outside the training split.
    """

    mean = np.mean(features, axis=0)
    scale = np.maximum(np.std(features, axis=0), 1.0e-6)
    standardized = (features - mean) / scale
    differences = []
    targets = []
    for indexes in groups.values():
        for left_offset, left in enumerate(indexes):
            for right in indexes[left_offset + 1 :]:
                differences.append(standardized[left] - standardized[right])
                targets.append(float(target[left] - target[right]))
    if not differences:
        raise ValueError("pairwise fit requires at least one within-root pair")
    design = np.asarray(differences, dtype=np.float64)
    pair_targets = np.asarray(targets, dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64)
    coefficients = np.linalg.solve(
        design.T @ design + alpha * penalty,
        design.T @ pair_targets,
    )
    # Reuse the common predictor, whose design has an intercept.  Pairwise
    # fitting leaves that intercept fixed at zero because it is not
    # identifiable from within-root differences and does not affect ranking.
    return {
        "mean": mean,
        "scale": scale,
        "coefficients": np.concatenate([np.zeros(1, dtype=np.float64), coefficients]),
    }


def _pairwise_fit_rmse(
    features: np.ndarray,
    target: np.ndarray,
    groups: dict[str, list[int]],
    fit: dict[str, np.ndarray],
) -> float:
    predicted = _predict(features, fit)
    errors = []
    for indexes in groups.values():
        for left_offset, left in enumerate(indexes):
            for right in indexes[left_offset + 1 :]:
                errors.append((predicted[left] - predicted[right]) - (target[left] - target[right]))
    return float(np.sqrt(np.mean(np.square(errors))))


def _action_summary(sequence: Any) -> np.ndarray:
    actions = np.asarray(sequence, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
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


def _goal_frame_action_summary(row: dict[str, Any]) -> np.ndarray:
    source_state = np.asarray(row.get("source_state_raw"), dtype=np.float64)
    goal = np.asarray(row.get("goal_raw"), dtype=np.float64)
    if source_state.ndim != 1 or source_state.size < 2 or goal.ndim != 1 or goal.size < 2:
        raise ValueError("goal-frame action features require source_state_raw and goal_raw")
    direct = goal[:2] - source_state[:2]
    norm = float(np.linalg.norm(direct))
    if norm <= 1.0e-8:
        raise ValueError("goal-frame action features require a nonzero source-goal displacement")
    direct = direct / norm
    perpendicular = np.asarray([-direct[1], direct[0]], dtype=np.float64)
    actions = np.asarray(row["action_sequence"], dtype=np.float64)
    canonical = np.stack([actions @ direct, actions @ perpendicular], axis=1)
    return _action_summary(canonical)


def _make_features(row: dict[str, Any], predicted: np.ndarray, variant: str) -> np.ndarray:
    source = np.asarray(row["source_latent_model"], dtype=np.float64)
    delta = predicted - source
    action = (
        _goal_frame_action_summary(row)
        if "goalframe_action" in variant
        else _action_summary(row["action_sequence"])
    )
    if variant.endswith("predicted_terminal"):
        return predicted
    if variant.endswith("source_delta"):
        return np.concatenate([source, delta])
    if variant.endswith("source_delta_action") or variant.endswith("goalframe_action"):
        return np.concatenate([source, delta, action])
    if variant.endswith("interaction"):
        source_delta = (source[:, None] * delta[None, :]).reshape(-1)
        delta_action = (delta[:, None] * action[None, :]).reshape(-1)
        return np.concatenate([source, delta, action, source_delta, delta_action])
    raise ValueError(f"unknown variant: {variant}")


def _evaluate(rows: list[dict[str, Any]], prediction: np.ndarray, margin: float) -> dict[str, Any]:
    scored = []
    for row, predicted_cost in zip(rows, prediction):
        item = dict(row)
        item["teacher_score"] = float(-predicted_cost)
        scored.append(item)
    by_split = {}
    regrets = {}
    root_metrics = {}
    for split in sorted({str(row["split"]) for row in scored}):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scored:
            if str(row["split"]) == split:
                groups[str(row["instance_id"])].append(row)
        root_regrets = []
        root_metrics[split] = {}
        for root, group in groups.items():
            selected = min(group, key=lambda row: -float(row["teacher_score"]))
            oracle = min(group, key=lambda row: float(row["real_task_cost"]))
            regret = float(selected["real_task_cost"] - oracle["real_task_cost"])
            root_regrets.append(regret)
            pairs = _pairs([group], margin)
            root_metrics[split][root] = {
                "regret": regret,
                "comparable_pair_count": pairs["comparable_pair_count"],
                "execution_consistency": pairs["execution_consistency"],
            }
        by_split[split] = {"root_count": len(groups), "pairs": _pairs(groups.values(), margin)}
        regrets[split] = {
            "root_count": len(root_regrets),
            "mean_regret": None if not root_regrets else float(np.mean(root_regrets)),
            "zero_regret_root_fraction": None
            if not root_regrets
            else float(np.mean(np.asarray(root_regrets) <= margin)),
        }
    return {
        "execution_consistency": by_split,
        "candidate_regret": regrets,
        "root_metrics": root_metrics,
    }


def audit(candidate_dataset: Path, arrays_path: Path, *, alpha: float = 1.0e-2, margin: float = 1.0e-3) -> dict[str, Any]:
    metadata, records = _load(candidate_dataset)
    arrays = np.load(arrays_path, allow_pickle=False)
    rows = []
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
        raise ValueError("train and held-out roots must be non-empty and disjoint")
    target = np.asarray([float(row["real_task_cost"]) for row in rows], dtype=np.float64)
    train_mask = np.asarray([str(row["split"]) == "train" for row in rows])
    train_target = target[train_mask].copy()
    train_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if train_mask[index]:
            train_groups[str(row["instance_id"])].append(index)
    train_row_indices = np.flatnonzero(train_mask)
    train_local_index = {int(row_index): local for local, row_index in enumerate(train_row_indices)}
    train_groups_local = {
        root: [train_local_index[index] for index in indexes]
        for root, indexes in train_groups.items()
    }
    root_means = {root: float(np.mean(target[indexes])) for root, indexes in train_groups.items()}
    centered_target = train_target.copy()
    for local_index, row_index in enumerate(np.flatnonzero(train_mask)):
        centered_target[local_index] -= root_means[str(rows[row_index]["instance_id"])]

    results = {}
    low_rank_basis: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    interaction_variant = "root_centered_source_delta_action_interaction"
    interaction_features = np.asarray(
        [_make_features(row, row["_predicted"], interaction_variant) for row in rows],
        dtype=np.float64,
    )
    base_dimension = _make_features(rows[0], rows[0]["_predicted"], "root_centered_source_delta_action").size
    interaction_block = interaction_features[:, base_dimension:]
    train_interaction = interaction_block[train_mask]
    interaction_mean = np.mean(train_interaction, axis=0)
    centered_interaction = train_interaction - interaction_mean
    _, _, right_singular_vectors = np.linalg.svd(
        centered_interaction, full_matrices=False
    )
    for rank in (2, 4, 8, 16):
        # Keep the requested output width even for tiny synthetic controls.  If
        # the train matrix has fewer independent rows, the remaining basis
        # directions are explicit zero columns rather than train/test leakage.
        basis = np.zeros((rank, right_singular_vectors.shape[1]), dtype=np.float64)
        effective_rank = min(rank, right_singular_vectors.shape[0])
        basis[:effective_rank] = right_singular_vectors[:effective_rank]
        low_rank_basis[rank] = (interaction_mean, basis)

    for variant in VARIANTS:
        pairwise = variant.endswith("_pairwise")
        feature_variant = variant.removesuffix("_pairwise")
        if "_lr" in feature_variant:
            rank = int(feature_variant.rsplit("_lr", 1)[1])
            if rank not in low_rank_basis:
                raise ValueError(f"low-rank interaction rank {rank} is unavailable")
            mean, basis = low_rank_basis[rank]
            projection = (interaction_block - mean) @ basis.T
            base_features = interaction_features[:, :base_dimension]
            features = np.concatenate([base_features, projection], axis=1)
        else:
            features = np.asarray([_make_features(row, row["_predicted"], feature_variant) for row in rows])
        train_features = features[train_mask]
        centered = feature_variant.startswith("root_centered_")
        fit_target = centered_target if centered else train_target
        if pairwise:
            fit = _fit_pairwise(train_features, train_target, train_groups_local, alpha)
        else:
            fit = _fit(train_features, fit_target, alpha)
        predicted_cost = _predict(features, fit)
        result = _evaluate(rows, predicted_cost, margin)
        train_predicted = _predict(train_features, fit)
        results[variant] = {
            "target": "train_pairwise_cost_difference" if pairwise else (
                "train_root_centered_cost" if centered else "absolute_cost"
            ),
            "feature_dimension": int(features.shape[1]),
            "train_fit_rmse": (
                _pairwise_fit_rmse(train_features, train_target, train_groups_local, fit)
                if pairwise
                else float(np.sqrt(np.mean((train_predicted - fit_target) ** 2)))
            ),
            **result,
        }
    return {
        "schema": "gdd_wm_relative_action_effect_geometry.v0.1",
        "status": "completed_diagnostic",
        "claim_boundary": (
            "train-only relative-effect and interaction control; not a D-JEPA teacher, "
            "not preference-stability evidence, and not closed-loop planning"
        ),
        "protocol": {
            "candidate_dataset": str(candidate_dataset.resolve()),
            "arrays": str(arrays_path.resolve()),
            "variant_order": list(VARIANTS),
            "fit_split": "train",
            "evaluation_splits": ["calibration", "outcome"],
            "ridge_alpha": float(alpha),
            "pair_margin": float(margin),
            "low_rank_interaction_ranks": sorted(low_rank_basis),
            "low_rank_basis_fit_split": "train",
            "outcome_used_for_fit_or_variant_selection": False,
        },
        "summary": {
            "record_count": len(rows),
            "train_root_count": len(train_roots),
            "heldout_root_count": len(heldout_roots),
            "latent_dimension": int(rows[0]["_predicted"].size),
        },
        "variants": results,
        "decision": {"variant_selection": "not_performed", "d1_ready": False, "d2_ready": False},
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
