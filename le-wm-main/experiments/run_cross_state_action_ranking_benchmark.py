"""Root-disjoint benchmark for RGB-only contact action-effect ranking.

This benchmark isolates the remaining failure mode of the visual adapter.  It
does not run a planner and does not use simulator successors as runtime
features.  Successors are used only to create frozen offline labels.

The proposed ``contact-canonical KNN`` represents each candidate action in the
RGB-estimated contact normal/tangent frame and retrieves effects from
root-disjoint training interventions.  This is deliberately a small,
non-parametric comparator: if it cannot transfer across contact geometry or
unseen action magnitudes, adding another MLP head is not a credible repair.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import stable_worldmodel  # noqa: F401

try:
    from .run_rgb_local_effect_adapter import (
        POSE_DIM,
        PUSHT_STATE_SIZE,
        action_bank,
        attach_targets,
        balanced_source_cases,
        block_pose,
        frozen_features,
        load_base,
        make_local_effect_dataset,
        predict_targets,
        train_adapter,
        wrap,
    )
except ImportError:
    from run_rgb_local_effect_adapter import (  # type: ignore
        POSE_DIM,
        PUSHT_STATE_SIZE,
        action_bank,
        attach_targets,
        balanced_source_cases,
        block_pose,
        frozen_features,
        load_base,
        make_local_effect_dataset,
        predict_targets,
        train_adapter,
        wrap,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-roots", type=int, default=96)
    parser.add_argument("--validation-roots", type=int, default=48)
    parser.add_argument("--test-roots", type=int, default=48)
    parser.add_argument("--root-rollout-steps", type=int, default=8)
    parser.add_argument("--contact-root-fraction", type=float, default=0.75)
    parser.add_argument("--contact-approach-steps", type=int, default=48)
    parser.add_argument("--action-directions", type=int, default=16)
    parser.add_argument("--train-action-magnitudes", type=float, nargs="+",
                        default=[0.08, 0.16, 0.24, 0.35])
    parser.add_argument("--test-action-magnitudes", type=float, nargs="+",
                        default=[0.08, 0.16, 0.24, 0.35, 0.60, 0.85])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--knn-context-weight", type=float, default=1.0)
    parser.add_argument("--knn-action-weight", type=float, default=2.0)
    parser.add_argument("--test-geometry-bins", type=int, default=4)
    return parser.parse_args()


def _root_context(features: np.ndarray) -> np.ndarray:
    """Return RGB-only root context, excluding candidate-specific action chart."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 60:
        raise ValueError("expected frozen adapter features with at least 60 columns")
    # Use relative and shape quantities rather than absolute canvas position.
    # Absolute x/y made nearest-neighbor transfer spuriously depend on where a
    # root happened to be sampled, while the local contact law is expressed in
    # the block-agent frame.  All quantities below are RGB-derived encoder
    # moments, not simulator state.
    agent = values[:, 1:3]
    block = values[:, 11:13]
    relative = block - agent
    block_shape = values[:, 13:20]
    temporal_agent = values[:, 31:33] - values[:, 1:3]
    temporal_block = values[:, 41:43] - values[:, 11:13]
    return np.concatenate([
        relative, np.linalg.norm(relative, axis=1, keepdims=True),
        block_shape, temporal_agent, temporal_block,
    ], axis=1).astype(np.float32)


def _contact_action_chart(features: np.ndarray) -> np.ndarray:
    """Extract [normal action, tangent action, magnitude] from RGB chart."""
    values = np.asarray(features, dtype=np.float32)
    # frozen_features layout: current(30), temporal(30), action(2),
    # contact chart(10), boundary chart(10).
    chart = values[:, 62:72]
    return chart[:, 7:10].astype(np.float32)


def _world_from_contact(effect: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Rotate predicted [zero, contact-frame effect] using the RGB chart."""
    predicted = np.asarray(effect, dtype=np.float32).copy()
    chart = np.asarray(features, dtype=np.float32)[:, 62:72]
    normal = chart[:, 4:6]
    # contact chart columns are [agent(2), block(2), relative(2), distance,
    # action_normal, action_tangent, action_magnitude].  Its relative vector
    # is block-agent, hence the contact normal is chart[4:6] after
    # normalization only if the chart is reconstructed explicitly below.
    agent = features[:, 1:3] * (PUSHT_STATE_SIZE / 2.0)
    block = features[:, 11:13] * (PUSHT_STATE_SIZE / 2.0)
    relative = block - agent
    norm = np.linalg.norm(relative, axis=1, keepdims=True)
    normal = relative / np.maximum(norm, 1e-6)
    tangent = np.column_stack([-normal[:, 1], normal[:, 0]])
    normal_effect = predicted[:, POSE_DIM].copy()
    tangent_effect = predicted[:, POSE_DIM + 1].copy()
    predicted[:, POSE_DIM] = normal_effect * normal[:, 0] + tangent_effect * tangent[:, 0]
    predicted[:, POSE_DIM + 1] = normal_effect * normal[:, 1] + tangent_effect * tangent[:, 1]
    return predicted


def _canonical_queries(features: np.ndarray) -> np.ndarray:
    """Build candidate action coordinates from the RGB-derived feature chart."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must be two-dimensional")
    return _contact_action_chart(values)


def contact_canonical_knn(
    train_features: np.ndarray,
    train_targets_contact: np.ndarray,
    test_features: np.ndarray,
    train_root_ids: np.ndarray,
    test_root_ids: np.ndarray,
    k: int,
    context_weight: float,
    action_weight: float,
) -> np.ndarray:
    """Predict local effects with root-disjoint RGB/contact-frame KNN.

    The leave-one-root boundary is structural: all neighbors come from train
    roots, and the function rejects overlapping root ids.  The prediction is
    made in the contact frame and returned in the adapter's six-dimensional
    [zero_delta, effect] format.
    """
    train_features = np.asarray(train_features, dtype=np.float32)
    test_features = np.asarray(test_features, dtype=np.float32)
    train_targets_contact = np.asarray(train_targets_contact, dtype=np.float32)
    train_root_ids = np.asarray(train_root_ids)
    test_root_ids = np.asarray(test_root_ids)
    if np.intersect1d(np.unique(train_root_ids), np.unique(test_root_ids)).size:
        raise ValueError("KNN train/test roots overlap")
    context_train = _root_context(train_features)
    context_test = _root_context(test_features)
    mean = context_train.mean(axis=0)
    std = np.maximum(context_train.std(axis=0), 1e-4)
    context_train = (context_train - mean) / std
    context_test = (context_test - mean) / std
    action_train = _canonical_queries(train_features)
    action_test = _canonical_queries(test_features)
    action_mean = action_train.mean(axis=0)
    action_std = np.maximum(action_train.std(axis=0), 1e-4)
    action_train = (action_train - action_mean) / action_std
    action_test = (action_test - action_mean) / action_std
    predictions = np.zeros((len(test_features), 2 * POSE_DIM), dtype=np.float32)
    for row in range(len(test_features)):
        distance = context_weight * np.mean(
            np.square(context_train - context_test[row]), axis=1,
        )
        distance += action_weight * np.mean(
            np.square(action_train - action_test[row]), axis=1,
        )
        order = np.argsort(distance, kind="stable")
        selected: list[int] = []
        seen_roots: set[int] = set()
        for index in order:
            root = int(train_root_ids[index])
            if root not in seen_roots:
                selected.append(int(index))
                seen_roots.add(root)
            if len(selected) >= max(1, int(k)):
                break
        selected_array = np.asarray(selected, dtype=np.int64)
        weights = 1.0 / np.maximum(distance[selected_array], 1e-5)
        weights /= weights.sum()
        predictions[row] = np.sum(
            train_targets_contact[selected_array] * weights[:, None], axis=0,
        )
    return predictions


def _goal_poses(data: dict[str, np.ndarray]) -> np.ndarray:
    roots = np.unique(data["root_id"])
    first = [np.flatnonzero(data["root_id"] == root)[0] for root in roots]
    goals = np.asarray([data["goal_state"][index] for index in first], dtype=np.float32)
    return np.asarray([
        [goal[2], goal[3], np.sin(goal[4]), np.cos(goal[4])] for goal in goals
    ], dtype=np.float32)


def _attach_goal_states(
    data: dict[str, np.ndarray], roots: list[Any],
) -> dict[str, np.ndarray]:
    """Attach offline goal states for evaluation only; never used as features."""
    out = dict(data)
    goals = np.asarray([root[1] for root in roots], dtype=np.float32)
    out["goal_state"] = goals[out["root_id"]]
    return out


def _rank_metrics(
    data: dict[str, np.ndarray],
    predicted_world: np.ndarray,
    goal_poses: np.ndarray,
    candidate_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    hits: list[float] = []
    contact_hits: list[float] = []
    regrets: list[float] = []
    contact_regrets: list[float] = []
    for root_id in np.unique(data["root_id"]):
        indices = np.flatnonzero(data["root_id"] == root_id)
        if candidate_mask is not None:
            indices = indices[np.asarray(candidate_mask, dtype=bool)[indices]]
        if len(indices) < 2:
            raise RuntimeError("candidate support must retain at least two actions per root")
        goal = goal_poses[root_id]
        goal_angle = float(np.arctan2(goal[2], goal[3]))
        goal_pose = np.asarray([goal[0], goal[1], goal_angle], dtype=np.float32)
        current = block_pose(data["state"][indices[0]])
        true = np.asarray([block_pose(row) for row in data["next_state"][indices]])
        true_cost = np.linalg.norm(true[:, :2] - goal_pose[None, :2], axis=1) / 20.0
        true_cost += 10.0 * np.abs(wrap(true[:, 2] - goal_pose[2]))
        predicted_next = np.tile(current, (len(indices), 1)) + predicted_world[indices[0], :POSE_DIM]
        predicted_next = predicted_next + predicted_world[indices, POSE_DIM:]
        predicted_cost = np.linalg.norm(predicted_next[:, :2] - goal_pose[None, :2], axis=1) / 20.0
        predicted_cost += 10.0 * np.abs(wrap(predicted_next[:, 2] - goal_pose[2]))
        true_best = int(np.argmin(true_cost))
        predicted_best = int(np.argmin(predicted_cost))
        hits.append(float(predicted_best == true_best))
        # Regret is measured against the *true* task cost, not against the
        # selected model's own minimum (which would be identically zero).
        regret = float(true_cost[predicted_best] - np.min(true_cost))
        regrets.append(regret)
        if bool(data["contact_root"][indices[0]]):
            contact_hits.append(hits[-1])
            contact_regrets.append(regret)
    return {
        "roots": int(len(hits)),
        "top1": float(np.mean(hits)) if hits else float("nan"),
        "mean_regret": float(np.mean(regrets)) if regrets else float("nan"),
        "contact_roots": int(len(contact_hits)),
        "contact_top1": float(np.mean(contact_hits)) if contact_hits else float("nan"),
        "contact_mean_regret": float(np.mean(contact_regrets)) if contact_regrets else float("nan"),
    }


def _make_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )


def _prepare_roots(
    env: gym.Env, args: argparse.Namespace,
) -> tuple[list[Any], list[Any], list[Any]]:
    train = balanced_source_cases(
        env, args.train_roots, args.root_rollout_steps, args.seed,
        args.contact_root_fraction, args.contact_approach_steps,
    )
    validation = balanced_source_cases(
        env, args.validation_roots, args.root_rollout_steps, args.seed + 50_000,
        args.contact_root_fraction, args.contact_approach_steps,
    )
    test = balanced_source_cases(
        env, args.test_roots, args.root_rollout_steps, args.seed + 100_000,
        args.contact_root_fraction, args.contact_approach_steps,
    )
    train_hashes = {tuple(np.asarray(root[0]).round(6)) for root in train}
    validation_hashes = {tuple(np.asarray(root[0]).round(6)) for root in validation}
    test_hashes = {tuple(np.asarray(root[0]).round(6)) for root in test}
    if train_hashes & validation_hashes or train_hashes & test_hashes or validation_hashes & test_hashes:
        raise RuntimeError("train/validation/test source roots overlap")
    return train, validation, test


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, _normalizer, _base_args = load_base(Path(args.base_checkpoint).resolve(), args.device)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=200)
    train_actions = action_bank(args.action_directions, args.train_action_magnitudes)
    test_actions = action_bank(args.action_directions, args.test_action_magnitudes)
    train_roots, validation_roots, test_roots = _prepare_roots(env, args)
    train = attach_targets(
        make_local_effect_dataset(env, train_roots, 64, train_actions, 1), train_actions,
    )
    validation = attach_targets(
        make_local_effect_dataset(env, validation_roots, 64, train_actions, 1), train_actions,
    )
    validation = _attach_goal_states(validation, validation_roots)
    test = attach_targets(
        make_local_effect_dataset(env, test_roots, 64, test_actions, 1), test_actions,
    )
    test = _attach_goal_states(test, test_roots)
    train_features = frozen_features(model, train, 64, args.device)
    validation_features = frozen_features(model, validation, 64, args.device)
    test_features = frozen_features(model, test, 64, args.device)
    # Paired baseline: the legacy 82-column adapter cannot see the previous
    # action; the conditioned adapter receives the two appended action values.
    train_args = _make_args(args)
    train_targets = train["local_target"]
    legacy_adapter, legacy_training = train_adapter(
        train_features[:, :82], train_targets, validation_features[:, :82],
        validation["local_target"], np.ones(len(train_features), dtype=np.float32), train_args,
    )
    legacy_val_pred = predict_targets(
        legacy_adapter, validation_features[:, :82], legacy_training, args.device,
    )
    legacy_pred = predict_targets(legacy_adapter, test_features[:, :82], legacy_training, args.device)
    adapter, adapter_training = train_adapter(
        train_features, train_targets, validation_features, validation["local_target"],
        np.ones(len(train_features), dtype=np.float32), train_args,
    )
    mlp_pred = predict_targets(adapter, test_features, adapter_training, args.device)
    mlp_val_pred = predict_targets(adapter, validation_features, adapter_training, args.device)
    # KNN is trained only on contact-root interventions and predicts contact
    # frame effects; non-contact roots retain the global MLP for a fair mixed
    # evaluation, while the primary reported quantity is contact-only.
    train_contact = train["contact_root"]
    train_contact_features = train_features[train_contact]
    train_contact_targets = train["local_target_contact"][train_contact]
    train_contact_root_ids = train["root_id"][train_contact]
    test_contact = test["contact_root"]
    knn_contact_pred = contact_canonical_knn(
        train_contact_features, train_contact_targets,
        test_features[test_contact], train_contact_root_ids + 1_000_000,
        test["root_id"][test_contact], args.knn_k,
        args.knn_context_weight, args.knn_action_weight,
    )
    validation_contact = validation["contact_root"]
    knn_validation_contact_pred = contact_canonical_knn(
        train_contact_features, train_contact_targets,
        validation_features[validation_contact], train_contact_root_ids,
        validation["root_id"][validation_contact] + 2_000_000, args.knn_k,
        args.knn_context_weight, args.knn_action_weight,
    )
    knn_world_contact = _world_from_contact(knn_contact_pred, test_features[test_contact])
    knn_world_validation_contact = _world_from_contact(
        knn_validation_contact_pred, validation_features[validation_contact],
    )
    knn_pred = mlp_pred.copy()
    knn_pred[test_contact] = knn_world_contact
    knn_validation_pred = mlp_val_pred.copy()
    knn_validation_pred[validation_contact] = knn_world_validation_contact
    # The baseline's continuous prediction is in world coordinates.
    goal_poses = _goal_poses(test)
    small_action_mask = np.linalg.norm(test["action"], axis=1) <= 0.35 + 1e-6
    large_action_mask = np.linalg.norm(test["action"], axis=1) >= 0.60 - 1e-6
    validation_small_action_mask = np.linalg.norm(validation["action"], axis=1) <= 0.35 + 1e-6
    validation_large_action_mask = np.linalg.norm(validation["action"], axis=1) >= 0.60 - 1e-6
    validation_goal_poses = _goal_poses(validation)
    # The gate is fixed before test evaluation.  It uses only validation
    # contact regret and does not inspect test labels or planning outcomes.
    gate_candidates: list[dict[str, Any]] = []
    for weight in np.linspace(0.0, 1.0, 5):
        blended = mlp_val_pred.copy()
        blended[validation_contact] = (
            (1.0 - float(weight)) * mlp_val_pred[validation_contact]
            + float(weight) * knn_validation_pred[validation_contact]
        )
        metrics = _rank_metrics(validation, blended, validation_goal_poses)
        gate_candidates.append({
            "knn_weight": float(weight),
            "validation_contact_mean_regret": metrics["contact_mean_regret"],
            "validation_contact_top1": metrics["contact_top1"],
        })
    selected_gate = min(
        gate_candidates,
        key=lambda row: (
            float(row["validation_contact_mean_regret"]),
            -float(row["validation_contact_top1"]),
            float(row["knn_weight"]),
        ),
    )
    selected_weight = float(selected_gate["knn_weight"])
    gated_pred = mlp_pred.copy()
    gated_pred[test_contact] = (
        (1.0 - selected_weight) * mlp_pred[test_contact]
        + selected_weight * knn_pred[test_contact]
    )
    report = {
        "protocol": {
            "runtime_state_access": False,
            "runtime_successor_verification": False,
            "offline_label_role": "matched simulator one-step interventions",
            "train_roots": len(train_roots),
            "validation_roots": len(validation_roots),
            "test_roots": len(test_roots),
            "train_contact_roots": int(sum(bool(root[-1]) for root in train_roots)),
            "test_contact_roots": int(sum(bool(root[-1]) for root in test_roots)),
            "validation_contact_roots": int(sum(bool(root[-1]) for root in validation_roots)),
            "train_action_magnitudes": [float(x) for x in args.train_action_magnitudes],
            "test_action_magnitudes": [float(x) for x in args.test_action_magnitudes],
            "root_disjoint": True,
            "candidate_count_train": int(len(train_actions)),
            "candidate_count_test": int(len(test_actions)),
        },
        "global_mlp_without_previous_action": {
            "all": _rank_metrics(test, legacy_pred, goal_poses),
            "contact_small_support": _rank_metrics(test, legacy_pred, goal_poses, small_action_mask),
            "contact_large_support": _rank_metrics(test, legacy_pred, goal_poses, large_action_mask),
            "feature_width": 82,
        },
        "global_mlp_previous_action_conditioned": {
            "all": _rank_metrics(test, mlp_pred, goal_poses),
            "contact_small_support": _rank_metrics(test, mlp_pred, goal_poses, small_action_mask),
            "contact_large_support": _rank_metrics(test, mlp_pred, goal_poses, large_action_mask),
            "feature_width": int(train_features.shape[1]),
        },
        "contact_canonical_knn": {
            "all": _rank_metrics(test, knn_pred, goal_poses),
            "contact_small_support": _rank_metrics(test, knn_pred, goal_poses, small_action_mask),
            "contact_large_support": _rank_metrics(test, knn_pred, goal_poses, large_action_mask),
            "k": int(args.knn_k),
            "context_weight": float(args.knn_context_weight),
            "action_weight": float(args.knn_action_weight),
        },
        "validation_gated_ensemble": {
            "all": _rank_metrics(test, gated_pred, goal_poses),
            "contact_small_support": _rank_metrics(test, gated_pred, goal_poses, small_action_mask),
            "contact_large_support": _rank_metrics(test, gated_pred, goal_poses, large_action_mask),
            "selected_knn_weight": selected_weight,
            "selection_metric": "validation_contact_mean_regret",
            "validation_candidates": gate_candidates,
            "validation_support_note": (
                "validation contains only train-action support; large-magnitude test support is evaluated once on test"
            ),
        },
        "claim_boundary": {
            "status": "benchmark_only",
            "primary_test": "held-out contact-root ranking under root-disjoint RGB-only evaluation",
            "promotion_rule": "canonical KNN must beat global MLP on contact_top1 and contact_mean_regret on both action supports",
            "no_planning_claim": True,
        },
        "sanity_checks": {
            "finite_features": bool(np.isfinite(train_features).all() and np.isfinite(test_features).all()),
            "finite_predictions": bool(np.isfinite(mlp_pred).all() and np.isfinite(knn_pred).all()),
            "root_overlap": False,
            "train_rows": int(len(train_features)),
            "test_rows": int(len(test_features)),
            "validation_rows": int(len(validation_features)),
        },
        "wall_seconds": float(time.perf_counter() - started),
    }
    (output / "cross_state_action_ranking.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    lines = [
        "# Cross-state action-ranking benchmark",
        "",
        "This artifact is root-disjoint. Simulator successors are offline labels only.",
        "",
        "| method | split | contact top-1 | contact mean regret |",
        "|---|---:|---:|---:|",
    ]
    for method in (
        "global_mlp_without_previous_action",
        "global_mlp_previous_action_conditioned",
        "contact_canonical_knn",
    ):
        for split in ("all", "contact_small_support", "contact_large_support"):
            row = report[method][split]
            lines.append(
                f"| {method} | {split} | {row['contact_top1']:.4f} | {row['contact_mean_regret']:.6f} |"
            )
    lines.extend(["", "Promotion rule: " + report["claim_boundary"]["promotion_rule"] + "."])
    (output / "cross_state_action_ranking.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    env.close()


if __name__ == "__main__":
    main()
