"""Conditional covariance--transition geometry on frozen external LeWM data.

The proposed next step for the project is to replace one global operator by
conditional/local operators.  This audit implements the smallest externally
testable version without retraining: transitions are split into low/high
action-magnitude groups using training-fold quantiles, a ridge operator is fit
within each group, and its covariance-relative geometry is evaluated on the
held-out episode.  A size-matched random partition is the control.

This is a conditional linear diagnostic of frozen trajectories, not a
Jacobian-identification, causal-control, or online-planning experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.run_external_trajectory_operator_audit import (  # noqa: E402
    CONTEXT_LENGTH,
    _apply_row_operator,
    _bootstrap_clustered,
    _fit_row_operator,
    _load_trajectories,
    _pca_fit,
    _pca_transform,
)


DEFAULT_BINS = 2


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _covariance_stats(states: np.ndarray) -> dict[str, float]:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("states must contain at least two rows")
    centered = states - states.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(states.shape[0] - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0.0)
    total = max(float(eigenvalues.sum()), 1e-12)
    frobenius_sq = max(float(np.square(eigenvalues).sum()), 1e-12)
    positive = eigenvalues[eigenvalues > max(eigenvalues.max() * 1e-10, 1e-12)]
    return {
        "trace": float(total),
        "max_eigenvalue": float(eigenvalues[-1]),
        "mean_eigenvalue": float(total / eigenvalues.size),
        "anisotropy_max_over_mean": float(eigenvalues[-1] / max(total / eigenvalues.size, 1e-12)),
        "stable_rank": float(total * total / frobenius_sq),
        "effective_rank": float(np.exp(-(eigenvalues / total * np.log(np.maximum(eigenvalues / total, 1e-30))).sum())),
        "positive_rank": int(positive.size),
    }


def _normalized_commutator(covariance_states: np.ndarray, row_operator: np.ndarray) -> float:
    states = np.asarray(covariance_states, dtype=np.float64)
    centered = states - states.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(states.shape[0] - 1, 1)
    operator = np.asarray(row_operator, dtype=np.float64).T
    commutator = covariance @ operator - operator @ covariance
    denominator = max(float(np.linalg.norm(covariance, "fro") * np.linalg.norm(operator, "fro")), 1e-12)
    return float(np.linalg.norm(commutator, "fro") / denominator)


def _quantile_labels(scores: np.ndarray, num_bins: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or num_bins < 2:
        raise ValueError("scores must be one-dimensional and num_bins >= 2")
    quantiles = np.linspace(0.0, 1.0, int(num_bins) + 1)[1:-1]
    thresholds = np.quantile(scores, quantiles) if quantiles.size else np.empty(0)
    labels = np.digitize(scores, thresholds, right=False)
    return np.clip(labels, 0, int(num_bins) - 1).astype(np.int64), np.asarray(thresholds, dtype=np.float64)


def _labels_from_thresholds(
    scores: np.ndarray, thresholds: np.ndarray, num_bins: int
) -> np.ndarray:
    """Apply training-fold thresholds without silently dropping held-out rows."""
    labels = np.digitize(np.asarray(scores, dtype=np.float64), thresholds, right=False)
    return np.clip(labels, 0, int(num_bins) - 1).astype(np.int64)


def _direction_scores(
    actions: np.ndarray, reference_actions: np.ndarray
) -> np.ndarray:
    """Return signed direction scores relative to a training-fold action axis.

    The score is cosine-like, so this control tests action direction rather
    than merely selecting large-magnitude vectors.  The axis is estimated on
    the training fold only; its arbitrary sign does not affect quantile bins.
    """
    reference = np.asarray(reference_actions, dtype=np.float64)
    centered = reference - reference.mean(axis=0, keepdims=True)
    if centered.ndim != 2 or centered.shape[1] < 1:
        raise ValueError("reference_actions must be a two-dimensional matrix")
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0 or float(singular_values[0]) <= 1e-12:
        axis = np.zeros(reference.shape[1], dtype=np.float64)
        axis[0] = 1.0
    else:
        axis = right_vectors[0]
    values = np.asarray(actions, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1)
    return (values @ axis) / np.maximum(norms, 1e-12)


def _state_matched_permutation_labels(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_states: np.ndarray,
    test_states: np.ndarray,
    seed: int,
    num_strata: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute condition labels within state-PCA strata.

    The null preserves the condition-label composition within coarse state
    strata, while breaking the alignment between the action condition and the
    transition row.  Strata are defined on training rows only and transferred
    to held-out rows with the training quantile thresholds.
    """
    train_states = np.asarray(train_states, dtype=np.float64)
    test_states = np.asarray(test_states, dtype=np.float64)
    if train_states.ndim != 2 or test_states.ndim != 2 or train_states.shape[1] < 1:
        raise ValueError("state arrays must be two-dimensional with at least one feature")
    train_state_labels, thresholds = _quantile_labels(train_states[:, 0], num_strata)
    test_state_labels = _labels_from_thresholds(test_states[:, 0], thresholds, num_strata)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    if train_labels.shape != train_state_labels.shape or test_labels.shape != test_state_labels.shape:
        raise ValueError("condition and state-label arrays must have matching row counts")
    rng = np.random.default_rng(seed)
    shuffled_train = train_labels.copy()
    shuffled_test = test_labels.copy()
    for stratum in range(int(num_strata)):
        train_mask = train_state_labels == stratum
        test_mask = test_state_labels == stratum
        if int(train_mask.sum()) > 1:
            shuffled_train[train_mask] = rng.permutation(train_labels[train_mask])
        if int(test_mask.sum()) > 1:
            shuffled_test[test_mask] = rng.permutation(test_labels[test_mask])
    return shuffled_train, shuffled_test


def _r2(predicted: np.ndarray, target: np.ndarray) -> float:
    numerator = float(np.square(np.asarray(predicted) - np.asarray(target)).sum())
    denominator = max(float(np.square(np.asarray(target) - np.asarray(target).mean(axis=0)).sum()), 1e-12)
    return float(1.0 - numerator / denominator)


def _nested_metric(record: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = record
    for key in path:
        value = value[key]
    return float(value)


def _fit_partition(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    num_bins: int,
    ridge: float,
    min_samples: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for label in range(int(num_bins)):
        train_mask = train_labels == label
        test_mask = test_labels == label
        if int(train_mask.sum()) < int(min_samples) or not np.any(test_mask):
            continue
        fit = _fit_row_operator(train_x[train_mask], train_y[train_mask], ridge=ridge)
        prediction = _apply_row_operator(fit, test_x[test_mask])
        records.append(
            {
                "bin": int(label),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "commutator": _normalized_commutator(train_x[train_mask], fit["row_operator"]),
                "covariance": _covariance_stats(train_x[train_mask]),
                "test_mse": float(np.square(prediction - test_y[test_mask]).mean()),
                "test_r2": _r2(prediction, test_y[test_mask]),
                "_row_operator": fit["row_operator"],
            }
        )
    if not records:
        raise ValueError("partition produced no estimable bins")
    total_test = sum(item["test_rows"] for item in records)
    operators = [np.asarray(item["_row_operator"], dtype=np.float64) for item in records]
    pairwise_heterogeneity = []
    for left in range(len(operators)):
        for right in range(left + 1, len(operators)):
            denominator = max(
                float(
                    np.linalg.norm(operators[left], ord="fro")
                    + np.linalg.norm(operators[right], ord="fro")
                ),
                1e-12,
            )
            pairwise_heterogeneity.append(
                float(
                    np.linalg.norm(operators[left] - operators[right], ord="fro")
                    / denominator
                )
            )
    return {
        "bins": [
            {key: value for key, value in item.items() if key != "_row_operator"}
            for item in records
        ],
        "operator_heterogeneity": float(np.mean(pairwise_heterogeneity)) if pairwise_heterogeneity else 0.0,
        "weighted_commutator": float(sum(item["commutator"] * item["test_rows"] for item in records) / total_test),
        "weighted_anisotropy": float(sum(item["covariance"]["anisotropy_max_over_mean"] * item["test_rows"] for item in records) / total_test),
        "weighted_stable_rank": float(sum(item["covariance"]["stable_rank"] * item["test_rows"] for item in records) / total_test),
        "weighted_test_mse": float(sum(item["test_mse"] * item["test_rows"] for item in records) / total_test),
        "weighted_test_r2": float(sum(item["test_r2"] * item["test_rows"] for item in records) / total_test),
        "covered_test_rows": int(total_test),
    }


def _evaluate_source(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    source_name: str,
    pca_dimension: int,
    ridge: float,
    num_bins: int,
    min_samples: int,
    seed: int,
    condition: str,
    state_strata: int,
) -> dict[str, Any]:
    pca_fit = _pca_fit(
        np.concatenate(
            [np.stack([row["true"] for row in train_rows]), np.stack([row["predicted"] for row in train_rows])]
        ),
        pca_dimension,
    )
    train_states = np.stack([_pca_transform(row[source_name], pca_fit) for row in train_rows])
    test_states = np.stack([_pca_transform(row[source_name], pca_fit) for row in test_rows])
    train_actions = np.stack([row["action_features"] for row in train_rows])
    test_actions = np.stack([row["action_features"] for row in test_rows])
    dimension = train_states.shape[-1]
    train_x = train_states[:, :-1].reshape(-1, dimension)
    train_y = train_states[:, 1:].reshape(-1, dimension)
    test_x = test_states[:, :-1].reshape(-1, dimension)
    test_y = test_states[:, 1:].reshape(-1, dimension)
    if condition == "magnitude":
        train_scores = np.linalg.norm(train_actions, axis=-1).reshape(-1)
        test_scores = np.linalg.norm(test_actions, axis=-1).reshape(-1)
    elif condition == "direction":
        train_flat = train_actions.reshape(-1, train_actions.shape[-1])
        test_flat = test_actions.reshape(-1, test_actions.shape[-1])
        train_scores = _direction_scores(train_flat, train_flat)
        test_scores = _direction_scores(test_flat, train_flat)
    else:
        raise ValueError(f"unsupported condition: {condition}")
    train_labels, thresholds = _quantile_labels(train_scores, num_bins)
    test_labels = _labels_from_thresholds(test_scores, thresholds, num_bins)

    global_fit = _fit_row_operator(train_x, train_y, ridge=ridge)
    global_prediction = _apply_row_operator(global_fit, test_x)
    global_metrics = {
        "commutator": _normalized_commutator(train_x, global_fit["row_operator"]),
        "covariance": _covariance_stats(train_x),
        "test_mse": float(np.square(global_prediction - test_y).mean()),
        "test_r2": _r2(global_prediction, test_y),
    }
    rng = np.random.default_rng(seed)
    random_train_labels = rng.permutation(train_labels)
    random_test_labels = rng.permutation(test_labels)
    state_matched_train_labels, state_matched_test_labels = _state_matched_permutation_labels(
        train_labels,
        test_labels,
        train_x,
        test_x,
        seed + 7919,
        num_strata=state_strata,
    )
    conditional = _fit_partition(train_x, train_y, test_x, test_y, train_labels, test_labels, num_bins, ridge, min_samples)
    random_partition = _fit_partition(train_x, train_y, test_x, test_y, random_train_labels, random_test_labels, num_bins, ridge, min_samples)
    state_matched_partition = _fit_partition(
        train_x,
        train_y,
        test_x,
        test_y,
        state_matched_train_labels,
        state_matched_test_labels,
        num_bins,
        ridge,
        min_samples,
    )
    return {
        "pca_dimension": int(dimension),
        "condition": condition,
        "action_thresholds": thresholds.tolist(),
        "global": global_metrics,
        "conditional": conditional,
        "random_partition": random_partition,
        "state_matched_partition": state_matched_partition,
    }


def _crossfit(
    trajectories: list[dict[str, Any]],
    pca_dimension: int,
    ridge: float,
    num_bins: int,
    min_samples: int,
    state_strata: int = 8,
) -> dict[str, Any]:
    if int(state_strata) < 2:
        raise ValueError("state_strata must be at least 2")
    episode_ids = np.asarray(sorted({int(row["episode_id"]) for row in trajectories}), dtype=np.int64)
    folds: list[dict[str, Any]] = []
    for test_episode in episode_ids.tolist():
        train_rows = [row for row in trajectories if int(row["episode_id"]) != test_episode]
        test_rows = [row for row in trajectories if int(row["episode_id"]) == test_episode]
        fold = {"test_episode": int(test_episode), "windows": len(test_rows)}
        for index, source_name in enumerate(("true", "predicted")):
            fold[source_name] = {}
            for condition_index, condition in enumerate(("magnitude", "direction")):
                fold[source_name][condition] = _evaluate_source(
                    train_rows,
                    test_rows,
                    source_name,
                    pca_dimension,
                    ridge,
                    num_bins,
                    min_samples,
                    20261004 + index * 100003 + condition_index * 1009 + test_episode,
                    condition,
                    state_strata,
                )
        folds.append(fold)

    summary: dict[str, Any] = {"by_source": {}}
    for source in ("true", "predicted"):
        summary["by_source"][source] = {}
        for condition_index, condition in enumerate(("magnitude", "direction")):
            condition_summary = {}
            metric_paths = (
                ("global", "commutator"),
                ("global", "test_r2"),
                ("global", "covariance", "anisotropy_max_over_mean"),
                ("global", "covariance", "stable_rank"),
                ("conditional", "weighted_commutator"),
                ("random_partition", "weighted_commutator"),
                ("conditional", "operator_heterogeneity"),
                ("random_partition", "operator_heterogeneity"),
                ("state_matched_partition", "operator_heterogeneity"),
                ("conditional", "weighted_anisotropy"),
                ("random_partition", "weighted_anisotropy"),
                ("conditional", "weighted_stable_rank"),
                ("random_partition", "weighted_stable_rank"),
                ("conditional", "weighted_test_r2"),
                ("random_partition", "weighted_test_r2"),
                ("conditional", "covered_test_rows"),
                ("random_partition", "covered_test_rows"),
                ("state_matched_partition", "covered_test_rows"),
            )
            for metric_index, metric_path in enumerate(metric_paths):
                values = np.asarray(
                    [
                        _nested_metric(fold[source][condition], metric_path)
                        for fold in folds
                    ],
                    dtype=np.float64,
                )
                condition_summary["/".join(metric_path)] = _bootstrap_clustered(
                    values,
                    episode_ids,
                    repeats=2000,
                    seed=20261100 + condition_index * 100 + metric_index,
                )
            cond_comm = np.asarray(
                [fold[source][condition]["conditional"]["weighted_commutator"] for fold in folds]
            )
            rand_comm = np.asarray(
                [fold[source][condition]["random_partition"]["weighted_commutator"] for fold in folds]
            )
            state_matched_comm = np.asarray(
                [fold[source][condition]["state_matched_partition"]["weighted_commutator"] for fold in folds]
            )
            cond_r2 = np.asarray(
                [fold[source][condition]["conditional"]["weighted_test_r2"] for fold in folds]
            )
            rand_r2 = np.asarray(
                [fold[source][condition]["random_partition"]["weighted_test_r2"] for fold in folds]
            )
            state_matched_r2 = np.asarray(
                [fold[source][condition]["state_matched_partition"]["weighted_test_r2"] for fold in folds]
            )
            global_comm = np.asarray(
                [fold[source][condition]["global"]["commutator"] for fold in folds]
            )
            condition_summary["conditional_minus_random/commutator"] = _bootstrap_clustered(
                cond_comm - rand_comm,
                episode_ids,
                repeats=2000,
                seed=20261201 + condition_index,
            )
            condition_summary["conditional_minus_state_matched/commutator"] = _bootstrap_clustered(
                cond_comm - state_matched_comm,
                episode_ids,
                repeats=2000,
                seed=20261202 + condition_index,
            )
            cond_heterogeneity = np.asarray(
                [fold[source][condition]["conditional"]["operator_heterogeneity"] for fold in folds]
            )
            rand_heterogeneity = np.asarray(
                [fold[source][condition]["random_partition"]["operator_heterogeneity"] for fold in folds]
            )
            condition_summary["conditional_minus_random/operator_heterogeneity"] = _bootstrap_clustered(
                cond_heterogeneity - rand_heterogeneity,
                episode_ids,
                repeats=2000,
                seed=20261203 + condition_index,
            )
            state_matched_heterogeneity = np.asarray(
                [fold[source][condition]["state_matched_partition"]["operator_heterogeneity"] for fold in folds]
            )
            condition_summary["conditional_minus_state_matched/operator_heterogeneity"] = _bootstrap_clustered(
                cond_heterogeneity - state_matched_heterogeneity,
                episode_ids,
                repeats=2000,
                seed=20261205 + condition_index,
            )
            condition_summary["conditional_minus_global/commutator"] = _bootstrap_clustered(
                cond_comm - global_comm,
                episode_ids,
                repeats=2000,
                seed=20261207 + condition_index,
            )
            condition_summary["conditional_minus_random/test_r2"] = _bootstrap_clustered(
                cond_r2 - rand_r2,
                episode_ids,
                repeats=2000,
                seed=20261209 + condition_index,
            )
            condition_summary["conditional_minus_state_matched/test_r2"] = _bootstrap_clustered(
                cond_r2 - state_matched_r2,
                episode_ids,
                repeats=2000,
                seed=20261211 + condition_index,
            )
            summary["by_source"][source][condition] = condition_summary
    return {"folds": folds, "summary": summary}


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["crossfit"]["summary"]["by_source"]
    lines = [
        "# External frozen LeWM conditional operator geometry",
        "",
        "Action-magnitude and action-direction bins are defined from training-fold quantiles. Every operator and covariance is fit outside the held-out episode. The random partition preserves bin sizes but breaks their action alignment.",
        "",
        "| source | condition | aligned heterogeneity | random heterogeneity | state-matched heterogeneity | $\\Delta H$ vs random | $\\Delta H$ vs state-matched | $\\Delta R^2$ vs state-matched | coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in ("true", "predicted"):
        for condition in ("magnitude", "direction"):
            item = summary[source][condition]
            def f(key: str) -> str:
                value = item[key]
                return f"{value['mean']:.6g} [{value['ci95'][0]:.6g}, {value['ci95'][1]:.6g}]"
            conditional_coverage = item["conditional/covered_test_rows"]["mean"]
            random_coverage = item["random_partition/covered_test_rows"]["mean"]
            state_matched_coverage = item["state_matched_partition/covered_test_rows"]["mean"]
            coverage = f"{conditional_coverage:.1f}/{random_coverage:.1f}/{state_matched_coverage:.1f}"
            lines.append(
                f"| {source} | {condition} | {f('conditional/operator_heterogeneity')} | "
                f"{f('random_partition/operator_heterogeneity')} | {f('state_matched_partition/operator_heterogeneity')} | "
                f"{f('conditional_minus_random/operator_heterogeneity')} | "
                f"{f('conditional_minus_state_matched/operator_heterogeneity')} | "
                f"{f('conditional_minus_state_matched/test_r2')} | {coverage} |"
            )
    lines.extend([
        "",
        "| source | condition | global anisotropy | conditional anisotropy | random-partition anisotropy | global stable rank | conditional stable rank | random stable rank |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for source in ("true", "predicted"):
        for condition in ("magnitude", "direction"):
            item = summary[source][condition]
            def g(key: str) -> str:
                return f"{item[key]['mean']:.6g}"
            lines.append(
                f"| {source} | {condition} | {g('global/covariance/anisotropy_max_over_mean')} | "
                f"{g('conditional/weighted_anisotropy')} | {g('random_partition/weighted_anisotropy')} | "
                f"{g('global/covariance/stable_rank')} | {g('conditional/weighted_stable_rank')} | "
                f"{g('random_partition/weighted_stable_rank')} |"
            )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The conditional operator is a fold-fitted linear summary, not a nonlinear Jacobian. A conditional-minus-random contrast indicates that the corresponding action grouping changes the measured operator geometry. The state-matched null additionally preserves coarse state-PC strata, making a positive conditional-minus-state-matched contrast stronger evidence against a state-only explanation. Operator heterogeneity is the normalized pairwise Frobenius distance between fitted bin operators; it is a direct local-dynamics diagnostic, but it does not establish that the grouping is causal or that lower commutator improves planning.",
        "",
        "The direction condition is a cosine-like projection onto a training-fold action axis and is included to test whether a magnitude-only finding survives a directional control. Global and conditional covariance statistics are descriptive representation measurements; they do not establish trained alignment-regularizer gains, online planning, task success, or a universal isotropy-versus-dynamics law.",
    ])
    path.write_text("\n".join(lines) + "\n")


def _save_trajectory_cache(
    trajectories: list[dict[str, Any]],
    path: Path,
    normalization_mode: str,
    windows_per_episode: int,
    max_horizon: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        true=np.stack([row["true"] for row in trajectories]),
        predicted=np.stack([row["predicted"] for row in trajectories]),
        action_features=np.stack([row["action_features"] for row in trajectories]),
        episode_id=np.asarray([row["episode_id"] for row in trajectories], dtype=np.int64),
        start_observation=np.asarray(
            [row["start_observation"] for row in trajectories], dtype=np.int64
        ),
        normalization=np.asarray(normalization_mode),
        windows_per_episode=np.asarray(int(windows_per_episode), dtype=np.int64),
        max_horizon=np.asarray(int(max_horizon), dtype=np.int64),
    )


def _load_trajectory_cache(
    path: Path,
    normalization_mode: str,
    windows_per_episode: int,
    max_horizon: int,
) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        cached_normalization = str(np.asarray(data["normalization"]).item())
        cached_windows = int(np.asarray(data["windows_per_episode"]).item())
        cached_horizon = int(np.asarray(data["max_horizon"]).item())
        if cached_normalization != normalization_mode:
            raise ValueError(
                f"trajectory cache normalization {cached_normalization!r} does not match {normalization_mode!r}"
            )
        if cached_windows != int(windows_per_episode) or cached_horizon != int(max_horizon):
            raise ValueError("trajectory cache protocol does not match requested windows or horizon")
        true = np.asarray(data["true"], dtype=np.float64)
        predicted = np.asarray(data["predicted"], dtype=np.float64)
        action_features = np.asarray(data["action_features"], dtype=np.float64)
        episode_id = np.asarray(data["episode_id"], dtype=np.int64)
        starts = np.asarray(data["start_observation"], dtype=np.int64)
    if not (true.shape[0] == predicted.shape[0] == action_features.shape[0] == episode_id.size == starts.size):
        raise ValueError("trajectory cache arrays have inconsistent row counts")
    return [
        {
            "episode_id": int(episode_id[index]),
            "start_observation": int(starts[index]),
            "true": true[index],
            "predicted": predicted[index],
            "action_features": action_features[index],
        }
        for index in range(true.shape[0])
    ]


def run_report(
    val_root: Path,
    archive: Path,
    policy: str,
    output: Path,
    windows_per_episode: int = 2,
    pca_dimension: int = 16,
    ridge: float = 1e-2,
    min_samples: int = 16,
    num_bins: int = DEFAULT_BINS,
    device_name: str = "cpu",
    normalization_mode: str = "public",
    state_strata: int = 8,
    trajectory_cache: Path | None = None,
) -> dict[str, Any]:
    with (val_root / "seq_lengths.pkl").open("rb") as handle:
        episode_count = len(pickle.load(handle))
    if trajectory_cache is not None and trajectory_cache.exists():
        trajectories = _load_trajectory_cache(
            trajectory_cache,
            normalization_mode,
            windows_per_episode,
            max_horizon=8,
        )
    else:
        from experiments.io_utils import load_policy

        model = load_policy(policy, device=device_name)
        trajectories = _load_trajectories(
            model, val_root, torch.device(device_name), normalization_mode, windows_per_episode, 8
        )
        if trajectory_cache is not None:
            _save_trajectory_cache(
                trajectories, trajectory_cache, normalization_mode, windows_per_episode, max_horizon=8
            )
    crossfit = _crossfit(
        trajectories, pca_dimension, ridge, num_bins, min_samples, state_strata=state_strata
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "artifact": "external_frozen_lewm_conditional_operator_geometry",
        "status": "frozen_external_conditional_linear_geometry_diagnostic",
        "source": {
            "archive": str(archive),
            "archive_bytes": int(archive.stat().st_size),
            "archive_sha256": _sha256(archive),
            "validation_root": str(val_root),
            "episodes": int(episode_count),
            "windows": int(len(trajectories)),
        },
        "model": {"policy": policy, "device": device_name},
        "protocol": {
            "context_length": CONTEXT_LENGTH,
            "max_horizon": 8,
            "windows_per_episode": int(windows_per_episode),
            "action_condition": "separate magnitude and direction quantile-bin conditions; direction uses a training-fold action axis",
            "bins": int(num_bins),
            "operator": "ridge row-wise linear transition fit within each action bin",
            "random_control": "independent size-matched random train/test partition plus state-matched within-stratum permutation",
            "crossfit": "leave-one-episode-out",
            "pca_dimension": int(pca_dimension),
            "ridge": float(ridge),
            "min_samples": int(min_samples),
            "state_strata": int(state_strata),
            "action_normalization": normalization_mode,
            "bootstrap": "episode-cluster bootstrap, 2000 replicates",
        },
        "crossfit": crossfit,
        "claim_boundary": [
            "This is a frozen external conditional linear diagnostic, not nonlinear Jacobian identification.",
            "The random partition controls bin-size effects but does not make action grouping a causal intervention.",
            "The state-matched null is a coarse first-state-PC stratification; sensitivity to the declared stratum count is reported separately when requested.",
            "Conditional geometry does not establish trained alignment-regularizer gains, online planning, task success, or universal isotropy-versus-dynamics behavior.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("notes.txt").write_text(
        "Frozen external conditional operator geometry audit completed.\n"
        "Action-magnitude bins, random partition control, and cross-fitting are explicit.\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--pca-dimension", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--min-samples", type=int, default=16)
    parser.add_argument("--bins", type=int, default=2)
    parser.add_argument("--action-normalization", choices=["public", "val_zscore", "raw_div100"], default="public")
    parser.add_argument("--state-strata", type=int, default=8)
    parser.add_argument("--trajectory-cache", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_report(
        val_root=args.val_root,
        archive=args.archive,
        policy=args.policy,
        output=args.output,
        windows_per_episode=args.windows_per_episode,
        pca_dimension=args.pca_dimension,
        ridge=args.ridge,
        min_samples=args.min_samples,
        num_bins=args.bins,
        device_name=args.device,
        normalization_mode=args.action_normalization,
        state_strata=args.state_strata,
        trajectory_cache=args.trajectory_cache,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
