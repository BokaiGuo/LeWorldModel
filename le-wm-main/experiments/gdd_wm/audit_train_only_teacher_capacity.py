"""Audit whether a fixed nonlinear residual readout adds latent information.

This is a capacity diagnostic, not a teacher or planning result.  A linear
action-summary model is fit on train roots, then both a fixed linear residual
readout and one fixed shallow MLP residual readout are fit on the same
train-only source/delta/action features.  The MLP is intentionally not tuned
on calibration or outcome data and requires CUDA for an actual run. The
explicit action-frame and target-mode controls make raw-coordinate and
root-cost confounds auditable rather than silently folding them into the
latent residual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

# Preserve the repository's direct-script entry point as well as
# ``python -m experiments...``. Direct execution otherwise places only the
# ``experiments/gdd_wm`` directory on sys.path.
if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch import nn

from experiments.gdd_wm.build_train_only_readout_teacher_package import (
    _action_hash,
    _action_summary,
    _feature_key,
    _fit_ridge,
    _load,
    _predict,
    _record_features,
)


_BILINEAR_PCA_RANK = 4


def _wilson_lower_bound(successes: int, total: int, *, confidence: float = 0.90) -> float:
    """One-sided Wilson lower bound for a calibration support proportion."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson support counts are invalid")
    if not 0.5 < confidence < 1.0:
        raise ValueError("Wilson confidence must lie in (0.5, 1)")
    proportion = successes / total
    z = NormalDist().inv_cdf(confidence)
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * np.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return float(max(0.0, (center - radius) / denominator))


def _bootstrap(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def _pair_consistency(rows: list[tuple[float, float]], margin: float) -> float | None:
    correct = total = 0
    for left_index, (left_score, left_cost) in enumerate(rows):
        for right_score, right_cost in rows[left_index + 1:]:
            score_delta = left_score - right_score
            # Both values are costs and therefore share the same ordering:
            # lower predicted cost should correspond to lower realized cost.
            # Utility-score audits elsewhere negate cost and intentionally use
            # the opposite sign; this capacity audit does not.
            cost_delta = left_cost - right_cost
            if abs(score_delta) <= margin or abs(cost_delta) <= margin:
                continue
            total += 1
            correct += int((score_delta > 0.0) == (cost_delta > 0.0))
    return None if total == 0 else float(correct / total)


def _root_metrics(
    scores: np.ndarray, costs: np.ndarray, roots: list[str], mask: np.ndarray, margin: float,
) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in np.flatnonzero(mask):
        grouped[roots[int(index)]].append(int(index))
    result: dict[str, dict[str, float | None]] = {}
    for root, indices in sorted(grouped.items()):
        root_scores = scores[indices]
        root_costs = costs[indices]
        # The readouts predict terminal *cost*, so lower is better.  Using
        # argmax here would deliberately select the worst candidate and
        # manufacture a negative planning result.
        selected = int(np.argmin(root_scores))
        result[root] = {
            "pairwise_consistency": _pair_consistency(
                [(float(root_scores[i]), float(root_costs[i])) for i in range(len(indices))], margin
            ),
            "regret": float(root_costs[selected] - np.min(root_costs)),
        }
    return result


def _macro_root_metrics(
    cell_metrics: dict[str, dict[str, float | None]],
    cell_to_root: dict[str, str],
) -> dict[str, dict[str, float | None]]:
    """Average context-specific cells within each unique root.

    Contexts are repeated diagnostic views of a root. They must never be
    merged into one candidate set before ranking, and must not be counted as
    independent bootstrap roots.
    """
    grouped: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    for cell, metrics in cell_metrics.items():
        grouped[cell_to_root[cell]].append(metrics)
    result: dict[str, dict[str, float | None]] = {}
    for root, values in sorted(grouped.items()):
        consistency = [item["pairwise_consistency"] for item in values]
        consistency = [float(value) for value in consistency if value is not None]
        result[root] = {
            "pairwise_consistency": None if not consistency else float(np.mean(consistency)),
            "regret": float(np.mean([float(item["regret"]) for item in values])),
        }
    return result


def _macro_mean_regret(
    prediction: np.ndarray,
    costs: np.ndarray,
    group_roots: list[str],
    base_roots: list[str],
    mask: np.ndarray,
    margin: float,
) -> float:
    """Return a root-macro regret for a frozen split.

    Calibration contains repeated context views of the same root.  Selecting
    a hyperparameter on the raw row average would therefore overweight roots
    with more contexts; this helper uses the same context-cell then root-macro
    aggregation as the final audit.
    """
    cell = _root_metrics(prediction, costs, group_roots, mask, margin)
    cell_to_root = {
        group: base
        for group, base, included in zip(group_roots, base_roots, mask)
        if included
    }
    root = _macro_root_metrics(cell, cell_to_root)
    if not root:
        raise ValueError("calibration split contains no complete root cells")
    return float(np.mean([float(item["regret"]) for item in root.values()]))


def _macro_root_regrets(
    prediction: np.ndarray,
    costs: np.ndarray,
    group_roots: list[str],
    base_roots: list[str],
    mask: np.ndarray,
    margin: float,
) -> dict[str, float]:
    cell = _root_metrics(prediction, costs, group_roots, mask, margin)
    cell_to_root = {
        group: base
        for group, base, included in zip(group_roots, base_roots, mask)
        if included
    }
    root = _macro_root_metrics(cell, cell_to_root)
    return {key: float(value["regret"]) for key, value in root.items()}


def _macro_root_metric_map(
    prediction: np.ndarray,
    costs: np.ndarray,
    group_roots: list[str],
    base_roots: list[str],
    mask: np.ndarray,
    margin: float,
) -> dict[str, dict[str, float | None]]:
    cell = _root_metrics(prediction, costs, group_roots, mask, margin)
    cell_to_root = {
        group: base
        for group, base, included in zip(group_roots, base_roots, mask)
        if included
    }
    return _macro_root_metrics(cell, cell_to_root)


def _fit_linear_residual_predictions(
    x: np.ndarray,
    action_x: np.ndarray,
    train_target: np.ndarray,
    train_mask: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the fixed action baseline plus a linear latent residual readout."""
    action_fit, residual_fit = _fit_linear_residual_models(
        x, action_x, train_target, train_mask, alpha
    )
    action_pred = _predict(action_x, action_fit)
    return action_pred, action_pred + _predict(x, residual_fit)


def _fit_linear_residual_models(
    x: np.ndarray,
    action_x: np.ndarray,
    train_target: np.ndarray,
    train_mask: np.ndarray,
    alpha: float,
):
    """Return the two frozen ridge fits used by the runtime-safe readout."""
    action_fit = _fit_ridge(action_x[train_mask], train_target, alpha)
    action_pred = _predict(action_x, action_fit)
    residual_train = train_target - action_pred[train_mask]
    residual_fit = _fit_ridge(x[train_mask], residual_train, alpha)
    return action_fit, residual_fit


class _ResidualMLP(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _fit_mlp_residual_model(
    features: np.ndarray,
    residual_train: np.ndarray,
    train_mask: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Fit one fixed train-only residual MLP and return deployment arrays."""
    feature_mean = np.mean(features[train_mask], axis=0)
    feature_scale = np.maximum(np.std(features[train_mask], axis=0), 1.0e-6)
    residual_mean = float(np.mean(residual_train))
    residual_scale = max(float(np.std(residual_train)), 1.0e-6)
    torch.manual_seed(seed)
    model = _ResidualMLP(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_x = torch.as_tensor(
        (features[train_mask] - feature_mean) / feature_scale,
        dtype=torch.float32,
        device=device,
    )
    train_y = torch.as_tensor(
        (residual_train - residual_mean) / residual_scale,
        dtype=torch.float32,
        device=device,
    )
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(train_x) - train_y) ** 2)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        all_x = torch.as_tensor(
            (features - feature_mean) / feature_scale,
            dtype=torch.float32,
            device=device,
        )
        prediction = model(all_x).cpu().numpy()
    state = {
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_scale": np.asarray(feature_scale, dtype=np.float32),
        "residual_mean": float(residual_mean),
        "residual_scale": float(residual_scale),
    }
    for key, value in model.state_dict().items():
        state[key.replace("net.", "layer_").replace(".", "_")] = (
            value.detach().cpu().numpy().astype(np.float32)
        )
    return prediction * residual_scale + residual_mean, state


def _fit_mlp_residual_predictions(
    features: np.ndarray,
    residual_train: np.ndarray,
    train_mask: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    prediction, _ = _fit_mlp_residual_model(
        features,
        residual_train,
        train_mask,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
    )
    return prediction


def _fit_listwise_mlp_predictions(
    features: np.ndarray,
    train_target: np.ndarray,
    train_mask: np.ndarray,
    group_roots: list[str],
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    temperature: float = 0.25,
    mse_weight: float = 0.1,
) -> np.ndarray:
    """Fit a train-only root-grouped cost ranker.

    Cross entropy between target and predicted within-root soft rankings makes
    the training objective match candidate selection.  The small MSE term
    fixes scale without dominating the ranking objective.
    """
    feature_mean = np.mean(features[train_mask], axis=0)
    feature_scale = np.maximum(np.std(features[train_mask], axis=0), 1.0e-6)
    train_groups = np.asarray(group_roots, dtype=object)[train_mask]
    grouped: dict[str, list[int]] = defaultdict(list)
    for position, root in enumerate(train_groups):
        grouped[str(root)].append(position)
    group_sizes = {len(indices) for indices in grouped.values()}
    if len(group_sizes) != 1 or not group_sizes or min(group_sizes) < 2:
        raise ValueError("listwise training requires equal candidate counts >=2 per train cell")
    group_index = np.asarray([grouped[root] for root in sorted(grouped)], dtype=np.int64)
    target_scale = max(float(np.std(train_target)), 1.0e-6)
    normalized_target = np.asarray(train_target, dtype=np.float64) / target_scale

    torch.manual_seed(seed)
    model = _ResidualMLP(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_x = torch.as_tensor(
        (features[train_mask] - feature_mean) / feature_scale,
        dtype=torch.float32,
        device=device,
    )
    train_y = torch.as_tensor(normalized_target, dtype=torch.float32, device=device)
    tensor_group_index = torch.as_tensor(group_index, dtype=torch.long, device=device)
    grouped_target = train_y[tensor_group_index]
    target_probability = torch.softmax(-grouped_target / float(temperature), dim=1)
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train_x)
        grouped_prediction = prediction[tensor_group_index]
        ranking_loss = -torch.mean(torch.sum(
            target_probability
            * torch.log_softmax(-grouped_prediction / float(temperature), dim=1),
            dim=1,
        ))
        loss = ranking_loss + float(mse_weight) * torch.mean((prediction - train_y) ** 2)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        all_x = torch.as_tensor(
            (features - feature_mean) / feature_scale,
            dtype=torch.float32,
            device=device,
        )
        prediction = model(all_x).cpu().numpy()
    return prediction * target_scale


def _shuffle_predicted_delta_within_cells(
    features: np.ndarray,
    group_roots: list[str],
    latent_dimension: int,
) -> np.ndarray:
    """Break candidate-to-prediction pairing without crossing root/context."""
    shuffled = np.asarray(features, dtype=np.float64).copy()
    start = int(latent_dimension)
    stop = 2 * int(latent_dimension)
    if start <= 0 or shuffled.ndim != 2 or shuffled.shape[1] < stop:
        raise ValueError("features do not contain source and predicted-delta blocks")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, root in enumerate(group_roots):
        grouped[str(root)].append(index)
    for indices in grouped.values():
        if len(indices) < 2:
            raise ValueError("prediction shuffle requires at least two candidates per cell")
        block = shuffled[np.asarray(indices), start:stop].copy()
        shuffled[np.asarray(indices), start:stop] = np.roll(block, shift=1, axis=0)
    return shuffled


def _bilinear_transfer_features(
    source: np.ndarray,
    delta: np.ndarray,
    action: np.ndarray,
    *,
    include_predicted_dynamics: bool,
) -> np.ndarray:
    """Build the minimal state-action interaction needed by a fixed bank.

    ``source x action`` is the no-dynamics decision-repair baseline.  The full
    variant adds predicted displacement and its action/state interactions, so
    any gain over the baseline is attributable to candidate-conditioned
    predicted dynamics rather than to root-conditioned action values alone.
    """
    source = np.asarray(source, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if source.ndim != 2 or delta.shape != source.shape or action.ndim != 2:
        raise ValueError("bilinear features require aligned rank-2 arrays")
    source_action = np.einsum("ni,nj->nij", source, action).reshape(len(source), -1)
    blocks = [action, source_action]
    if include_predicted_dynamics:
        delta_action = np.einsum("ni,nj->nij", delta, action).reshape(len(delta), -1)
        source_delta = np.einsum("ni,nj->nij", source, delta).reshape(len(source), -1)
        blocks.extend([delta, delta_action, source_delta])
    return np.concatenate(blocks, axis=1)


def _fit_train_pca(
    values: np.ndarray, train_mask: np.ndarray, rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a deterministic PCA chart using train rows only.

    The previous full interaction formed a 36,864-dimensional
    ``source x delta`` block and made ridge fitting quadratic in that width.
    A small train-only chart keeps the state-conditioned action interaction
    identifiable without reading calibration/outcome geometry.
    """
    values = np.asarray(values, dtype=np.float64)
    train_mask = np.asarray(train_mask, dtype=bool)
    if values.ndim != 2 or train_mask.shape != (len(values),):
        raise ValueError("PCA values/mask must be aligned rank-2/rank-1 arrays")
    if rank < 1 or int(train_mask.sum()) < 2:
        raise ValueError("PCA requires rank >= 1 and at least two train rows")
    train = values[train_mask]
    mean = np.mean(train, axis=0)
    _, _, right = np.linalg.svd(train - mean, full_matrices=False)
    effective_rank = min(int(rank), right.shape[0], right.shape[1])
    components = right[:effective_rank].copy()
    # SVD vectors have an arbitrary sign. Canonicalize it so exported
    # artifacts and regression tests are byte-stable across reruns.
    for index in range(len(components)):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0.0:
            components[index] *= -1.0
    return mean, components


def _project_pca(
    values: np.ndarray, mean: np.ndarray, components: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    components = np.asarray(components, dtype=np.float64)
    if values.ndim != 2 or mean.shape != (values.shape[1],):
        raise ValueError("PCA projection mean does not match input width")
    if components.ndim != 2 or components.shape[1] != values.shape[1]:
        raise ValueError("PCA components do not match input width")
    return (values - mean) @ components.T


def _low_rank_bilinear_features(
    source: np.ndarray,
    delta: np.ndarray,
    action: np.ndarray,
    *,
    source_mean: np.ndarray,
    source_components: np.ndarray,
    delta_mean: np.ndarray,
    delta_components: np.ndarray,
    include_predicted_dynamics: bool,
) -> np.ndarray:
    """Build deployable low-rank state/action transfer features."""
    source_projected = _project_pca(source, source_mean, source_components)
    delta_projected = _project_pca(delta, delta_mean, delta_components)
    action = np.asarray(action, dtype=np.float64)
    if action.ndim != 2 or len(action) != len(source_projected):
        raise ValueError("low-rank bilinear action rows do not align")
    source_action = np.einsum(
        "ni,nj->nij", source_projected, action,
    ).reshape(len(action), -1)
    blocks = [action, source_action]
    if include_predicted_dynamics:
        delta_action = np.einsum(
            "ni,nj->nij", delta_projected, action,
        ).reshape(len(action), -1)
        source_delta = np.einsum(
            "ni,nj->nij", source_projected, delta_projected,
        ).reshape(len(action), -1)
        blocks.extend([delta_projected, delta_action, source_delta])
    return np.concatenate(blocks, axis=1)


def _crossfit_root_centered_ridge_predictions(
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    group_roots: list[str],
    base_roots: list[str],
    *,
    alpha: float,
    folds: int = 5,
) -> np.ndarray:
    """Return root-disjoint train OOF predictions for a fixed ridge family."""
    train_roots = sorted({
        str(base_roots[index]) for index in np.flatnonzero(train_mask)
    })
    if len(train_roots) < folds:
        raise ValueError("cross-fitting requires at least one train root per fold")
    fold_for_root = {root: index % folds for index, root in enumerate(train_roots)}
    prediction = np.full(len(target), np.nan, dtype=np.float64)
    root_array = np.asarray(base_roots, dtype=object)
    group_array = np.asarray(group_roots, dtype=object)
    for fold in range(folds):
        validation_roots = {
            root for root, assigned in fold_for_root.items() if assigned == fold
        }
        validation_mask = train_mask & np.asarray([
            str(root) in validation_roots for root in root_array
        ])
        fit_mask = train_mask & ~validation_mask
        fit_target = np.asarray(target[fit_mask], dtype=np.float64).copy()
        fit_groups = group_array[fit_mask]
        for group in np.unique(fit_groups):
            local = fit_groups == group
            fit_target[local] -= np.mean(fit_target[local])
        fit = _fit_ridge(features[fit_mask], fit_target, alpha)
        prediction[validation_mask] = _predict(features[validation_mask], fit)
    if np.any(~np.isfinite(prediction[train_mask])):
        raise RuntimeError("cross-fitting did not cover every train row")
    return prediction


def _crossfit_low_rank_bilinear_predictions(
    source: np.ndarray,
    delta: np.ndarray,
    action: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    group_roots: list[str],
    base_roots: list[str],
    *,
    include_predicted_dynamics: bool,
    alpha: float,
    rank: int = _BILINEAR_PCA_RANK,
    folds: int = 5,
) -> np.ndarray:
    """Cross-fit PCA and ridge together so even the chart is root-disjoint."""
    train_roots = sorted({
        str(base_roots[index]) for index in np.flatnonzero(train_mask)
    })
    if len(train_roots) < folds:
        raise ValueError("cross-fitting requires at least one train root per fold")
    fold_for_root = {root: index % folds for index, root in enumerate(train_roots)}
    prediction = np.full(len(target), np.nan, dtype=np.float64)
    root_array = np.asarray(base_roots, dtype=object)
    group_array = np.asarray(group_roots, dtype=object)
    for fold in range(folds):
        validation_roots = {
            root for root, assigned in fold_for_root.items() if assigned == fold
        }
        validation_mask = train_mask & np.asarray([
            str(root) in validation_roots for root in root_array
        ])
        fit_mask = train_mask & ~validation_mask
        source_mean, source_components = _fit_train_pca(source, fit_mask, rank)
        delta_mean, delta_components = _fit_train_pca(delta, fit_mask, rank)
        features = _low_rank_bilinear_features(
            source,
            delta,
            action,
            source_mean=source_mean,
            source_components=source_components,
            delta_mean=delta_mean,
            delta_components=delta_components,
            include_predicted_dynamics=include_predicted_dynamics,
        )
        fit_target = np.asarray(target[fit_mask], dtype=np.float64).copy()
        fit_groups = group_array[fit_mask]
        for group in np.unique(fit_groups):
            local = fit_groups == group
            fit_target[local] -= np.mean(fit_target[local])
        fit = _fit_ridge(features[fit_mask], fit_target, alpha)
        prediction[validation_mask] = _predict(features[validation_mask], fit)
    if np.any(~np.isfinite(prediction[train_mask])):
        raise RuntimeError("low-rank cross-fitting did not cover every train row")
    return prediction


def _goalframe_action_summary(row: dict[str, Any]) -> np.ndarray:
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
    midpoint = max(1, canonical.shape[0] // 2)
    second_half = canonical[midpoint:] if midpoint < canonical.shape[0] else canonical[-1:]
    return np.concatenate([
        np.mean(canonical, axis=0),
        np.std(canonical, axis=0),
        canonical[0],
        canonical[-1],
        np.mean(canonical[:midpoint], axis=0),
        np.mean(second_half, axis=0),
    ])


def _proposal_relative_action_summaries(
    actions: np.ndarray,
    group_roots: list[str],
) -> np.ndarray:
    """Express each action summary in its local proposal-cloud chart."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 12:
        raise ValueError("proposal-relative actions require an [N, 12] summary matrix")
    if len(group_roots) != len(actions):
        raise ValueError("proposal groups must align with action rows")
    groups = np.asarray(group_roots, dtype=object)
    normalized = np.empty_like(actions)
    for group in np.unique(groups):
        mask = groups == group
        local = actions[mask]
        if len(local) < 2:
            raise ValueError("proposal-relative charts require at least two candidates per cell")
        mean = np.mean(local, axis=0)
        scale = np.maximum(np.std(local, axis=0), 1.0e-6)
        normalized[mask] = (local - mean) / scale
    return normalized


def _feature_vector(
    row: dict[str, Any], predicted: np.ndarray, action_frame: str,
    goal_conditioning: str = "none",
) -> np.ndarray:
    if action_frame in ("raw", "proposal_relative"):
        base = _record_features(row, predicted, "source_delta_action")
    elif action_frame == "goal_relative":
        source = np.asarray(row.get("source_latent_model"), dtype=np.float64)
        if source.ndim != 1 or source.shape != predicted.shape:
            raise ValueError("source_latent_model and predicted latent dimensions do not match")
        base = np.concatenate([source, predicted - source, _goalframe_action_summary(row)])
    else:
        raise ValueError(f"unsupported action frame: {action_frame!r}")
    if goal_conditioning == "none":
        return base
    if goal_conditioning != "latent_relative":
        raise ValueError(f"unsupported goal conditioning: {goal_conditioning!r}")
    source = np.asarray(row.get("source_latent_model"), dtype=np.float64)
    goal = np.asarray(row.get("goal_latent_model"), dtype=np.float64)
    if source.ndim != 1 or source.shape != predicted.shape or goal.shape != predicted.shape:
        raise ValueError("goal_latent_model must match source/predicted latent dimensions")
    return np.concatenate([base, goal - source, predicted - goal])


def _load_context_rows(
    candidate_datasets: list[Path], arrays_paths: list[Path],
    *, require_multiple_contexts: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[np.ndarray]]:
    minimum = 2 if require_multiple_contexts else 1
    if len(candidate_datasets) < minimum or len(candidate_datasets) != len(arrays_paths):
        if require_multiple_contexts:
            raise ValueError("at least two candidate datasets and matching arrays files are required")
        raise ValueError("at least one candidate dataset and matching arrays file are required")
    loaded = [_load(path) for path in candidate_datasets]
    maps: list[dict[tuple[str, str], dict[str, Any]]] = []
    for path, (_, source_rows) in zip(candidate_datasets, loaded):
        current: dict[tuple[str, str], dict[str, Any]] = {}
        for row in source_rows:
            key = (str(row["instance_id"]), _action_hash(row))
            if key in current:
                raise ValueError(f"{path}: duplicate root/action pair {key}")
            current[key] = row
        maps.append(current)
    common = set(maps[0])
    for current in maps[1:]:
        common &= set(current)
    if not common:
        raise ValueError("candidate datasets have no common root/action records")
    arrays = [np.load(path, allow_pickle=False) for path in arrays_paths]
    contexts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, (path, (metadata, _)) in enumerate(zip(candidate_datasets, loaded)):
        source_hash = str(metadata.get("candidate_bank_hash", "")) or hashlib.sha256(path.read_bytes()).hexdigest()[:20]
        context_id = f"capacity_context_{index:02d}_{source_hash}"
        contexts.append({"context_id": context_id, "source": str(path.resolve())})
        for root, action_hash in sorted(common):
            row = dict(maps[index][(root, action_hash)])
            row["instance_id"] = root
            row["candidate_id"] = f"action_{action_hash}"
            row["candidate_bank_context_id"] = context_id
            row["teacher_set_id"] = f"{root}__{context_id}"
            row["predicted_latent_ref"] = f"{context_id}::{row['predicted_latent_ref']}"
            rows.append(row)
    return rows, contexts, arrays


def run(
    *, candidate_datasets: list[Path], arrays_paths: list[Path], output: Path,
    epochs: int = 800, learning_rate: float = 1.0e-2, weight_decay: float = 1.0e-3,
    margin: float = 1.0e-3, draws: int = 10000, seed: int = 20261016,
    action_frame: str = "raw", target_mode: str = "absolute",
    goal_conditioning: str = "none",
    require_multiple_contexts: bool = True,
    mlp_residual_start_iteration: int = 7,
    mlp_planner_call_index: int = 1,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this capacity audit; refusing CPU fallback")
    if mlp_residual_start_iteration < 0 or mlp_planner_call_index < 0:
        raise ValueError("MLP residual iteration/call gates must be nonnegative")
    rows, contexts, arrays = _load_context_rows(
        candidate_datasets,
        arrays_paths,
        require_multiple_contexts=require_multiple_contexts,
    )
    features: list[np.ndarray] = []
    no_dynamics_features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    targets: list[float] = []
    group_roots: list[str] = []
    base_roots: list[str] = []
    splits: list[str] = []
    context_lookup = {item["context_id"]: index for index, item in enumerate(contexts)}
    for row in rows:
        context_index = context_lookup[str(row["candidate_bank_context_id"])]
        key = _feature_key(row)
        key = key.split("::", 1)[-1] if "::" in key else key
        value = np.asarray(arrays[context_index][key], dtype=np.float64)
        features.append(_feature_vector(row, value, action_frame, goal_conditioning))
        action_feature = (
            _goalframe_action_summary(row)
            if action_frame == "goal_relative"
            else _action_summary(row["action_sequence"])
        )
        source_feature = np.asarray(row["source_latent_model"], dtype=np.float64)
        no_dynamics_features.append(np.concatenate([source_feature, action_feature]))
        actions.append(action_feature)
        targets.append(float(row["real_task_cost"]))
        context_id = str(row["candidate_bank_context_id"])
        group_roots.append(f"{context_id}::{row['instance_id']}")
        base_roots.append(str(row["instance_id"]))
        splits.append(str(row["split"]))
    x = np.asarray(features, dtype=np.float64)
    no_dynamics_x = np.asarray(no_dynamics_features, dtype=np.float64)
    action_x = np.asarray(actions, dtype=np.float64)
    latent_dimension = int(np.asarray(rows[0]["source_latent_model"]).size)
    if action_frame == "proposal_relative":
        action_x = _proposal_relative_action_summaries(action_x, group_roots)
        x[:, 2 * latent_dimension : 2 * latent_dimension + action_x.shape[1]] = action_x
        no_dynamics_x = np.concatenate([x[:, :latent_dimension], action_x], axis=1)
    y = np.asarray(targets, dtype=np.float64)
    train_mask = np.asarray([split == "train" for split in splits])
    calibration_mask = np.asarray([split == "calibration" for split in splits])
    outcome_mask = np.asarray([split == "outcome" for split in splits])
    if target_mode == "absolute":
        train_target = y[train_mask]
    elif target_mode == "root_centered":
        train_target = y[train_mask].copy()
        train_roots = np.asarray(group_roots)[train_mask]
        for root in np.unique(train_roots):
            root_mask = train_roots == root
            train_target[root_mask] -= np.mean(train_target[root_mask])
    else:
        raise ValueError(f"unsupported target mode: {target_mode!r}")
    action_pred, linear_pred = _fit_linear_residual_predictions(
        x, action_x, train_target, train_mask, 1.0e-2
    )
    residual_train = train_target - action_pred[train_mask]

    device = torch.device("cuda")
    shuffled_x = _shuffle_predicted_delta_within_cells(x, group_roots, latent_dimension)
    source_block = x[:, :latent_dimension]
    delta_block = x[:, latent_dimension : 2 * latent_dimension]
    shuffled_delta_block = shuffled_x[:, latent_dimension : 2 * latent_dimension]
    source_pca_mean, source_pca_components = _fit_train_pca(
        source_block, train_mask, _BILINEAR_PCA_RANK,
    )
    delta_pca_mean, delta_pca_components = _fit_train_pca(
        delta_block, train_mask, _BILINEAR_PCA_RANK,
    )
    low_rank_kwargs = {
        "source_mean": source_pca_mean,
        "source_components": source_pca_components,
        "delta_mean": delta_pca_mean,
        "delta_components": delta_pca_components,
    }
    bilinear_no_dynamics_x = _low_rank_bilinear_features(
        source_block, delta_block, action_x,
        include_predicted_dynamics=False, **low_rank_kwargs,
    )
    bilinear_full_x = _low_rank_bilinear_features(
        source_block, delta_block, action_x,
        include_predicted_dynamics=True, **low_rank_kwargs,
    )
    bilinear_shuffled_x = _low_rank_bilinear_features(
        source_block, shuffled_delta_block, action_x,
        include_predicted_dynamics=True, **low_rank_kwargs,
    )
    bilinear_no_dynamics_fit = _fit_ridge(
        bilinear_no_dynamics_x[train_mask], train_target, 1.0e-2
    )
    bilinear_full_fit = _fit_ridge(bilinear_full_x[train_mask], train_target, 1.0e-2)
    bilinear_shuffled_fit = _fit_ridge(
        bilinear_shuffled_x[train_mask], train_target, 1.0e-2
    )
    bilinear_no_dynamics_pred = _predict(
        bilinear_no_dynamics_x, bilinear_no_dynamics_fit
    )
    bilinear_full_pred = _predict(bilinear_full_x, bilinear_full_fit)
    bilinear_shuffled_pred = _predict(bilinear_shuffled_x, bilinear_shuffled_fit)
    crossfit_action_pred = _crossfit_root_centered_ridge_predictions(
        action_x, y, train_mask, group_roots, base_roots, alpha=1.0e-2
    )
    crossfit_bilinear_no_dynamics_pred = _crossfit_low_rank_bilinear_predictions(
        source_block,
        delta_block,
        action_x,
        y,
        train_mask,
        group_roots,
        base_roots,
        include_predicted_dynamics=False,
        alpha=1.0e-2,
    )
    crossfit_bilinear_full_pred = _crossfit_low_rank_bilinear_predictions(
        source_block,
        delta_block,
        action_x,
        y,
        train_mask,
        group_roots,
        base_roots,
        include_predicted_dynamics=True,
        alpha=1.0e-2,
    )
    mlp_pred = action_pred + _fit_mlp_residual_predictions(
        x, residual_train, train_mask, epochs=epochs,
        learning_rate=learning_rate, weight_decay=weight_decay,
        seed=seed, device=device,
    )
    no_dynamics_mlp_residual, no_dynamics_mlp_artifact = _fit_mlp_residual_model(
        no_dynamics_x, residual_train, train_mask, epochs=epochs,
        learning_rate=learning_rate, weight_decay=weight_decay,
        seed=seed, device=device,
    )
    no_dynamics_mlp_pred = action_pred + no_dynamics_mlp_residual
    shuffled_mlp_pred = action_pred + _fit_mlp_residual_predictions(
        shuffled_x, residual_train, train_mask, epochs=epochs,
        learning_rate=learning_rate, weight_decay=weight_decay,
        seed=seed, device=device,
    )
    listwise_pred = _fit_listwise_mlp_predictions(
        x, train_target, train_mask, group_roots, epochs=epochs,
        learning_rate=learning_rate, weight_decay=weight_decay,
        seed=seed, device=device,
    )
    listwise_no_dynamics_pred = _fit_listwise_mlp_predictions(
        no_dynamics_x, train_target, train_mask, group_roots, epochs=epochs,
        learning_rate=learning_rate, weight_decay=weight_decay,
        seed=seed, device=device,
    )
    listwise_shuffled_pred = _fit_listwise_mlp_predictions(
        shuffled_x, train_target, train_mask, group_roots, epochs=epochs,
        learning_rate=learning_rate, weight_decay=weight_decay,
        seed=seed, device=device,
    )

    predictions = {
        "action_only": action_pred,
        "linear_residual": linear_pred,
        "mlp_residual": mlp_pred,
        "mlp_no_dynamics_source_action": no_dynamics_mlp_pred,
        "mlp_shuffled_prediction": shuffled_mlp_pred,
        "listwise_mlp": listwise_pred,
        "listwise_no_dynamics_source_action": listwise_no_dynamics_pred,
        "listwise_shuffled_prediction": listwise_shuffled_pred,
        "bilinear_no_dynamics_source_action": bilinear_no_dynamics_pred,
        "bilinear_full_predicted_dynamics": bilinear_full_pred,
        "bilinear_shuffled_prediction": bilinear_shuffled_pred,
    }

    calibration_selection: dict[str, Any] | None = None
    nonlinear_calibration_selection: dict[str, Any] | None = None
    crossfit_selected = "action_only"
    nonlinear_selected = "action_only"
    selected_alpha_for_artifact = 1.0e-2
    gate_passed = False
    if calibration_mask.any():
        # The grid is fixed before looking at outcome roots.  Calibration is
        # used only to control the high-dimensional latent residual's
        # variance; outcome remains untouched until the final report.
        alpha_grid = (1.0e-2, 1.0e-1, 1.0, 1.0e1, 1.0e2, 1.0e3)
        calibration_scores: dict[str, float] = {}
        candidate_predictions: dict[float, np.ndarray] = {}
        action_calibration_roots = _macro_root_regrets(
            action_pred,
            y,
            group_roots,
            base_roots,
            calibration_mask,
            margin,
        )
        for candidate_alpha in alpha_grid:
            _, candidate_prediction = _fit_linear_residual_predictions(
                x, action_x, train_target, train_mask, candidate_alpha
            )
            candidate_predictions[candidate_alpha] = candidate_prediction
            calibration_scores[str(candidate_alpha)] = _macro_mean_regret(
                candidate_prediction,
                y,
                group_roots,
                base_roots,
                calibration_mask,
                margin,
            )
        selected_alpha = min(
            alpha_grid,
            key=lambda candidate: (calibration_scores[str(candidate)], candidate),
        )
        selected_alpha_for_artifact = float(selected_alpha)
        predictions["calibrated_linear_residual"] = candidate_predictions[selected_alpha]
        selected_calibration_roots = _macro_root_regrets(
            candidate_predictions[selected_alpha],
            y,
            group_roots,
            base_roots,
            calibration_mask,
            margin,
        )
        common_calibration_roots = sorted(
            set(action_calibration_roots) & set(selected_calibration_roots)
        )
        calibration_delta = np.asarray(
            [
                selected_calibration_roots[root] - action_calibration_roots[root]
                for root in common_calibration_roots
            ],
            dtype=np.float64,
        )
        calibration_delta_ci = _bootstrap(
            calibration_delta,
            np.random.default_rng(seed + 991),
            draws,
        )
        gate_passed = bool(calibration_delta_ci["ci95"][1] < 0.0)
        predictions["calibration_safe_gate"] = (
            candidate_predictions[selected_alpha] if gate_passed else action_pred
        )
        calibration_selection = {
            "alpha_grid": [float(value) for value in alpha_grid],
            "calibration_mean_regret": calibration_scores,
            "selected_alpha": float(selected_alpha),
            "action_only_mean_regret": float(np.mean(list(action_calibration_roots.values()))),
            "selected_mean_regret": float(np.mean(list(selected_calibration_roots.values()))),
            "selected_minus_action_regret_delta": {
                "mean": float(np.mean(calibration_delta)),
                "ci95": calibration_delta_ci["ci95"],
                "root_count": len(common_calibration_roots),
            },
            "safe_gate": {
                "rule": "enable residual only when calibration paired regret CI95 upper bound < 0",
                "passed": gate_passed,
                "fallback": not gate_passed,
                "fallback_target": "action_only",
            },
            "selection_split": "calibration",
            "evaluation_split": "outcome",
            "outcome_used_for_selection": False,
        }
        calibration_predictions = {
            "action_only": action_pred,
            "mlp_no_dynamics_source_action": no_dynamics_mlp_pred,
            "mlp_residual": mlp_pred,
            "mlp_shuffled_prediction": shuffled_mlp_pred,
        }
        calibration_regrets = {
            name: _macro_root_regrets(
                prediction, y, group_roots, base_roots, calibration_mask, margin
            )
            for name, prediction in calibration_predictions.items()
        }

        def paired_gate_delta(left: str, right: str, offset: int) -> dict[str, Any]:
            common = sorted(set(calibration_regrets[left]) & set(calibration_regrets[right]))
            delta = np.asarray([
                calibration_regrets[left][root] - calibration_regrets[right][root]
                for root in common
            ], dtype=np.float64)
            return _bootstrap(delta, np.random.default_rng(seed + offset), draws)

        no_dynamics_vs_action = paired_gate_delta(
            "mlp_no_dynamics_source_action", "action_only", 1201
        )
        full_vs_no_dynamics = paired_gate_delta(
            "mlp_residual", "mlp_no_dynamics_source_action", 1202
        )
        full_dynamics_passed = bool(full_vs_no_dynamics["ci95"][1] < 0.0)
        no_dynamics_passed = bool(no_dynamics_vs_action["ci95"][1] < 0.0)
        if full_dynamics_passed:
            nonlinear_selected = "mlp_residual"
        elif no_dynamics_passed:
            nonlinear_selected = "mlp_no_dynamics_source_action"
        else:
            nonlinear_selected = "action_only"
        predictions["calibration_safe_nonlinear"] = calibration_predictions[nonlinear_selected]
        nonlinear_calibration_selection = {
            "rule": (
                "select full MLP only when its paired calibration regret CI95 beats "
                "the no-dynamics MLP; otherwise select no-dynamics MLP only when it "
                "beats action-only; otherwise fail closed to action-only"
            ),
            "full_vs_no_dynamics_regret_delta": full_vs_no_dynamics,
            "no_dynamics_vs_action_regret_delta": no_dynamics_vs_action,
            "selected": nonlinear_selected,
            "full_dynamics_gate_passed": full_dynamics_passed,
            "no_dynamics_gate_passed": no_dynamics_passed,
            "selection_split": "calibration",
            "evaluation_split": "outcome",
            "outcome_used_for_selection": False,
            "status": "exploratory_gate_frozen_after_initial_capacity_pilot",
        }
        listwise_calibration_predictions = {
            "action_only": action_pred,
            "listwise_no_dynamics_source_action": listwise_no_dynamics_pred,
            "listwise_mlp": listwise_pred,
        }
        listwise_calibration_regrets = {
            name: _macro_root_regrets(
                prediction, y, group_roots, base_roots, calibration_mask, margin
            )
            for name, prediction in listwise_calibration_predictions.items()
        }

        def listwise_gate_delta(left: str, right: str, offset: int) -> dict[str, Any]:
            common = sorted(
                set(listwise_calibration_regrets[left])
                & set(listwise_calibration_regrets[right])
            )
            delta = np.asarray([
                listwise_calibration_regrets[left][root]
                - listwise_calibration_regrets[right][root]
                for root in common
            ], dtype=np.float64)
            return _bootstrap(delta, np.random.default_rng(seed + offset), draws)

        listwise_no_dynamics_vs_action = listwise_gate_delta(
            "listwise_no_dynamics_source_action", "action_only", 1301
        )
        listwise_full_vs_no_dynamics = listwise_gate_delta(
            "listwise_mlp", "listwise_no_dynamics_source_action", 1302
        )
        listwise_full_passed = bool(
            listwise_full_vs_no_dynamics["ci95"][1] < 0.0
        )
        listwise_no_dynamics_passed = bool(
            listwise_no_dynamics_vs_action["ci95"][1] < 0.0
        )
        if listwise_full_passed:
            listwise_selected = "listwise_mlp"
        elif listwise_no_dynamics_passed:
            listwise_selected = "listwise_no_dynamics_source_action"
        else:
            listwise_selected = "action_only"
        predictions["calibration_safe_listwise"] = listwise_calibration_predictions[
            listwise_selected
        ]
        nonlinear_calibration_selection["listwise"] = {
            "rule": (
                "select full listwise MLP only when it beats the no-dynamics "
                "listwise MLP on paired calibration regret; otherwise require "
                "no-dynamics to beat action-only; otherwise fail closed"
            ),
            "full_vs_no_dynamics_regret_delta": listwise_full_vs_no_dynamics,
            "no_dynamics_vs_action_regret_delta": listwise_no_dynamics_vs_action,
            "selected": listwise_selected,
            "full_dynamics_gate_passed": listwise_full_passed,
            "no_dynamics_gate_passed": listwise_no_dynamics_passed,
            "selection_split": "calibration",
            "evaluation_split": "outcome",
            "outcome_used_for_selection": False,
            "status": "exploratory_objective_after_mse_capacity_diagnosis",
        }
        bilinear_calibration_predictions = {
            "action_only": action_pred,
            "bilinear_no_dynamics_source_action": bilinear_no_dynamics_pred,
            "bilinear_full_predicted_dynamics": bilinear_full_pred,
        }
        bilinear_calibration_regrets = {
            name: _macro_root_regrets(
                prediction, y, group_roots, base_roots, calibration_mask, margin
            )
            for name, prediction in bilinear_calibration_predictions.items()
        }
        bilinear_calibration_metrics = {
            name: _macro_root_metric_map(
                prediction, y, group_roots, base_roots, calibration_mask, margin
            )
            for name, prediction in bilinear_calibration_predictions.items()
        }

        def bilinear_gate_delta(left: str, right: str, offset: int) -> dict[str, Any]:
            common = sorted(
                set(bilinear_calibration_regrets[left])
                & set(bilinear_calibration_regrets[right])
            )
            delta = np.asarray([
                bilinear_calibration_regrets[left][root]
                - bilinear_calibration_regrets[right][root]
                for root in common
            ], dtype=np.float64)
            return _bootstrap(delta, np.random.default_rng(seed + offset), draws)

        bilinear_no_dynamics_vs_action = bilinear_gate_delta(
            "bilinear_no_dynamics_source_action", "action_only", 1401
        )
        bilinear_full_vs_no_dynamics = bilinear_gate_delta(
            "bilinear_full_predicted_dynamics",
            "bilinear_no_dynamics_source_action",
            1402,
        )
        calibration_no_dynamics_consistency_delta = _bootstrap(
            np.asarray([
                float(bilinear_calibration_metrics[
                    "bilinear_no_dynamics_source_action"
                ][root]["pairwise_consistency"])
                - float(bilinear_calibration_metrics["action_only"][root][
                    "pairwise_consistency"
                ])
                for root in sorted(bilinear_calibration_metrics["action_only"])
            ], dtype=np.float64),
            np.random.default_rng(seed + 1403),
            draws,
        )
        bilinear_full_passed = bool(
            bilinear_full_vs_no_dynamics["ci95"][1] < 0.0
        )
        bilinear_no_dynamics_passed = bool(
            bilinear_no_dynamics_vs_action["ci95"][1] < 0.0
        )
        if bilinear_full_passed:
            bilinear_selected = "bilinear_full_predicted_dynamics"
        elif bilinear_no_dynamics_passed:
            bilinear_selected = "bilinear_no_dynamics_source_action"
        else:
            bilinear_selected = "action_only"
        predictions["calibration_safe_bilinear"] = bilinear_calibration_predictions[
            bilinear_selected
        ]
        nonlinear_calibration_selection["bilinear"] = {
            "rule": (
                "select full bilinear transfer only when it beats source-action "
                "bilinear on paired calibration regret; otherwise require the "
                "source-action model to beat action-only; otherwise fail closed"
            ),
            "full_vs_no_dynamics_regret_delta": bilinear_full_vs_no_dynamics,
            "no_dynamics_vs_action_regret_delta": bilinear_no_dynamics_vs_action,
            "selected": bilinear_selected,
            "full_dynamics_gate_passed": bilinear_full_passed,
            "no_dynamics_gate_passed": bilinear_no_dynamics_passed,
            "selection_split": "calibration",
            "evaluation_split": "outcome",
            "outcome_used_for_selection": False,
            "status": "exploratory_missing_source_action_interaction_repair",
        }
        crossfit_predictions = {
            "action_only": crossfit_action_pred,
            "bilinear_no_dynamics_source_action": crossfit_bilinear_no_dynamics_pred,
            "bilinear_full_predicted_dynamics": crossfit_bilinear_full_pred,
        }
        crossfit_regrets = {
            name: _macro_root_regrets(
                prediction, y, group_roots, base_roots, train_mask, margin
            )
            for name, prediction in crossfit_predictions.items()
        }
        crossfit_metrics = {
            name: _macro_root_metric_map(
                prediction, y, group_roots, base_roots, train_mask, margin
            )
            for name, prediction in crossfit_predictions.items()
        }

        def crossfit_gate_delta(left: str, right: str, offset: int) -> dict[str, Any]:
            common = sorted(set(crossfit_regrets[left]) & set(crossfit_regrets[right]))
            delta = np.asarray([
                crossfit_regrets[left][root] - crossfit_regrets[right][root]
                for root in common
            ], dtype=np.float64)
            return _bootstrap(delta, np.random.default_rng(seed + offset), draws)

        crossfit_no_dynamics_vs_action = crossfit_gate_delta(
            "bilinear_no_dynamics_source_action", "action_only", 1501
        )
        crossfit_full_vs_no_dynamics = crossfit_gate_delta(
            "bilinear_full_predicted_dynamics",
            "bilinear_no_dynamics_source_action",
            1502,
        )
        crossfit_no_dynamics_consistency_delta = _bootstrap(
            np.asarray([
                float(crossfit_metrics["bilinear_no_dynamics_source_action"][root][
                    "pairwise_consistency"
                ])
                - float(crossfit_metrics["action_only"][root]["pairwise_consistency"])
                for root in sorted(crossfit_metrics["action_only"])
            ], dtype=np.float64),
            np.random.default_rng(seed + 1503),
            draws,
        )
        crossfit_full_passed = bool(
            crossfit_full_vs_no_dynamics["ci95"][1] < 0.0
            and bilinear_full_vs_no_dynamics["mean"] <= 0.0
        )
        crossfit_no_dynamics_passed = bool(
            crossfit_no_dynamics_vs_action["ci95"][1] < 0.0
            and bilinear_no_dynamics_vs_action["mean"] <= 0.0
        )
        if crossfit_full_passed:
            crossfit_selected = "bilinear_full_predicted_dynamics"
        elif crossfit_no_dynamics_passed:
            crossfit_selected = "bilinear_no_dynamics_source_action"
        else:
            crossfit_selected = "action_only"
        predictions["train_crossfit_safe_bilinear"] = bilinear_calibration_predictions[
            crossfit_selected
        ]
        nonlinear_calibration_selection["bilinear_train_crossfit_gate"] = {
            "rule": (
                "for predicted dynamics require root-disjoint train OOF regret CI95 "
                "below zero versus no-dynamics plus nonpositive calibration regret; "
                "for decision-only repair require train OOF regret CI95 below zero "
                "plus nonpositive calibration regret; consistency is reported as a "
                "secondary gate diagnostic; never inspect outcome"
            ),
            "train_crossfit_full_vs_no_dynamics_regret_delta": (
                crossfit_full_vs_no_dynamics
            ),
            "train_crossfit_no_dynamics_vs_action_regret_delta": (
                crossfit_no_dynamics_vs_action
            ),
            "train_crossfit_no_dynamics_vs_action_consistency_delta": (
                crossfit_no_dynamics_consistency_delta
            ),
            "calibration_full_vs_no_dynamics_regret_delta": (
                bilinear_full_vs_no_dynamics
            ),
            "calibration_no_dynamics_vs_action_regret_delta": (
                bilinear_no_dynamics_vs_action
            ),
            "calibration_no_dynamics_vs_action_consistency_delta": (
                calibration_no_dynamics_consistency_delta
            ),
            "selected": crossfit_selected,
            "full_dynamics_gate_passed": crossfit_full_passed,
            "no_dynamics_gate_passed": crossfit_no_dynamics_passed,
            "fit_split": "train",
            "crossfit_folds": 5,
            "direction_check_split": "calibration",
            "evaluation_split": "outcome",
            "outcome_used_for_selection": False,
            "status": "exploratory_train_crossfit_gate_after_calibration_power_diagnosis",
        }
    metrics: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    baseline_cells: dict[str, dict[str, float | None]] | None = None
    outcome_cell_to_root = {
        group: base
        for group, base, is_outcome in zip(
            group_roots, base_roots, outcome_mask
        )
        if is_outcome
    }
    for name, prediction in predictions.items():
        cell_root = _root_metrics(prediction, y, group_roots, outcome_mask, margin)
        root = _macro_root_metrics(cell_root, outcome_cell_to_root)
        consistency = np.asarray([float(item["pairwise_consistency"]) for item in root.values()])
        regrets = np.asarray([float(item["regret"]) for item in root.values()])
        metrics[name] = {
            "outcome_root_count": len(root),
            "context_root_cell_count": len(cell_root),
            "pairwise_consistency": _bootstrap(consistency, rng, draws),
            "mean_regret": _bootstrap(regrets, rng, draws),
            "root_metrics": root,
            "context_root_metrics": cell_root,
        }
        if name == "action_only":
            baseline_cells = cell_root
    assert baseline_cells is not None
    for name, prediction in predictions.items():
        cell_root = _root_metrics(prediction, y, group_roots, outcome_mask, margin)
        root = _macro_root_metrics(cell_root, outcome_cell_to_root)
        baseline_root = _macro_root_metrics(baseline_cells, outcome_cell_to_root)
        consistency_delta = np.asarray([
            float(root[key]["pairwise_consistency"]) - float(baseline_root[key]["pairwise_consistency"])
            for key in sorted(root)
        ])
        regret_delta = np.asarray([
            float(root[key]["regret"]) - float(baseline_root[key]["regret"])
            for key in sorted(root)
        ])
        metrics[name]["vs_action_only"] = {
            "pairwise_consistency_delta": _bootstrap(consistency_delta, rng, draws),
            "regret_delta": _bootstrap(regret_delta, rng, draws),
        }
    output.mkdir(parents=True, exist_ok=True)
    readout_artifact = None
    if action_frame in ("raw", "proposal_relative"):
        # This export is deployable from RGB/model tensors alone.  The
        # goal-relative branch is intentionally not exported because its
        # action coordinates require evaluator-only source/goal state.
        action_fit, residual_fit = _fit_linear_residual_models(
            x,
            action_x,
            train_target,
            train_mask,
            selected_alpha_for_artifact,
        )
        proposal_support_mean = np.mean(action_x[train_mask], axis=0)
        proposal_support_scale = np.maximum(
            np.std(action_x[train_mask], axis=0), 1.0e-6,
        )
        proposal_support_distance = np.sum(
            ((action_x[train_mask] - proposal_support_mean) / proposal_support_scale) ** 2,
            axis=1,
        )
        proposal_support_radius_sq = float(
            np.quantile(proposal_support_distance, 0.99)
        )
        all_proposal_support_distance = np.sum(
            ((action_x - proposal_support_mean) / proposal_support_scale) ** 2,
            axis=1,
        )
        all_proposal_support = all_proposal_support_distance <= proposal_support_radius_sq
        calibration_support_lower_bounds = []
        for group in sorted({group_roots[i] for i in np.flatnonzero(calibration_mask)}):
            group_mask = calibration_mask & np.asarray(
                [candidate_group == group for candidate_group in group_roots],
                dtype=bool,
            )
            calibration_support_lower_bounds.append(
                _wilson_lower_bound(
                    int(np.sum(all_proposal_support[group_mask])),
                    int(np.sum(group_mask)),
                    confidence=0.90,
                )
            )
        if not calibration_support_lower_bounds:
            raise ValueError("calibration split contains no proposal-support cells")
        # A deployment row must have support comparable to calibration rows,
        # while accounting for finite proposal counts.  We use the fixed fifth
        # percentile of one-sided 90% Wilson lower bounds.  Outcome labels and
        # online rewards never select this threshold.
        proposal_support_min_fraction = float(
            np.quantile(np.asarray(calibration_support_lower_bounds), 0.05)
        )
        readout_path = output / "capacity_readout.npz"
        np.savez_compressed(
            readout_path,
            action_coefficient=np.asarray(action_fit["coefficients"], dtype=np.float32),
            action_mean=np.asarray(action_fit["mean"], dtype=np.float32),
            action_scale=np.asarray(action_fit["scale"], dtype=np.float32),
            residual_coefficient=np.asarray(residual_fit["coefficients"], dtype=np.float32),
            residual_mean=np.asarray(residual_fit["mean"], dtype=np.float32),
            residual_scale=np.asarray(residual_fit["scale"], dtype=np.float32),
            gate_enabled=np.asarray([int(gate_passed)], dtype=np.int8),
            bilinear_no_dynamics_coefficient=np.asarray(
                bilinear_no_dynamics_fit["coefficients"], dtype=np.float32
            ),
            bilinear_no_dynamics_mean=np.asarray(
                bilinear_no_dynamics_fit["mean"], dtype=np.float32
            ),
            bilinear_no_dynamics_scale=np.asarray(
                bilinear_no_dynamics_fit["scale"], dtype=np.float32
            ),
            bilinear_full_coefficient=np.asarray(
                bilinear_full_fit["coefficients"], dtype=np.float32
            ),
            bilinear_full_mean=np.asarray(bilinear_full_fit["mean"], dtype=np.float32),
            bilinear_full_scale=np.asarray(bilinear_full_fit["scale"], dtype=np.float32),
            bilinear_source_mean=np.asarray(source_pca_mean, dtype=np.float32),
            bilinear_source_components=np.asarray(source_pca_components, dtype=np.float32),
            bilinear_delta_mean=np.asarray(delta_pca_mean, dtype=np.float32),
            bilinear_delta_components=np.asarray(delta_pca_components, dtype=np.float32),
            bilinear_rank=np.asarray([source_pca_components.shape[0]], dtype=np.int64),
            bilinear_selected=np.asarray([crossfit_selected]),
            bilinear_gate_enabled=np.asarray(
                [int(crossfit_selected != "action_only")], dtype=np.int8
            ),
            selected_alpha=np.asarray([selected_alpha_for_artifact], dtype=np.float64),
            feature_family=np.asarray(["source_delta_action"]),
            action_frame=np.asarray([action_frame]),
            goal_conditioning=np.asarray([goal_conditioning]),
            target_mode=np.asarray([target_mode]),
            mlp_no_dynamics_gate_enabled=np.asarray(
                [int(nonlinear_selected == "mlp_no_dynamics_source_action")],
                dtype=np.int8,
            ),
            proposal_support_mean=np.asarray(proposal_support_mean, dtype=np.float32),
            proposal_support_scale=np.asarray(proposal_support_scale, dtype=np.float32),
            proposal_support_radius_sq=np.asarray(
                [proposal_support_radius_sq], dtype=np.float32,
            ),
            proposal_support_min_fraction=np.asarray(
                [proposal_support_min_fraction], dtype=np.float32,
            ),
            proposal_support_threshold_method=np.asarray([
                "calibration_cell_p05_of_one_sided_wilson90"
            ]),
            mlp_residual_start_iteration=np.asarray(
                [mlp_residual_start_iteration], dtype=np.int64,
            ),
            mlp_planner_call_index=np.asarray([mlp_planner_call_index], dtype=np.int64),
            **{
                f"mlp_no_dynamics_{key}": np.asarray(value, dtype=np.float32)
                for key, value in no_dynamics_mlp_artifact.items()
            },
        )
        readout_artifact = str(readout_path)
    report = {
        "schema": "gdd_wm_train_only_teacher_capacity_audit.v0.1",
        "status": "completed_cuda_diagnostic",
        "protocol": {
            "fit_split": "train",
            "evaluation_split": "outcome",
            "feature_family": "source_delta_action",
            "action_frame": action_frame,
            "goal_conditioning": goal_conditioning,
            "target_mode": target_mode,
            "multiple_contexts_required": bool(require_multiple_contexts),
            "mlp_residual_start_iteration": int(mlp_residual_start_iteration),
            "mlp_planner_call_index": int(mlp_planner_call_index),
            "bilinear_pca_rank": int(source_pca_components.shape[0]),
            "action_baseline": "standardized_ridge_train_only",
            "linear_residual": "standardized_ridge_train_only",
            "mlp_residual": {"hidden": [64, 32], "activation": "SiLU", "epochs": epochs, "learning_rate": learning_rate, "weight_decay": weight_decay},
            "listwise_mlp": {
                "hidden": [64, 32],
                "activation": "SiLU",
                "epochs": epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "temperature": 0.25,
                "mse_weight": 0.1,
                "grouping": "checkpoint_context_x_root",
            },
            "bilinear_transfer": {
                "ridge_alpha": 0.01,
                "no_dynamics_terms": ["action", "source_x_action"],
                "predicted_dynamics_terms": [
                    "predicted_delta",
                    "predicted_delta_x_action",
                    "source_x_predicted_delta",
                ],
            },
            "family_or_capacity_selection": "not_performed",
            "device": str(device),
            "seed": seed,
            "root_bootstrap_draws": draws,
            "pair_margin": margin,
        },
        "summary": {
            "record_count": len(rows),
            "train_root_count": len({base_roots[i] for i in np.flatnonzero(train_mask)}),
            "outcome_root_count": len({base_roots[i] for i in np.flatnonzero(outcome_mask)}),
            "train_context_root_cell_count": len({group_roots[i] for i in np.flatnonzero(train_mask)}),
            "outcome_context_root_cell_count": len({group_roots[i] for i in np.flatnonzero(outcome_mask)}),
            "context_count": len(contexts),
            "feature_dimension": x.shape[1],
        },
        "calibration_selection": calibration_selection,
        "nonlinear_calibration_selection": nonlinear_calibration_selection,
        "runtime_readout_artifact": readout_artifact,
        "metrics": metrics,
        "claim_boundary": "capacity diagnostic only; no D-JEPA teacher, D2, or planning claim",
    }
    (output / "capacity_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", action="append", required=True)
    parser.add_argument("--arrays", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument(
        "--action-frame",
        choices=("raw", "proposal_relative", "goal_relative"),
        default="raw",
    )
    parser.add_argument("--target-mode", choices=("absolute", "root_centered"), default="absolute")
    parser.add_argument("--goal-conditioning", choices=("none", "latent_relative"), default="none")
    parser.add_argument(
        "--allow-single-context", action="store_true",
        help="allow one root-disjoint panel when no cross-context intersection is required",
    )
    parser.add_argument("--mlp-residual-start-iteration", type=int, default=7)
    parser.add_argument("--mlp-planner-call-index", type=int, default=1)
    args = parser.parse_args()
    run(
        candidate_datasets=[Path(item) for item in args.candidate_dataset],
        arrays_paths=[Path(item) for item in args.arrays],
        output=args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        draws=args.draws,
        seed=args.seed,
        action_frame=args.action_frame,
        target_mode=args.target_mode,
        goal_conditioning=args.goal_conditioning,
        require_multiple_contexts=not args.allow_single_context,
        mlp_residual_start_iteration=args.mlp_residual_start_iteration,
        mlp_planner_call_index=args.mlp_planner_call_index,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
