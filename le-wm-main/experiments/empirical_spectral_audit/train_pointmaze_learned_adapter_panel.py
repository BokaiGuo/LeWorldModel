"""Train a real PointMaze learned-adapter panel for the matched-quality audit.

The released JEPA/DINO temporal exports are frozen encoder/world-model outputs.
They do not contain multiple learned seeds, so they cannot by themselves support
the conditional learned-model claim.  This runner trains small, action-
conditioned adapters on a fixed, episode-disjoint split of those exports.

The panel is deliberately modest and auditable:

* two predictor architectures (linear and MLP);
* five independent optimizer/model seeds;
* a fixed training/calibration/outcome split per frozen export;
* PCA fitted on training rows only;
* independent action-risk on calibration rows; and
* a fixed horizon candidate bank {1, 2, 4} whose actual task consequence is
  scored only on the outcome rows.

This is a learned-adapter benchmark, not a claim that the released encoder
itself has five seeds.  All output metadata keeps that boundary explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.enriched_record_schema import make_payload
from experiments.empirical_spectral_audit.pointmaze_action_charts import action_blocks_for_chart


HORIZONS = (1, 2, 4)
ARCHITECTURES = ("linear", "mlp")
SEEDS = (8411, 8412, 8413, 8414, 8415)
ACTION_DISTRIBUTION = "pointmaze_observed_shared_v1"
# Full-rank principal-angle diagnostics are trivial: every d-dimensional
# subspace in R^d overlaps perfectly.  The declared planning readout in this
# benchmark is two-dimensional positional progress, so its task-relevant
# geometry rank is fixed to two before outcome evaluation.
TASK_GEOMETRY_RANK = 2

# These are calibration-only safety thresholds, frozen before the outcome
# split is read.  The task RMSE threshold is normalized by the train-only
# h=1 task-effect scale, so it is comparable across latent charts.  The
# action-risk threshold is in the train-fitted PCA latent chart.  A failed
# permission gate does not manufacture a positive result: it rejects the
# learned candidate and returns the fixed h=1 baseline.
PLANNING_PERMISSION_VERSION = "calibration_transfer_permission_v1"
MIN_CALIBRATION_ROWS_FOR_PERMISSION = 4
MAX_H1_NORMALIZED_TASK_RMSE = 0.75
MAX_CANDIDATE_NORMALIZED_TASK_RMSE = 1.25
MAX_CALIBRATION_ACTION_INVERSE_RISK = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("record_schema") != "enriched_learned_model.v1":
        raise ValueError(f"{path} is not a canonical enriched export")
    if not isinstance(payload.get("environment"), str) or not payload["environment"]:
        raise ValueError(f"{path} lacks a declared environment")
    if len(payload.get("enriched_records", [])) < 24:
        raise ValueError(f"{path} has too few rows for a split learned panel")
    return payload


def _pca(rows: list[dict[str, Any]], rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = []
    for row in rows:
        values.append(np.asarray(row["source_latent"], dtype=np.float64))
        for horizon in HORIZONS:
            values.append(np.asarray(row["latent_by_horizon"][str(horizon)]["target"], dtype=np.float64))
    matrix = np.asarray(values)
    mean = matrix.mean(axis=0)
    _, _, vh = np.linalg.svd(matrix - mean, full_matrices=False)
    basis = vh[: min(rank, vh.shape[0], matrix.shape[0] - 1)]
    if basis.shape[0] < 8:
        raise ValueError("PCA rank is not identifiable on the training rows")
    return mean, basis, matrix


def _project(value: Any, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=np.float64) - mean) @ basis.T


def _causal_history_eligible_rows(rows: list[dict[str, Any]], history_windows: int) -> list[dict[str, Any]]:
    """Keep rows with enough same-episode past context for a causal window."""
    if history_windows <= 1:
        return list(rows)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode_id"]), []).append(row)
    for episode_rows in grouped.values():
        episode_rows.sort(key=lambda item: int(item.get("window_start", 0)))
    eligible = []
    for row in rows:
        current_start = int(row.get("window_start", 0))
        history_count = sum(
            int(item.get("window_start", 0)) <= current_start
            for item in grouped[int(row["episode_id"])]
        )
        if history_count >= history_windows:
            eligible.append(row)
    return eligible


def _features(
    rows: list[dict[str, Any]],
    mean: np.ndarray,
    basis: np.ndarray,
    action_feature_chart: str = "raw",
    history_windows: int = 1,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    if history_windows < 1:
        raise ValueError("history_windows must be positive")
    original_rows = list(rows)
    grouped: dict[int, list[dict[str, Any]]] = {}
    if history_windows > 1:
        for row in original_rows:
            grouped.setdefault(int(row["episode_id"]), []).append(row)
        for episode_rows in grouped.values():
            episode_rows.sort(key=lambda item: int(item.get("window_start", 0)))
        if not all(row.get("source_history_raw") is not None for row in original_rows):
            rows = _causal_history_eligible_rows(original_rows, history_windows)
    sources, contexts, actions, states = [], [], [], []
    target_states = {h: [] for h in HORIZONS}
    targets = {h: [] for h in HORIZONS}
    for row in rows:
        current_source = _project(row["source_latent"], mean, basis)
        if history_windows == 1:
            context = current_source
        elif row.get("source_history_raw") is not None:
            history_values = list(row["source_history_raw"])
            if len(history_values) < history_windows:
                raise ValueError(
                    f"row {row.get('sample_id', '<unknown>')} has incomplete stored history: "
                    f"need {history_windows}, found {len(history_values)}"
                )
            context = np.concatenate([
                _project(value, mean, basis)
                for value in history_values[-history_windows:]
            ])
        else:
            episode_rows = grouped[int(row["episode_id"])]
            current_start = int(row.get("window_start", 0))
            history = [item for item in episode_rows if int(item.get("window_start", 0)) <= current_start]
            if len(history) < history_windows:
                raise ValueError(
                    f"row {row.get('sample_id', '<unknown>')} lacks causal history: "
                    f"need {history_windows}, found {len(history)}"
                )
            context = np.concatenate([
                _project(item["source_latent"], mean, basis)
                for item in history[-history_windows:]
            ])
        # Keep the current source separately for task geometry and residual
        # readouts; x carries the full causal context when requested.
        sources.append(current_source)
        contexts.append(context)
        # The two frozen exports use different historical conventions for the
        # field named ``action_blocks_normalized`` (DINO is approximately
        # standardized while JEPA is already in [-1, 1]).  The raw blocks are
        # the shared physical action chart and are the only valid cross-model
        # input for this matched audit.
        if action_feature_chart in {"goal_relative", "hybrid"}:
            transformed = action_blocks_for_chart(
                [row], np.asarray([row["action_blocks_raw"]], dtype=np.float32), action_feature_chart
            )[0]
            actions.append(np.asarray(transformed, dtype=np.float64).reshape(-1))
        else:
            action_key = "action_blocks_raw" if action_feature_chart == "raw" else "action_blocks_normalized"
            actions.append(np.asarray(row[action_key], dtype=np.float64).reshape(-1))
        source_state = np.asarray(row["source_state_raw"], dtype=np.float64)
        states.append(source_state)
        for h in HORIZONS:
            targets[h].append(_project(row["latent_by_horizon"][str(h)]["target"], mean, basis))
            target_states[h].append(np.asarray(row["latent_by_horizon"][str(h)]["target_state_raw"], dtype=np.float32))
    source = np.asarray(sources, dtype=np.float32)
    action = np.asarray(actions, dtype=np.float32)
    context_array = np.asarray(contexts, dtype=np.float32)
    x = np.concatenate([context_array, action], axis=1)
    y = {h: np.asarray(targets[h], dtype=np.float32) for h in HORIZONS}
    return x, y, source, np.asarray(states, dtype=np.float32), {h: np.asarray(target_states[h], dtype=np.float32) for h in HORIZONS}


def _episode_disjoint_split(
    rows: list[dict[str, Any]], outcome_row_budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split complete episodes into train, calibration, and outcome rows.

    PointMaze happens to have one exported row per episode, which previously
    hid this issue.  PushT has multiple temporal windows per episode; slicing
    rows would leak the same episode across all three roles.  The split is
    therefore defined over sorted episode groups and uses the requested row
    count only as a lower-bound budget for the held-out episode groups.
    """
    if outcome_row_budget < 1:
        raise ValueError("outcome-count must be positive")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode_id"]), []).append(row)
    episode_ids = sorted(grouped)
    if len(episode_ids) < 6:
        raise ValueError("episode-disjoint split requires at least six episodes")
    outcome_ids: list[int] = []
    outcome_rows_count = 0
    for episode_id in reversed(episode_ids):
        outcome_ids.append(episode_id)
        outcome_rows_count += len(grouped[episode_id])
        if outcome_rows_count >= outcome_row_budget:
            break
    outcome_ids = sorted(outcome_ids)
    fit_ids = [episode_id for episode_id in episode_ids if episode_id not in set(outcome_ids)]
    calibration_ids: list[int] = []
    calibration_rows_count = 0
    calibration_budget = max(4, int(round(0.2 * sum(len(grouped[i]) for i in fit_ids))))
    for episode_id in reversed(fit_ids):
        calibration_ids.append(episode_id)
        calibration_rows_count += len(grouped[episode_id])
        if calibration_rows_count >= calibration_budget:
            break
    calibration_ids = sorted(calibration_ids)
    train_ids = [episode_id for episode_id in fit_ids if episode_id not in set(calibration_ids)]

    def collect(ids: list[int]) -> list[dict[str, Any]]:
        return [row for episode_id in ids for row in grouped[episode_id]]

    train_rows = collect(train_ids)
    calibration_rows = collect(calibration_ids)
    fit_rows = train_rows + calibration_rows
    outcome_rows = collect(outcome_ids)
    if len(train_rows) < 8 or len(calibration_rows) < 4 or len(outcome_rows) < 8:
        raise ValueError(
            f"episode-disjoint split has insufficient support: train={len(train_rows)}, "
            f"calibration={len(calibration_rows)}, outcome={len(outcome_rows)}"
        )
    return fit_rows, outcome_rows, train_rows, calibration_rows


class LinearAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96), nn.SiLU(),
            nn.Linear(96, 96), nn.SiLU(),
            nn.Linear(96, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLPAdapter(nn.Module):
    """Predict standardized finite-horizon latent residuals.

    The frozen exports have very different latent scales.  Directly fitting
    absolute target latents makes a small sample adapter spend capacity on
    reproducing representation scale and mean, while the planner only needs
    the action-conditioned effect.  This module is intentionally architecture
    compatible with ``MLPAdapter``; the residual/standardization contract is
    implemented in ``_fit``/``_predict`` using train-only statistics.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96), nn.SiLU(),
            nn.Linear(96, 96), nn.SiLU(),
            nn.Linear(96, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StandardizedMLPAdapter(nn.Module):
    """Predict absolute target latents with train-only normalization.

    This controlled counterpart to ``ResidualMLPAdapter`` keeps the same
    capacity but does not change the prediction target to a residual.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96), nn.SiLU(),
            nn.Linear(96, 96), nn.SiLU(),
            nn.Linear(96, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskAlignedMLPAdapter(StandardizedMLPAdapter):
    """Same predictor as ``standardized_mlp``; task loss is added in ``_fit``."""


def _fit(
    architecture: str,
    seed: int,
    x: np.ndarray,
    y: dict[int, np.ndarray],
    epochs: int,
    device: str,
    task_targets: dict[int, np.ndarray] | None = None,
    task_decoders: dict[int, np.ndarray] | None = None,
    task_centers: dict[int, np.ndarray] | None = None,
    task_target_centers: dict[int, np.ndarray] | None = None,
    task_direction: np.ndarray | None = None,
    task_loss_weight: float = 1.0,
    spectral_target: np.ndarray | None = None,
    spectral_loss_weight: float = 0.0,
    distill_target: np.ndarray | None = None,
    distillation_weight: float = 0.0,
    history_windows: int = 1,
    action_feature_chart: str = "raw",
) -> tuple[nn.Module, dict[str, Any]]:
    _seed(seed)
    task_effect_mode = architecture in {"task_effect_mlp", "task_effect_linear", "task_effect_lowrank_mlp"}
    lowrank_action_mode = architecture == "task_effect_lowrank_mlp"
    model_x = np.asarray(x, dtype=np.float32)
    action_basis = None
    action_mean = None
    source_dim = y[1].shape[1]
    context_dim = source_dim * int(history_windows)
    if lowrank_action_mode:
        action = np.asarray(x[:, context_dim:], dtype=np.float64)
        action_mean = action.mean(axis=0)
        _, _, vh = np.linalg.svd(action - action_mean, full_matrices=False)
        action_rank = min(4, vh.shape[0], max(vh.shape[0], 1))
        action_basis = vh[:action_rank]
        action_projected = (action - action_mean) @ action_basis.T
        model_x = np.concatenate([x[:, :context_dim], action_projected], axis=1).astype(np.float32)
    output_dim = len(HORIZONS) if task_effect_mode else y[1].shape[1] * len(HORIZONS)
    residual_mode = architecture == "residual_mlp"
    standardized_mode = architecture in {"residual_mlp", "standardized_mlp", "task_aligned_mlp", "task_effect_mlp", "task_effect_linear", "task_effect_lowrank_mlp"}
    model_cls = (
        LinearAdapter
        if architecture == "linear"
        else ResidualMLPAdapter
        if residual_mode
        else StandardizedMLPAdapter
        if architecture == "standardized_mlp"
        else TaskAlignedMLPAdapter
        if architecture == "task_aligned_mlp"
        else TaskAlignedMLPAdapter
        if architecture == "task_effect_mlp"
        else LinearAdapter
        if architecture == "task_effect_linear"
        else TaskAlignedMLPAdapter
        if architecture == "task_effect_lowrank_mlp"
        else MLPAdapter
    )
    model = model_cls(model_x.shape[1], output_dim).to(device)
    x_mean = np.zeros(model_x.shape[1], dtype=np.float32)
    x_scale = np.ones(model_x.shape[1], dtype=np.float32)
    y_mean = np.zeros(output_dim, dtype=np.float32)
    y_scale = np.ones(output_dim, dtype=np.float32)
    target = (
        np.column_stack([task_targets[h] for h in HORIZONS]).astype(np.float32)
        if task_effect_mode
        else np.concatenate([y[h] for h in HORIZONS], axis=1).astype(np.float32)
    )
    if standardized_mode:
        source = np.asarray(x[:, (history_windows - 1) * source_dim:history_windows * source_dim], dtype=np.float32)
        if residual_mode:
            target = target - np.concatenate([source for _ in HORIZONS], axis=1)
        x_mean = np.mean(model_x, axis=0).astype(np.float32)
        x_scale = np.maximum(np.std(model_x, axis=0), 1e-3).astype(np.float32)
        y_mean = np.mean(target, axis=0).astype(np.float32)
        y_scale = np.maximum(np.std(target, axis=0), 1e-3).astype(np.float32)
        x_fit = (model_x - x_mean) / x_scale
        y_fit = (target - y_mean) / y_scale
    else:
        x_fit = model_x
        y_fit = target
    x_tensor = torch.as_tensor(x_fit, device=device)
    y_tensor = torch.as_tensor(y_fit, device=device)
    source_tensor = torch.as_tensor(
        np.asarray(x[:, (history_windows - 1) * source_dim:history_windows * source_dim], dtype=np.float32),
        device=device,
    )
    source_centered = source_tensor - source_tensor.mean(dim=0, keepdim=True)
    spectral_target_tensor = None if spectral_target is None else torch.as_tensor(
        np.asarray(spectral_target, dtype=np.float32), device=device
    )
    distill_target_tensor = None if distill_target is None else torch.as_tensor(
        np.asarray(distill_target, dtype=np.float32), device=device
    )
    task_tensors = None
    task_decoder_tensors = None
    task_center_tensors = None
    task_target_center_tensors = None
    task_direction_tensor = None
    if architecture == "task_aligned_mlp":
        if not all(value is not None for value in (task_targets, task_decoders, task_centers, task_target_centers, task_direction)):
            raise ValueError("task_aligned_mlp requires train-only task readout inputs")
        task_tensors = {h: torch.as_tensor(task_targets[h], dtype=torch.float32, device=device) for h in HORIZONS}
        task_decoder_tensors = {h: torch.as_tensor(task_decoders[h], dtype=torch.float32, device=device) for h in HORIZONS}
        task_center_tensors = {h: torch.as_tensor(task_centers[h], dtype=torch.float32, device=device) for h in HORIZONS}
        task_target_center_tensors = {h: torch.as_tensor(task_target_centers[h], dtype=torch.float32, device=device) for h in HORIZONS}
        task_direction_tensor = torch.as_tensor(task_direction, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        output = model(x_tensor)
        loss = torch.mean((output - y_tensor) ** 2)
        if architecture == "task_aligned_mlp":
            # Decode only with train-fitted readouts.  The latent loss remains
            # present, so this is a controlled task-alignment intervention,
            # not a direct outcome-trained planner.
            absolute_output = output * torch.as_tensor(y_scale, dtype=torch.float32, device=device)
            absolute_output = absolute_output + torch.as_tensor(y_mean, dtype=torch.float32, device=device)
            task_losses = []
            for index, h in enumerate(HORIZONS):
                pred_h = absolute_output[:, index * y[h].shape[1] : (index + 1) * y[h].shape[1]]
                pred_task = (
                    ((pred_h - task_center_tensors[h]) @ task_decoder_tensors[h])
                    + task_target_center_tensors[h]
                ) @ task_direction_tensor
                task_losses.append(torch.mean((pred_task - task_tensors[h]) ** 2))
            loss = loss + float(task_loss_weight) * torch.mean(torch.stack(task_losses))
        if spectral_target_tensor is not None and float(spectral_loss_weight) > 0.0:
            # Match spectral invariants of the train-only reference predictor,
            # rather than its full matrix.  Tr(A^k), k=1..d, identifies the
            # eigenvalue multiset while leaving eigenvectors/non-normal
            # geometry free to vary.  This is the intended quality-matched
            # intervention: prediction quality and spectrum are constrained,
            # not post-hoc selected from the outcome split.
            if standardized_mode:
                absolute_output = output * torch.as_tensor(y_scale, dtype=torch.float32, device=device)
                absolute_output = absolute_output + torch.as_tensor(y_mean, dtype=torch.float32, device=device)
                if residual_mode:
                    absolute_output = absolute_output + torch.cat([source_tensor for _ in HORIZONS], dim=1)
            else:
                absolute_output = output
            predicted_h1 = absolute_output[:, : y[1].shape[1]]
            ridge = 1e-3 * torch.eye(source_centered.shape[1], device=device)
            predictor = torch.linalg.solve(
                source_centered.T @ source_centered / max(source_centered.shape[0], 1) + ridge,
                source_centered.T @ (predicted_h1 - predicted_h1.mean(dim=0, keepdim=True)) / max(source_centered.shape[0], 1),
            )
            invariants = []
            for power in range(1, source_centered.shape[1] + 1):
                invariants.append(torch.trace(torch.matrix_power(predictor, power)))
            invariant_tensor = torch.stack(invariants)
            scale = torch.clamp(torch.abs(spectral_target_tensor), min=1.0)
            spectral_loss = torch.mean(((invariant_tensor - spectral_target_tensor) / scale) ** 2)
            loss = loss + float(spectral_loss_weight) * spectral_loss
        if distill_target_tensor is not None and float(distillation_weight) > 0.0:
            if standardized_mode:
                distill_output = output * torch.as_tensor(y_scale, dtype=torch.float32, device=device)
                distill_output = distill_output + torch.as_tensor(y_mean, dtype=torch.float32, device=device)
                if residual_mode:
                    distill_output = distill_output + torch.cat([source_tensor for _ in HORIZONS], dim=1)
            else:
                distill_output = output
            distill_scale = torch.as_tensor(
                np.maximum(np.std(np.asarray(distill_target, dtype=np.float32), axis=0), 1e-3),
                dtype=torch.float32,
                device=device,
            )
            distill_loss = torch.mean(((distill_output - distill_target_tensor) / distill_scale) ** 2)
            loss = loss + float(distillation_weight) * distill_loss
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    training = {
        "seed": seed,
        "architecture": architecture,
        "epochs": epochs,
        "final_train_loss": history[-1],
        "mode": (
            "standardized_residual" if residual_mode
            else "task_effect" if task_effect_mode
            else "standardized_absolute" if standardized_mode
            else "absolute_target"
        ),
        "x_mean": x_mean.tolist(),
        "x_scale": x_scale.tolist(),
        "y_mean": y_mean.tolist(),
        "y_scale": y_scale.tolist(),
        "task_loss_weight": float(task_loss_weight) if architecture == "task_aligned_mlp" else 0.0,
        "spectral_match": spectral_target is not None,
        "spectral_loss_weight": float(spectral_loss_weight),
        "distillation": distill_target is not None,
        "distillation_weight": float(distillation_weight),
        "history_windows": int(history_windows),
        "context_dimension": int(context_dim),
        "action_feature_chart": str(action_feature_chart),
    }
    if lowrank_action_mode:
        training["feature_mode"] = "train_action_pca4"
        training["action_mean"] = np.asarray(action_mean, dtype=np.float32).tolist()
        training["action_basis"] = np.asarray(action_basis, dtype=np.float32).tolist()
        training["source_dimension"] = int(context_dim)
    if task_effect_mode:
        if not all(value is not None for value in (task_decoders, task_centers, task_target_centers, task_direction)):
            raise ValueError("task_effect_mlp requires train-only task readout inputs")
        bases = {}
        for h in HORIZONS:
            readout_direction = np.asarray(task_decoders[h], dtype=np.float64) @ np.asarray(task_direction, dtype=np.float64)
            denom = float(readout_direction @ readout_direction)
            if denom <= 1e-12:
                raise ValueError(f"task readout has no identifiable latent direction at horizon {h}")
            bases[str(h)] = (readout_direction / denom).tolist()
        training["task_bases"] = bases
        training["task_centers"] = {str(h): np.asarray(task_centers[h], dtype=np.float64).tolist() for h in HORIZONS}
        training["task_target_centers"] = {
            str(h): float(np.asarray(task_target_centers[h], dtype=np.float64) @ np.asarray(task_direction, dtype=np.float64))
            for h in HORIZONS
        }
    return model.eval(), training


def _predict(
    model: nn.Module,
    x: np.ndarray,
    dimension: int,
    device: str,
    training: dict[str, Any] | None = None,
) -> dict[int, np.ndarray]:
    training = training or {}
    mode = training.get("mode", "absolute_target")
    model_x = np.asarray(x, dtype=np.float32)
    if training.get("feature_mode") == "train_action_pca4":
        source_dim = int(training["source_dimension"])
        action = model_x[:, source_dim:]
        action_mean = np.asarray(training["action_mean"], dtype=np.float32)
        action_basis = np.asarray(training["action_basis"], dtype=np.float32)
        model_x = np.concatenate([model_x[:, :source_dim], (action - action_mean) @ action_basis.T], axis=1)
    if mode in {"standardized_residual", "standardized_absolute", "task_effect"}:
        x_mean = np.asarray(training["x_mean"], dtype=np.float32)
        x_scale = np.asarray(training["x_scale"], dtype=np.float32)
        model_x = (model_x - x_mean) / np.maximum(x_scale, 1e-3)
    with torch.inference_mode():
        output = model(torch.as_tensor(model_x, dtype=torch.float32, device=device)).cpu().numpy()
    if mode == "task_effect":
        y_mean = np.asarray(training["y_mean"], dtype=np.float32)
        y_scale = np.asarray(training["y_scale"], dtype=np.float32)
        task_output = output * y_scale + y_mean
        output_blocks = []
        for index, h in enumerate(HORIZONS):
            basis = np.asarray(training["task_bases"][str(h)], dtype=np.float32).reshape(1, -1)
            center = np.asarray(training["task_centers"][str(h)], dtype=np.float32).reshape(1, -1)
            target_center = float(training.get("task_target_centers", {}).get(str(h), 0.0))
            output_blocks.append(center + (task_output[:, index:index + 1] - target_center) * basis)
        output = np.concatenate(output_blocks, axis=1)
    elif mode in {"standardized_residual", "standardized_absolute"}:
        y_mean = np.asarray(training["y_mean"], dtype=np.float32)
        y_scale = np.asarray(training["y_scale"], dtype=np.float32)
        output = output * y_scale + y_mean
        if mode == "standardized_residual":
            history_windows = int(training.get("history_windows", 1))
            source = np.asarray(
                x[:, (history_windows - 1) * dimension:history_windows * dimension],
                dtype=np.float32,
            )
            output = output + np.concatenate([source for _ in HORIZONS], axis=1)
    return {h: output[:, index * dimension : (index + 1) * dimension] for index, h in enumerate(HORIZONS)}


def _task_direction(train_states: np.ndarray, train_target_states: np.ndarray | None = None) -> np.ndarray:
    # PointMaze exports expose position and velocity, but not the goal.  The
    # planner task is therefore defined by the dominant *within-transition*
    # training displacement direction.  Using np.diff(train_states) here would
    # compare unrelated episode roots in file order and make the task metric
    # depend on episode ordering/split composition.  The fallback is retained
    # only for legacy callers; all canonical runners pass h=1 target states.
    if train_target_states is not None:
        displacement = np.asarray(train_target_states)[:, :2] - np.asarray(train_states)[:, :2]
    else:
        displacement = np.diff(train_states[:, :2], axis=0)
    direction = displacement.mean(axis=0)
    if np.linalg.norm(direction) < 1e-6:
        _, _, vh = np.linalg.svd(displacement - displacement.mean(0), full_matrices=False)
        direction = vh[0]
    return direction / max(float(np.linalg.norm(direction)), 1e-8)


def _task_readout(train_targets: np.ndarray, train_states: np.ndarray, train_target_states: np.ndarray) -> np.ndarray:
    # Decode raw position displacement from the predicted latent on train rows.
    latent = train_targets - train_targets.mean(0, keepdims=True)
    target = train_target_states[:, :2] - train_states[:, :2]
    ridge = 1e-3 * max(float(np.trace(latent.T @ latent) / max(latent.shape[1], 1)), 1e-6)
    decoder = np.linalg.solve(latent.T @ latent / len(latent) + ridge * np.eye(latent.shape[1]), latent.T @ target / len(latent))
    return decoder


def _task_readout_with_center(
    train_targets: np.ndarray,
    train_states: np.ndarray,
    train_target_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an affine train-only task readout.

    The centering constant is part of the fitted readout.  Re-centering each
    held-out outcome batch would erase its absolute task displacement and make
    every candidate's predicted progress have mean zero.  The complete readout
    is ``delta_hat = (z - mu_z) @ W + mu_delta``.  Both centers are estimated
    on training rows only.
    """
    train_targets = np.asarray(train_targets, dtype=np.float64)
    train_states = np.asarray(train_states, dtype=np.float64)
    train_target_states = np.asarray(train_target_states, dtype=np.float64)
    latent_center = np.mean(train_targets, axis=0)
    target_center = np.mean(train_target_states[:, :2] - train_states[:, :2], axis=0)
    decoder = _task_readout(train_targets, train_states, train_target_states)
    return decoder, latent_center, target_center


def _decode_task_readout(
    latent: np.ndarray,
    decoder: np.ndarray,
    latent_center: np.ndarray,
    target_center: np.ndarray,
) -> np.ndarray:
    """Decode task displacement using a frozen affine readout."""
    latent = np.asarray(latent, dtype=np.float64)
    return (latent - np.asarray(latent_center, dtype=np.float64).reshape(1, -1)) @ np.asarray(
        decoder, dtype=np.float64
    ) + np.asarray(target_center, dtype=np.float64).reshape(1, -1)


def _predictor_spectral_invariants(source: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Return train-only traces of powers of the fitted h=1 predictor."""
    source = np.asarray(source, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    centered_source = source - source.mean(axis=0, keepdims=True)
    centered_predicted = predicted - predicted.mean(axis=0, keepdims=True)
    dimension = centered_source.shape[1]
    scale = max(float(np.trace(centered_source.T @ centered_source) / max(len(source), 1)), 1e-6)
    ridge = 1e-3 * scale * np.eye(dimension, dtype=np.float64)
    predictor = np.linalg.solve(
        centered_source.T @ centered_source / max(len(source), 1) + ridge,
        centered_source.T @ centered_predicted / max(len(source), 1),
    )
    return np.asarray([np.trace(np.linalg.matrix_power(predictor, power)) for power in range(1, dimension + 1)], dtype=np.float32)


def _fit_train_distillation_head(
    candidate: dict[int, np.ndarray],
    teacher: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a train-only linear head that transfers candidate outputs to teacher quality."""
    values = np.concatenate([candidate[h] for h in HORIZONS], axis=1).astype(np.float64)
    teacher = np.asarray(teacher, dtype=np.float64)
    candidate_mean = values.mean(axis=0)
    teacher_mean = teacher.mean(axis=0)
    centered = values - candidate_mean
    scale = max(float(np.trace(centered.T @ centered) / max(len(centered), 1)), 1e-6)
    # The head is a train-only quality projection.  Use only numerical
    # regularization here; a large ridge leaves a systematic teacher mismatch
    # that reintroduces the very spectrum confound this intervention is meant
    # to remove.
    ridge = 1e-8 * scale * np.eye(centered.shape[1], dtype=np.float64)
    head = np.linalg.solve(centered.T @ centered + ridge, centered.T @ (teacher - teacher_mean))
    return candidate_mean.astype(np.float32), head.astype(np.float32), teacher_mean.astype(np.float32)


def _apply_train_distillation_head(
    predictions: dict[int, np.ndarray],
    candidate_mean: np.ndarray,
    head: np.ndarray,
    teacher_mean: np.ndarray,
) -> dict[int, np.ndarray]:
    values = np.concatenate([predictions[h] for h in HORIZONS], axis=1).astype(np.float32)
    mapped = (values - candidate_mean.reshape(1, -1)) @ head + teacher_mean.reshape(1, -1)
    dimension = predictions[HORIZONS[0]].shape[1]
    return {h: mapped[:, index * dimension : (index + 1) * dimension] for index, h in enumerate(HORIZONS)}


def _enforce_train_predictor_spectrum(
    train_predictions: dict[int, np.ndarray],
    predictions: dict[int, np.ndarray],
    source_train: np.ndarray,
    source_rows: np.ndarray,
    teacher: np.ndarray,
) -> dict[int, np.ndarray]:
    """Apply a train-fitted dynamics correction with the teacher's h-wise operator.

    The correction is a declared train-only linear successor head.  It solves
    for the smallest source-conditioned correction whose ridge-identified
    predictor equals the teacher predictor under the same diagnostic ridge.
    It is then evaluated on calibration/outcome source rows without looking at
    their targets.  This removes the ill-conditioned residual amplification
    that made a plain output distillation head fail spectrum matching.
    """
    train_source = np.asarray(source_train, dtype=np.float64)
    train_centered = train_source - train_source.mean(axis=0, keepdims=True)
    rows = np.asarray(source_rows, dtype=np.float64) - train_source.mean(axis=0, keepdims=True)
    teacher_values = np.asarray(teacher, dtype=np.float64)
    dimension = train_centered.shape[1]
    gram = train_centered.T @ train_centered / max(len(train_centered), 1)
    scale = max(float(np.trace(gram)) / max(dimension, 1), 1e-12)
    ridge = 1e-3 * scale * np.eye(dimension, dtype=np.float64)
    ridge_operator = np.linalg.solve(gram + ridge, gram)
    corrected: dict[int, np.ndarray] = {}
    for index, horizon in enumerate(HORIZONS):
        start = index * dimension
        stop = (index + 1) * dimension
        train_candidate = np.asarray(train_predictions[horizon], dtype=np.float64)
        candidate = np.asarray(predictions[horizon], dtype=np.float64)
        candidate_mean = train_candidate.mean(axis=0, keepdims=True)
        candidate_centered = candidate - candidate_mean
        train_candidate_centered = train_candidate - candidate_mean
        teacher_block = teacher_values[:, start:stop]
        teacher_mean = teacher_block.mean(axis=0, keepdims=True)
        candidate_operator = np.linalg.solve(
            gram + ridge,
            train_centered.T @ train_candidate_centered / max(len(train_centered), 1),
        )
        teacher_operator = np.linalg.solve(
            gram + ridge,
            train_centered.T @ (teacher_block - teacher_mean) / max(len(train_centered), 1),
        )
        correction = np.linalg.solve(ridge_operator, teacher_operator - candidate_operator)
        corrected[horizon] = (candidate_centered + rows @ correction + teacher_mean).astype(np.float32)
    return corrected


def _planning_risk(
    model_predictions: dict[int, np.ndarray],
    outcome_targets: dict[int, np.ndarray],
    outcome_rows: list[dict[str, Any]],
    decoder: np.ndarray | dict[int, np.ndarray],
    direction: np.ndarray,
    latent_center: np.ndarray | dict[int, np.ndarray] | None = None,
    target_center: np.ndarray | dict[int, np.ndarray] | None = None,
) -> tuple[float, dict[str, Any]]:
    # A fixed candidate bank {1,2,4}; candidate prediction and actual outcome
    # share the same held-out source row.  The model chooses a horizon by
    # predicted displacement, and the outcome score uses only the recorded
    # physical state after that horizon.
    predicted_progress = []
    actual_progress = []
    for h in HORIZONS:
        pred_latent = model_predictions[h]
        # Use the loop's horizon.  The old code referenced ``horizon`` here,
        # although the loop variable is ``h``; that made the canonical panel
        # runner fail at evaluation time and left the repair-only path as the
        # only executable path.  Keeping this helper executable is important:
        # rerunning the frozen protocol must produce the same kind of record
        # as the readout-repair artifact.
        decoder_h = decoder[h] if isinstance(decoder, dict) else decoder
        center_value = None if latent_center is None else (latent_center[h] if isinstance(latent_center, dict) else latent_center)
        target_center_value = None if target_center is None else (target_center[h] if isinstance(target_center, dict) else target_center)
        center = pred_latent.mean(0, keepdims=True) if center_value is None else np.asarray(center_value, dtype=np.float64).reshape(1, -1)
        pred_delta = (pred_latent - center) @ decoder_h
        if target_center_value is not None:
            pred_delta = pred_delta + np.asarray(target_center_value, dtype=np.float64).reshape(1, -1)
        actual_delta = np.asarray([
            np.asarray(row["latent_by_horizon"][str(h)]["target_state_raw"], dtype=np.float64)[:2]
            - np.asarray(row["source_state_raw"], dtype=np.float64)[:2]
            for row in outcome_rows
        ])
        predicted_progress.append(pred_delta @ direction)
        actual_progress.append(actual_delta @ direction)
    predicted = np.stack(predicted_progress, axis=1)
    actual = np.stack(actual_progress, axis=1)
    selected = np.argmax(predicted, axis=1)
    chosen = actual[np.arange(len(actual)), selected]
    oracle = np.max(actual, axis=1)
    risk = float(np.mean(oracle - chosen))
    return risk, {
        "candidate_horizons": list(HORIZONS),
        "selected_horizon_counts": {str(h): int(np.sum(selected == i)) for i, h in enumerate(HORIZONS)},
        "mean_chosen_progress": float(np.mean(chosen)),
        "mean_oracle_progress": float(np.mean(oracle)),
        "direction": direction.tolist(),
    }


def _planning_risk_with_calibration_gate(
    model_predictions: dict[int, np.ndarray],
    calibration_predictions: dict[int, np.ndarray],
    outcome_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    decoder: dict[int, np.ndarray],
    direction: np.ndarray,
    latent_center: dict[int, np.ndarray],
    target_center: dict[int, np.ndarray] | None = None,
    beta: float = 1.0,
    calibration_action_risk: float | None = None,
    train_task_scale: float | dict[int, float] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Use calibration-only error to fail closed on unreliable horizons.

    The gate is frozen before outcome rows are touched.  A candidate's score
    is its predicted task progress minus ``beta`` times its calibration task
    RMSE.  This is intentionally conservative: it can remove a speculative
    horizon, but it cannot use held-out planning outcomes to tune a threshold.
    The fixed h=1 candidate remains available as the fallback.
    """

    def progress(predictions: dict[int, np.ndarray], rows: list[dict[str, Any]]) -> np.ndarray:
        values = []
        for h in HORIZONS:
            predicted = np.asarray(predictions[h], dtype=np.float64)
            center = np.asarray(latent_center[h], dtype=np.float64).reshape(1, -1)
            decoded = (predicted - center) @ decoder[h] @ direction
            if target_center is not None:
                decoded = decoded + float(np.asarray(target_center[h], dtype=np.float64) @ direction)
            values.append(decoded)
        return np.stack(values, axis=1)

    def actual(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.stack([
            np.asarray([
                (np.asarray(row["latent_by_horizon"][str(h)]["target_state_raw"], dtype=np.float64)[:2]
                 - np.asarray(row["source_state_raw"], dtype=np.float64)[:2]) @ direction
                for row in rows
            ])
            for h in HORIZONS
        ], axis=1)

    calibration_predicted = progress(calibration_predictions, calibration_rows)
    calibration_actual = actual(calibration_rows)
    calibration_rmse = np.sqrt(np.mean((calibration_actual - calibration_predicted) ** 2, axis=0))
    if isinstance(train_task_scale, dict):
        task_scales = np.asarray([train_task_scale[h] for h in HORIZONS], dtype=np.float64)
    else:
        task_scales = np.full(len(HORIZONS), float(train_task_scale or 0.0), dtype=np.float64)
    task_scales = np.maximum(task_scales, 1e-6)
    normalized_rmse = calibration_rmse / task_scales
    permission_reasons: list[str] = []
    if len(calibration_rows) < MIN_CALIBRATION_ROWS_FOR_PERMISSION:
        permission_reasons.append("insufficient_calibration_rows")
    if not np.all(np.isfinite(normalized_rmse)):
        permission_reasons.append("nonfinite_calibration_task_rmse")
    if normalized_rmse[0] > MAX_H1_NORMALIZED_TASK_RMSE:
        permission_reasons.append("h1_task_effect_not_identifiable")
    if calibration_action_risk is None or not np.isfinite(calibration_action_risk):
        permission_reasons.append("missing_calibration_action_risk")
    elif float(calibration_action_risk) > MAX_CALIBRATION_ACTION_INVERSE_RISK:
        permission_reasons.append("action_effect_not_identifiable")
    permission = not permission_reasons
    allowed = [0]
    if permission:
        allowed.extend(
            index for index in range(1, len(HORIZONS))
            if normalized_rmse[index] <= MAX_CANDIDATE_NORMALIZED_TASK_RMSE
        )
    predicted = progress(model_predictions, outcome_rows)
    observed = actual(outcome_rows)
    score = predicted - float(beta) * calibration_rmse.reshape(1, -1)
    if not permission:
        score[:, 1:] = -np.inf
    else:
        disallowed = [index for index in range(len(HORIZONS)) if index not in allowed]
        if disallowed:
            score[:, disallowed] = -np.inf
    selected = np.argmax(score, axis=1)
    oracle = np.max(observed, axis=1)
    chosen = observed[np.arange(len(observed)), selected]
    fixed_h1_loss = float(np.mean(oracle - observed[:, 0]))
    risk = float(np.mean(oracle - chosen))
    return risk, {
        "gate": "calibration_task_rmse_penalty",
        "permission_version": PLANNING_PERMISSION_VERSION,
        "planning_permission": "accepted" if permission else "rejected_fail_closed",
        "permission_reasons": permission_reasons,
        "candidate_horizons_allowed": [HORIZONS[index] for index in allowed] if permission else [HORIZONS[0]],
        "beta": float(beta),
        "calibration_task_rmse": {str(h): float(calibration_rmse[i]) for i, h in enumerate(HORIZONS)},
        "normalized_calibration_task_rmse": {str(h): float(normalized_rmse[i]) for i, h in enumerate(HORIZONS)},
        "train_task_effect_scale": {str(h): float(task_scales[i]) for i, h in enumerate(HORIZONS)},
        "calibration_action_inverse_risk": None if calibration_action_risk is None else float(calibration_action_risk),
        "permission_thresholds": {
            "min_calibration_rows": MIN_CALIBRATION_ROWS_FOR_PERMISSION,
            "max_h1_normalized_task_rmse": MAX_H1_NORMALIZED_TASK_RMSE,
            "max_candidate_normalized_task_rmse": MAX_CANDIDATE_NORMALIZED_TASK_RMSE,
            "max_calibration_action_inverse_risk": MAX_CALIBRATION_ACTION_INVERSE_RISK,
        },
        "selected_horizon_counts": {str(h): int(np.sum(selected == i)) for i, h in enumerate(HORIZONS)},
        "fixed_h1_loss": fixed_h1_loss,
        "mean_chosen_progress": float(np.mean(chosen)),
        "mean_oracle_progress": float(np.mean(oracle)),
        "direction": direction.tolist(),
    }


def _write_record(path: Path, record: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoint_record.json").write_text(json.dumps({"record_schema": "real_checkpoint_selection.v1", "record": record, "protocol": record.get("protocol", {})}, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-export", type=Path)
    parser.add_argument("--dino-export", type=Path)
    parser.add_argument(
        "--single-export",
        type=Path,
        help="Run a multi-adapter panel over one frozen export. This is the controlled same-backbone intervention; it cannot be combined with --jepa-export/--dino-export.",
    )
    parser.add_argument(
        "--single-base-name",
        default="frozen",
        help="Stable label for --single-export in record provenance and architecture names.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--stages", type=int, nargs="+", default=[5, 10, 25, 50, 75])
    parser.add_argument(
        "--action-feature-chart",
        choices=("raw", "export", "goal_relative", "hybrid"),
        default="raw",
        help="Use shared raw physical coordinates, a goal-relative physical chart, or each export's native normalized chart.",
    )
    parser.add_argument(
        "--history-windows",
        type=int,
        default=1,
        help="Number of causal source-latent windows concatenated before the action blocks. 1 preserves the single-frame protocol.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("linear", "mlp", "residual_mlp", "standardized_mlp", "task_aligned_mlp", "task_effect_mlp", "task_effect_linear", "task_effect_lowrank_mlp"),
        default=list(ARCHITECTURES),
        help="Adapter families. standardized_mlp uses train-only input/absolute-target normalization; residual_mlp additionally changes the target to a residual; task_aligned_mlp adds a train-only task-readout loss; task_effect_* predicts only the train-fitted task effect.",
    )
    parser.add_argument(
        "--task-loss-weight",
        type=float,
        default=1.0,
        help="Fixed train-only auxiliary task-readout loss weight for task_aligned_mlp.",
    )
    parser.add_argument(
        "--match-spectrum-to-mlp",
        action="store_true",
        help="For non-MLP adapters, add a train-only spectral-invariant penalty targeting the same-seed/stage ordinary MLP.",
    )
    parser.add_argument(
        "--spectral-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the train-only spectral-invariant matching penalty.",
    )
    parser.add_argument(
        "--distill-to-mlp",
        action="store_true",
        help="For non-MLP adapters, add a train-only prediction distillation penalty to the same-seed/stage ordinary MLP.",
    )
    parser.add_argument(
        "--distillation-weight",
        type=float,
        default=10.0,
        help="Weight of the train-only teacher distillation penalty.",
    )
    parser.add_argument(
        "--train-distill-to-mlp",
        action="store_true",
        help="Also apply the optimization-time distillation loss; the default --distill-to-mlp uses only a fitted train-only output head.",
    )
    parser.add_argument(
        "--common-quality-anchor",
        action="store_true",
        help="Use the predeclared seed 8411 MLP as the train-only quality anchor for every seed at each stage.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional deterministic prefix cap applied identically to every frozen export for a support-matched audit.",
    )
    parser.add_argument(
        "--allow-unmatched-support",
        action="store_true",
        help="Permit exploratory panels with unequal export sizes; disabled by default because outcome support then confounds architecture comparisons.",
    )
    parser.add_argument(
        "--outcome-count",
        type=int,
        default=None,
        help="Optional frozen count of final rows reserved for held-out outcome scoring; omitted keeps the legacy min(32,max(8,n//3)) rule.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help="Independent adapter seeds; extra seeds increase matched-stratum power without changing the split or selector protocol.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.history_windows < 1:
        parser.error("--history-windows must be positive")
    panel_seeds = tuple(int(seed) for seed in args.seeds)
    if not panel_seeds or len(set(panel_seeds)) != len(panel_seeds):
        parser.error("--seeds must contain at least one unique seed")
    args.output.mkdir(parents=True, exist_ok=True)
    all_records = []
    inventories = []
    if args.single_export is not None:
        if args.jepa_export is not None or args.dino_export is not None:
            parser.error("--single-export cannot be combined with --jepa-export/--dino-export")
        source_items = ((str(args.single_base_name), args.single_export),)
    else:
        if args.jepa_export is None or args.dino_export is None:
            parser.error("provide both --jepa-export and --dino-export, or use --single-export")
        source_items = (("jepa", args.jepa_export), ("dino", args.dino_export))
    payloads = {base_name: _load(source_path) for base_name, source_path in source_items}
    source_lengths = {base_name: len(payload["enriched_records"]) for base_name, payload in payloads.items()}
    if args.max_rows is None and len(set(source_lengths.values())) > 1 and not args.allow_unmatched_support:
        raise ValueError(
            "frozen exports have unequal row support "
            f"{source_lengths}; pass --max-rows for a support-matched audit "
            "or --allow-unmatched-support for an explicitly exploratory run"
        )
    for base_name, source_path in source_items:
        payload = payloads[base_name]
        environment = str(payload["environment"])
        action_distribution = (
            f"{environment.lower()}_goal_relative_shared_v1"
            if args.action_feature_chart == "goal_relative"
            else f"{environment.lower()}_observed_shared_v1"
        )
        rows = sorted(payload["enriched_records"], key=lambda row: (int(row["episode_id"]), int(row.get("window_start", 0))))
        if args.max_rows is not None:
            if args.max_rows < 24:
                raise ValueError("max-rows must leave at least 24 rows for the declared split")
            rows = rows[: args.max_rows]
        if args.history_windows > 1:
            all_rows = list(rows)
            grouped_all: dict[int, list[dict[str, Any]]] = {}
            for row in all_rows:
                grouped_all.setdefault(int(row["episode_id"]), []).append(row)
            for episode_rows in grouped_all.values():
                episode_rows.sort(key=lambda item: int(item.get("window_start", 0)))
            eligible_rows = _causal_history_eligible_rows(all_rows, args.history_windows)
            rows = []
            for row in eligible_rows:
                current_start = int(row.get("window_start", 0))
                history = [
                    item for item in grouped_all[int(row["episode_id"])]
                    if int(item.get("window_start", 0)) <= current_start
                ][-args.history_windows:]
                item = dict(row)
                item["source_history_raw"] = [source_row["source_latent"] for source_row in history]
                rows.append(item)
        else:
            rows = _causal_history_eligible_rows(rows, args.history_windows)
        if len(rows) < 24:
            raise ValueError(
                f"history_windows={args.history_windows} leaves too few eligible rows: {len(rows)}"
            )
        outcome_count = (
            int(args.outcome_count)
            if args.outcome_count is not None
            else min(32, max(8, len(rows) // 3))
        )
        fit_rows, outcome_rows, train_rows, calibration_rows = _episode_disjoint_split(rows, outcome_count)
        pca_mean, pca_basis, _ = _pca(train_rows, args.rank)
        x_train, y_train, source_train, state_train, target_state_train = _features(train_rows, pca_mean, pca_basis, args.action_feature_chart, args.history_windows)
        x_fit, y_fit, source_fit, state_fit, target_state_fit = _features(fit_rows, pca_mean, pca_basis, args.action_feature_chart, args.history_windows)
        x_cal, y_cal, source_cal, state_cal, target_state_cal = _features(calibration_rows, pca_mean, pca_basis, args.action_feature_chart, args.history_windows)
        x_outcome, y_outcome, source_outcome, state_outcome, target_state_outcome = _features(outcome_rows, pca_mean, pca_basis, args.action_feature_chart, args.history_windows)
        direction = _task_direction(state_train, target_state_train[1])
        decoder = {}
        latent_center = {}
        target_center = {}
        for horizon in HORIZONS:
            decoder[horizon], latent_center[horizon], target_center[horizon] = _task_readout_with_center(
                y_train[horizon], state_train, target_state_train[horizon]
            )
        task_targets = {
            horizon: np.asarray([
                (target_state_train[horizon][i, :2] - state_train[i, :2]) @ direction
                for i in range(len(train_rows))
            ], dtype=np.float32)
            for horizon in HORIZONS
        }
        inventories.append({"base": base_name, "source": str(source_path), "rows": len(rows), "fit": len(fit_rows), "calibration": len(calibration_rows), "outcome": len(outcome_rows), "rank": int(pca_basis.shape[0])})
        reference_spectral_targets: dict[tuple[int, int], np.ndarray] = {}
        reference_distill_targets: dict[tuple[int, int], np.ndarray] = {}
        if (args.match_spectrum_to_mlp or args.distill_to_mlp) and "mlp" in args.architectures:
            # Reference quality is constructed only from train rows and is
            # deterministic for each seed/stage.  Calibration and outcome
            # rows are never read while creating this target.
            for seed in panel_seeds:
                for stage in sorted(set(args.stages)):
                    reference_model, reference_training = _fit(
                        "mlp", seed, x_train, y_train, stage, args.device,
                        task_targets=task_targets,
                        task_decoders=decoder,
                        task_centers=latent_center,
                        task_target_centers=target_center,
                        task_direction=direction,
                        task_loss_weight=args.task_loss_weight,
                        history_windows=args.history_windows,
                        action_feature_chart=args.action_feature_chart,
                    )
                    reference_prediction = _predict(
                        reference_model, x_train, pca_basis.shape[0], args.device, reference_training
                    )
                    reference_spectral_targets[(seed, stage)] = _predictor_spectral_invariants(
                        source_train, reference_prediction[1]
                    )
                    reference_distill_targets[(seed, stage)] = np.concatenate(
                        [reference_prediction[h] for h in HORIZONS], axis=1
                    ).astype(np.float32)
        for architecture in args.architectures:
            for seed in panel_seeds:
                for stage in sorted(set(args.stages)):
                    if stage < 1 or stage > args.epochs:
                        raise ValueError(f"stage {stage} must be in [1, epochs]")
                    model, training = _fit(
                        architecture, seed, x_train, y_train, stage, args.device,
                        task_targets=task_targets,
                        task_decoders=decoder,
                        task_centers=latent_center,
                        task_target_centers=target_center,
                        task_direction=direction,
                        task_loss_weight=args.task_loss_weight,
                        history_windows=args.history_windows,
                        action_feature_chart=args.action_feature_chart,
                        spectral_target=(
                            reference_spectral_targets.get((seed, stage))
                            if args.match_spectrum_to_mlp and architecture != "mlp"
                            else None
                        ),
                        spectral_loss_weight=(
                            args.spectral_loss_weight
                            if args.match_spectrum_to_mlp and architecture != "mlp"
                            else 0.0
                        ),
                        distill_target=(
                            reference_distill_targets.get((seed, stage))
                            if args.train_distill_to_mlp and architecture != "mlp"
                            else None
                        ),
                        distillation_weight=(
                            args.distillation_weight
                            if args.train_distill_to_mlp and architecture != "mlp"
                            else 0.0
                        ),
                    )
                    # Keep the three roles disjoint in the exported evidence:
                    # train rows define prediction quality/geometry, calibration
                    # rows define external action risk, and outcome rows define
                    # the held-out planning consequence.  The previous runner
                    # called train+calibration ``fit`` and used that mixture for
                    # prediction_mse and the transfer operator, which made the
                    # conditional matching estimand ambiguous.
                    pred_train = _predict(model, x_train, pca_basis.shape[0], args.device, training)
                    pred_cal = _predict(model, x_cal, pca_basis.shape[0], args.device, training)
                    pred_outcome = _predict(model, x_outcome, pca_basis.shape[0], args.device, training)
                    distillation_head = None
                    apply_quality_head = args.distill_to_mlp and (
                        architecture != "mlp" or (args.common_quality_anchor and seed != panel_seeds[0])
                    )
                    if apply_quality_head:
                        teacher_key = (panel_seeds[0], stage) if args.common_quality_anchor else (seed, stage)
                        teacher = reference_distill_targets[teacher_key]
                        head_mean, head_matrix, head_target_mean = _fit_train_distillation_head(pred_train, teacher)
                        pred_train = _apply_train_distillation_head(pred_train, head_mean, head_matrix, head_target_mean)
                        pred_cal = _apply_train_distillation_head(pred_cal, head_mean, head_matrix, head_target_mean)
                        pred_outcome = _apply_train_distillation_head(pred_outcome, head_mean, head_matrix, head_target_mean)
                        pred_train = _enforce_train_predictor_spectrum(
                            pred_train, pred_train, source_train, source_train, teacher
                        )
                        pred_cal = _enforce_train_predictor_spectrum(
                            pred_train, pred_cal, source_train, source_cal, teacher
                        )
                        pred_outcome = _enforce_train_predictor_spectrum(
                            pred_train, pred_outcome, source_train, source_outcome, teacher
                        )
                        distillation_head = {
                            "candidate_mean": head_mean.tolist(),
                            "matrix": head_matrix.tolist(),
                            "teacher_mean": head_target_mean.tolist(),
                            "fit_split": "train_only",
                            "teacher_source": "same_seed_stage ordinary MLP train prediction",
                        }
                        training["posthoc_train_distillation_head"] = True
                        training["posthoc_distillation_head_shape"] = [int(head_matrix.shape[0]), int(head_matrix.shape[1])]
                    calibration_action_risk = float(np.mean((pred_cal[1] - y_cal[1]) ** 2))
                    planning_risk, planning_details = _planning_risk(
                        pred_outcome,
                        y_outcome,
                        outcome_rows,
                        decoder,
                        direction,
                        latent_center,
                        target_center,
                    )
                    planning_risk_conservative, planning_details_conservative = _planning_risk_with_calibration_gate(
                        pred_outcome,
                        pred_cal,
                        outcome_rows,
                        calibration_rows,
                        decoder,
                        direction,
                        latent_center,
                        target_center,
                        calibration_action_risk=calibration_action_risk,
                        train_task_scale={h: float(np.std(task_targets[h])) for h in HORIZONS},
                    )
                    checkpoint = args.output / "checkpoints" / f"{base_name}_{architecture}" / f"seed_{seed}" / f"stage_{stage:03d}.pt"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"state_dict": model.state_dict(), "architecture": architecture, "seed": seed, "base": base_name, "stage": stage, "rank": int(pca_basis.shape[0]), "pca_mean": pca_mean, "pca_basis": pca_basis, "adapter_training": training, "posthoc_train_distillation_head": distillation_head, "protocol": {"train_episode_ids": [int(r["episode_id"]) for r in train_rows], "calibration_episode_ids": [int(r["episode_id"]) for r in calibration_rows], "outcome_episode_ids": [int(r["episode_id"]) for r in outcome_rows]}}, checkpoint)
                    def export(rows_subset, pred_subset, split_name):
                        enriched = []
                        for row_index, row in enumerate(rows_subset):
                            item = dict(row)
                            item["source_latent"] = _project(row["source_latent"], pca_mean, pca_basis).tolist()
                            item["action_blocks_normalized"] = np.asarray(row["action_blocks_normalized"], dtype=np.float64).tolist()
                            item["action_blocks_raw"] = np.asarray(row["action_blocks_raw"], dtype=np.float64).tolist()
                            item["latent_by_horizon"] = {str(h): {**row["latent_by_horizon"][str(h)], "predicted": pred_subset[h][row_index].tolist(), "target": _project(row["latent_by_horizon"][str(h)]["target"], pca_mean, pca_basis).tolist()} for h in HORIZONS}
                            enriched.append(item)
                        return make_payload(architecture=f"{environment.lower()}_{base_name}_{architecture}", checkpoint=str(checkpoint.resolve()), environment=environment, seed=str(seed), training_stage=stage, latent_dimension=int(pca_basis.shape[0]), action_distribution=action_distribution, split={"name": split_name, "episode_ids": [int(r["episode_id"]) for r in rows_subset], "provenance": "fixed frozen-export episode split; adapter trained on train subset only"}, horizons=HORIZONS, enriched_records=enriched, checkpoint_sha256=_sha256(checkpoint), dataset_sha256=_sha256(source_path), protocol={"base_frozen_export": str(source_path.resolve()), "adapter_architecture": architecture, "adapter_seed": int(seed), "split_role": split_name, "train_episode_disjoint": True, "pca_fit_split": "train_only", "action_distribution": action_distribution}, study_scope=f"real learned {environment} action-conditioned adapter panel", claim_boundary="This is a learned adapter over a frozen released temporal export; it is not evidence that the released frozen checkpoint has multiple training seeds.", runtime="learned_adapter_panel.v1")
                    candidate_root = args.output / "exports" / f"{base_name}_{architecture}_seed{seed}_stage{stage:03d}"
                    candidate_root.mkdir(parents=True, exist_ok=True)
                    fit_payload = export(train_rows, pred_train, "train")
                    outcome_payload = export(outcome_rows, pred_outcome, "outcome")
                    fit_path = candidate_root / "fit.json"; out_path = candidate_root / "outcome.json"
                    fit_path.write_text(json.dumps(fit_payload, indent=2) + "\n"); out_path.write_text(json.dumps(outcome_payload, indent=2) + "\n")
                    from experiments.empirical_spectral_audit.build_learned_model_selection_records import _fit_metric, _fit_transfer_operator, _ridge, _rows
                    from experiments.task_metric_diagnostics import metric_geometry
                    fit_source, fit_predicted, fit_target, fit_delta = _rows(fit_payload, 1)
                    centered = fit_source - fit_source.mean(0, keepdims=True)
                    covariance = centered.T @ centered / max(len(centered) - 1, 1)
                    predictor = _ridge(centered, fit_predicted - fit_predicted.mean(0, keepdims=True), 1e-3)
                    metric = _fit_metric(fit_source, fit_delta, 1e-3)
                    geometry = metric_geometry(
                        covariance,
                        predictor,
                        metric,
                        rank=min(TASK_GEOMETRY_RANK, fit_source.shape[1]),
                        horizons=range(1, 5),
                    )
                    # The transfer operator is a train-only diagnostic, so its
                    # action rows must come from the same train split as the
                    # source/prediction arrays above.
                    action_fit = x_train[:, fit_source.shape[1]:].astype(np.float64)
                    transfer_operator = _fit_transfer_operator(
                        fit_source, fit_predicted, action_fit, fit_delta, 1e-3, metric, HORIZONS,
                    )
                    A = np.asarray(transfer_operator["A"], dtype=np.float64)
                    B = np.asarray(transfer_operator["B"], dtype=np.float64)
                    gramian = np.zeros((A.shape[0], A.shape[0]), dtype=np.float64)
                    power = np.eye(A.shape[0], dtype=np.float64)
                    for _ in HORIZONS:
                        response = power @ B
                        gramian += response @ response.T
                        power = A @ power
                    geometry["reachability_score"] = float(np.trace(gramian))
                    cal_predicted = np.concatenate([pred_cal[h] for h in HORIZONS], axis=1)
                    cal_target = np.concatenate([y_cal[h] for h in HORIZONS], axis=1)
                    geometry["real_predicted_transition_transfer"] = float(np.mean(np.square(cal_predicted - cal_target)))
                    spectrum = [{"real": float(v.real), "imag": float(v.imag)} for v in np.linalg.eigvals(predictor)]
                    planning_loss = float(np.mean([np.mean((pred_outcome[h][i] - y_outcome[h][i]) ** 2) for h in HORIZONS for i in range(len(outcome_rows))]))
                    record = {"group": f"{environment}/learned/{base_name}_{architecture}/seed_{seed}", "architecture": f"{environment.lower()}_{base_name}_{architecture}", "checkpoint": str(checkpoint.resolve()), "model_factors": {"latent_dimension": int(pca_basis.shape[0]), "training_stage": stage, "action_distribution": action_distribution, "action_inverse_risk": calibration_action_risk, "adapter_mode": training.get("mode", "absolute_target")}, "diagnostics": {"prediction_mse": float(np.mean((fit_predicted - fit_target) ** 2)), "predictor_spectrum": spectrum, **{k: float(v) for k, v in geometry.items()}, "action_inverse_risk": calibration_action_risk}, "heldout": {"planning_loss": planning_loss, "planning_risk": planning_risk, "planning_details": planning_details, "planning_risk_conservative": planning_risk_conservative, "planning_details_conservative": planning_details_conservative}, "protocol": {"environment": environment, "independent_external_evidence": False, "action_risk_semantics": "calibration transition MSE; not an independent +a/-a/zero action intervention", "fit_episode_ids": [int(r["episode_id"]) for r in train_rows], "calibration_episode_ids": [int(r["episode_id"]) for r in calibration_rows], "outcome_episode_ids": [int(r["episode_id"]) for r in outcome_rows], "action_feature_source": f"{args.action_feature_chart} action chart", "action_chart_role": "formal shared physical chart" if args.action_feature_chart == "raw" else "exploratory export-native chart", "action_risk_split": "calibration rows never used to fit adapter", "planning_split": "outcome rows never used for fitting, calibration, metric, or selector", "task_direction_source": "train-only h=1 source-to-target displacement", "task_readout_source": "train rows only, horizon-specific", "task_geometry_subspace_rank": int(min(TASK_GEOMETRY_RANK, fit_source.shape[1])), "train_task_effect_scale": {str(h): float(np.std(task_targets[h])) for h in HORIZONS}, "prediction_center_source": "train target latent mean only, horizon-specific", "candidate_action_bank": list(HORIZONS), "runtime_state_access": False, "physical_state_role": "offline consequence scoring only", "adapter_mode": training.get("mode", "absolute_target"), "conservative_gate": "calibration_task_rmse_penalty_beta1 + transfer_permission_v1; fallback candidate h=1 retained", "claim_boundary": f"held-out {environment} planning consequence from fixed horizon candidate bank; independent action intervention evidence is not present"}, "provenance": {"fit_export": str(fit_path), "outcome_export": str(out_path), "base_frozen_export": str(source_path.resolve()), "training": training}}
                    record["diagnostics"]["transfer_operator"] = transfer_operator
                    record["protocol"]["transition_transfer_split"] = "independent calibration rows; not used for adapter fitting or outcome selection"
                    record_dir = args.output / "records" / f"{base_name}_{architecture}_seed{seed}_stage{stage:03d}"
                    _write_record(record_dir, record)
                    all_records.append(record)
    manifest = {"schema": "learned_adapter_panel.v1", "protocol": {"architectures": list(args.architectures), "seeds": list(panel_seeds), "training_stages": sorted(set(args.stages)), "action_distribution": action_distribution, "environment": environment, "action_feature_chart": args.action_feature_chart, "rank": args.rank, "epochs": args.epochs, "max_rows": args.max_rows, "outcome_count": args.outcome_count, "support_matching": args.max_rows is not None, "allow_unmatched_support": args.allow_unmatched_support, "support_match_rule": "same deterministic prefix of every export" if args.max_rows is not None else ("equal export sizes" if len(set(source_lengths.values())) == 1 else "unequal exploratory support"), "pca_fit_only_on_train": True, "planning_candidate_bank": list(HORIZONS), "outcome_used_for_selection": False, "calibration_used_for_training": False, "task_loss_weight": args.task_loss_weight, "match_spectrum_to_mlp": args.match_spectrum_to_mlp, "spectral_loss_weight": args.spectral_loss_weight, "distill_to_mlp": args.distill_to_mlp, "distillation_weight": args.distillation_weight, "train_distill_to_mlp": args.train_distill_to_mlp, "common_quality_anchor": args.common_quality_anchor, "quality_anchor": f"ordinary MLP seed {panel_seeds[0]} per stage" if args.common_quality_anchor else "same seed/stage ordinary MLP", "adapter_modes": {"linear": "absolute_target", "mlp": "absolute_target", "residual_mlp": "standardized_residual", "standardized_mlp": "standardized_absolute", "task_aligned_mlp": "standardized_absolute_plus_train_task_loss", "task_effect_mlp": "train_task_effect_only", "task_effect_linear": "train_task_effect_only_linear", "task_effect_lowrank_mlp": "train_task_effect_only_with_train_action_pca4"}}, "inventories": inventories, "record_count": len(all_records), "claim_boundary": f"Real learned {environment} adapter panel with independent calibration action-risk and held-out planning consequence; not a multi-seed claim about the frozen encoder."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
