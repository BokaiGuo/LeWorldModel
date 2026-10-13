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


HORIZONS = (1, 2, 4)
ARCHITECTURES = ("linear", "mlp")
SEEDS = (8411, 8412, 8413, 8414, 8415)
ACTION_DISTRIBUTION = "pointmaze_observed_shared_v1"

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
    if payload.get("environment") != "PointMaze":
        raise ValueError(f"{path} is not a PointMaze export")
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


def _features(
    rows: list[dict[str, Any]],
    mean: np.ndarray,
    basis: np.ndarray,
    action_feature_chart: str = "raw",
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    sources, actions, states = [], [], []
    target_states = {h: [] for h in HORIZONS}
    targets = {h: [] for h in HORIZONS}
    for row in rows:
        sources.append(_project(row["source_latent"], mean, basis))
        # The two frozen exports use different historical conventions for the
        # field named ``action_blocks_normalized`` (DINO is approximately
        # standardized while JEPA is already in [-1, 1]).  The raw blocks are
        # the shared physical action chart and are the only valid cross-model
        # input for this matched audit.
        action_key = "action_blocks_raw" if action_feature_chart == "raw" else "action_blocks_normalized"
        actions.append(np.asarray(row[action_key], dtype=np.float64).reshape(-1))
        source_state = np.asarray(row["source_state_raw"], dtype=np.float64)
        states.append(source_state)
        for h in HORIZONS:
            targets[h].append(_project(row["latent_by_horizon"][str(h)]["target"], mean, basis))
            target_states[h].append(np.asarray(row["latent_by_horizon"][str(h)]["target_state_raw"], dtype=np.float32))
    source = np.asarray(sources, dtype=np.float32)
    action = np.asarray(actions, dtype=np.float32)
    x = np.concatenate([source, action], axis=1)
    y = {h: np.asarray(targets[h], dtype=np.float32) for h in HORIZONS}
    return x, y, source, np.asarray(states, dtype=np.float32), {h: np.asarray(target_states[h], dtype=np.float32) for h in HORIZONS}


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
    task_direction: np.ndarray | None = None,
    task_loss_weight: float = 1.0,
) -> tuple[nn.Module, dict[str, Any]]:
    _seed(seed)
    task_effect_mode = architecture in {"task_effect_mlp", "task_effect_linear", "task_effect_lowrank_mlp"}
    lowrank_action_mode = architecture == "task_effect_lowrank_mlp"
    model_x = np.asarray(x, dtype=np.float32)
    action_basis = None
    action_mean = None
    if lowrank_action_mode:
        source_dim = y[1].shape[1]
        action = np.asarray(x[:, source_dim:], dtype=np.float64)
        action_mean = action.mean(axis=0)
        _, _, vh = np.linalg.svd(action - action_mean, full_matrices=False)
        action_rank = min(4, vh.shape[0], max(vh.shape[0], 1))
        action_basis = vh[:action_rank]
        action_projected = (action - action_mean) @ action_basis.T
        model_x = np.concatenate([x[:, :source_dim], action_projected], axis=1).astype(np.float32)
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
        source = np.asarray(x[:, : y[1].shape[1]], dtype=np.float32)
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
    task_tensors = None
    task_decoder_tensors = None
    task_center_tensors = None
    task_direction_tensor = None
    if architecture == "task_aligned_mlp":
        if not all(value is not None for value in (task_targets, task_decoders, task_centers, task_direction)):
            raise ValueError("task_aligned_mlp requires train-only task readout inputs")
        task_tensors = {h: torch.as_tensor(task_targets[h], dtype=torch.float32, device=device) for h in HORIZONS}
        task_decoder_tensors = {h: torch.as_tensor(task_decoders[h], dtype=torch.float32, device=device) for h in HORIZONS}
        task_center_tensors = {h: torch.as_tensor(task_centers[h], dtype=torch.float32, device=device) for h in HORIZONS}
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
                pred_task = ((pred_h - task_center_tensors[h]) @ task_decoder_tensors[h]) @ task_direction_tensor
                task_losses.append(torch.mean((pred_task - task_tensors[h]) ** 2))
            loss = loss + float(task_loss_weight) * torch.mean(torch.stack(task_losses))
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
    }
    if lowrank_action_mode:
        training["feature_mode"] = "train_action_pca4"
        training["action_mean"] = np.asarray(action_mean, dtype=np.float32).tolist()
        training["action_basis"] = np.asarray(action_basis, dtype=np.float32).tolist()
        training["source_dimension"] = int(y[1].shape[1])
    if task_effect_mode:
        if not all(value is not None for value in (task_decoders, task_centers, task_direction)):
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
            output_blocks.append(center + task_output[:, index:index + 1] * basis)
        output = np.concatenate(output_blocks, axis=1)
    elif mode in {"standardized_residual", "standardized_absolute"}:
        y_mean = np.asarray(training["y_mean"], dtype=np.float32)
        y_scale = np.asarray(training["y_scale"], dtype=np.float32)
        output = output * y_scale + y_mean
        if mode == "standardized_residual":
            source = np.asarray(x[:, :dimension], dtype=np.float32)
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
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a readout and retain its train-only latent centering constant.

    The centering constant is part of the fitted readout.  Re-centering each
    held-out outcome batch would erase its absolute task displacement and make
    every candidate's predicted progress have mean zero, which is a subtle
    evaluation bug rather than a valid normalization.
    """
    return _task_readout(train_targets, train_states, train_target_states), np.mean(train_targets, axis=0)


def _planning_risk(
    model_predictions: dict[int, np.ndarray],
    outcome_targets: dict[int, np.ndarray],
    outcome_rows: list[dict[str, Any]],
    decoder: np.ndarray | dict[int, np.ndarray],
    direction: np.ndarray,
    latent_center: np.ndarray | dict[int, np.ndarray] | None = None,
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
        center = pred_latent.mean(0, keepdims=True) if center_value is None else np.asarray(center_value, dtype=np.float64).reshape(1, -1)
        pred_delta = (pred_latent - center) @ decoder_h
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
    parser.add_argument("--jepa-export", type=Path, required=True)
    parser.add_argument("--dino-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--stages", type=int, nargs="+", default=[5, 10, 25, 50, 75])
    parser.add_argument(
        "--action-feature-chart",
        choices=("raw", "export"),
        default="raw",
        help="Use shared raw physical action coordinates (formal default) or each export's native normalized chart (exploratory only).",
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_records = []
    inventories = []
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
        rows = sorted(payload["enriched_records"], key=lambda row: (int(row["episode_id"]), int(row.get("window_start", 0))))
        if args.max_rows is not None:
            if args.max_rows < 24:
                raise ValueError("max-rows must leave at least 24 rows for the declared split")
            rows = rows[: args.max_rows]
        outcome_count = (
            int(args.outcome_count)
            if args.outcome_count is not None
            else min(32, max(8, len(rows) // 3))
        )
        if outcome_count < 8 or outcome_count >= len(rows) - 8:
            raise ValueError("outcome-count must leave at least 8 rows for fitting")
        fit_rows = rows[:-outcome_count]
        outcome_rows = rows[-outcome_count:]
        train_count = max(8, int(round(0.8 * len(fit_rows))))
        train_rows, calibration_rows = fit_rows[:train_count], fit_rows[train_count:]
        pca_mean, pca_basis, _ = _pca(train_rows, args.rank)
        x_train, y_train, source_train, state_train, target_state_train = _features(train_rows, pca_mean, pca_basis, args.action_feature_chart)
        x_fit, y_fit, source_fit, state_fit, target_state_fit = _features(fit_rows, pca_mean, pca_basis, args.action_feature_chart)
        x_cal, y_cal, source_cal, state_cal, target_state_cal = _features(calibration_rows, pca_mean, pca_basis, args.action_feature_chart)
        x_outcome, y_outcome, source_outcome, state_outcome, target_state_outcome = _features(outcome_rows, pca_mean, pca_basis, args.action_feature_chart)
        direction = _task_direction(state_train, target_state_train[1])
        decoder = {}
        latent_center = {}
        for horizon in HORIZONS:
            decoder[horizon], latent_center[horizon] = _task_readout_with_center(
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
        for architecture in args.architectures:
            for seed in SEEDS:
                for stage in sorted(set(args.stages)):
                    if stage < 1 or stage > args.epochs:
                        raise ValueError(f"stage {stage} must be in [1, epochs]")
                    model, training = _fit(
                        architecture, seed, x_train, y_train, stage, args.device,
                        task_targets=task_targets,
                        task_decoders=decoder,
                        task_centers=latent_center,
                        task_direction=direction,
                        task_loss_weight=args.task_loss_weight,
                    )
                    pred_fit = _predict(model, x_fit, pca_basis.shape[0], args.device, training)
                    pred_cal = _predict(model, x_cal, pca_basis.shape[0], args.device, training)
                    pred_outcome = _predict(model, x_outcome, pca_basis.shape[0], args.device, training)
                    calibration_action_risk = float(np.mean((pred_cal[1] - y_cal[1]) ** 2))
                    planning_risk, planning_details = _planning_risk(pred_outcome, y_outcome, outcome_rows, decoder, direction, latent_center)
                    planning_risk_conservative, planning_details_conservative = _planning_risk_with_calibration_gate(
                        pred_outcome,
                        pred_cal,
                        outcome_rows,
                        calibration_rows,
                        decoder,
                        direction,
                        latent_center,
                        calibration_action_risk=calibration_action_risk,
                        train_task_scale={h: float(np.std(task_targets[h])) for h in HORIZONS},
                    )
                    checkpoint = args.output / "checkpoints" / f"{base_name}_{architecture}" / f"seed_{seed}" / f"stage_{stage:03d}.pt"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"state_dict": model.state_dict(), "architecture": architecture, "seed": seed, "base": base_name, "stage": stage, "rank": int(pca_basis.shape[0]), "pca_mean": pca_mean, "pca_basis": pca_basis, "adapter_training": training, "protocol": {"train_episode_ids": [int(r["episode_id"]) for r in train_rows], "calibration_episode_ids": [int(r["episode_id"]) for r in calibration_rows], "outcome_episode_ids": [int(r["episode_id"]) for r in outcome_rows]}}, checkpoint)
                    def export(rows_subset, pred_subset, split_name):
                        enriched = []
                        for row_index, row in enumerate(rows_subset):
                            item = dict(row)
                            item["source_latent"] = _project(row["source_latent"], pca_mean, pca_basis).tolist()
                            item["action_blocks_normalized"] = np.asarray(row["action_blocks_normalized"], dtype=np.float64).tolist()
                            item["action_blocks_raw"] = np.asarray(row["action_blocks_raw"], dtype=np.float64).tolist()
                            item["latent_by_horizon"] = {str(h): {**row["latent_by_horizon"][str(h)], "predicted": pred_subset[h][row_index].tolist(), "target": _project(row["latent_by_horizon"][str(h)]["target"], pca_mean, pca_basis).tolist()} for h in HORIZONS}
                            enriched.append(item)
                        return make_payload(architecture=f"pointmaze_{base_name}_{architecture}", checkpoint=str(checkpoint.resolve()), environment="PointMaze", seed=str(seed), training_stage=stage, latent_dimension=int(pca_basis.shape[0]), action_distribution=ACTION_DISTRIBUTION, split={"name": split_name, "episode_ids": [int(r["episode_id"]) for r in rows_subset], "provenance": "fixed frozen-export episode split; adapter trained on train subset only"}, horizons=HORIZONS, enriched_records=enriched, checkpoint_sha256=_sha256(checkpoint), dataset_sha256=_sha256(source_path), protocol={"base_frozen_export": str(source_path.resolve()), "adapter_architecture": architecture, "adapter_seed": int(seed), "split_role": split_name, "train_episode_disjoint": True, "pca_fit_split": "train_only", "action_distribution": ACTION_DISTRIBUTION}, study_scope="real learned PointMaze action-conditioned adapter panel", claim_boundary="This is a learned adapter over a frozen released temporal export; it is not evidence that the released frozen checkpoint has multiple training seeds.", runtime="pointmaze_learned_adapter_panel.v1")
                    candidate_root = args.output / "exports" / f"{base_name}_{architecture}_seed{seed}_stage{stage:03d}"
                    candidate_root.mkdir(parents=True, exist_ok=True)
                    fit_payload = export(fit_rows, pred_fit, "fit")
                    outcome_payload = export(outcome_rows, pred_outcome, "outcome")
                    fit_path = candidate_root / "fit.json"; out_path = candidate_root / "outcome.json"
                    fit_path.write_text(json.dumps(fit_payload, indent=2) + "\n"); out_path.write_text(json.dumps(outcome_payload, indent=2) + "\n")
                    from experiments.empirical_spectral_audit.build_learned_model_selection_records import _fit_metric, _ridge, _rows
                    from experiments.task_metric_diagnostics import metric_geometry
                    fit_source, fit_predicted, fit_target, fit_delta = _rows(fit_payload, 1)
                    centered = fit_source - fit_source.mean(0, keepdims=True)
                    covariance = centered.T @ centered / max(len(centered) - 1, 1)
                    predictor = _ridge(centered, fit_predicted - fit_predicted.mean(0, keepdims=True), 1e-3)
                    metric = _fit_metric(fit_source, fit_delta, 1e-3)
                    geometry = metric_geometry(covariance, predictor, metric, rank=min(16, fit_source.shape[1]), horizons=range(1, 5))
                    spectrum = [{"real": float(v.real), "imag": float(v.imag)} for v in np.linalg.eigvals(predictor)]
                    planning_loss = float(np.mean([np.mean((pred_outcome[h][i] - y_outcome[h][i]) ** 2) for h in HORIZONS for i in range(len(outcome_rows))]))
                    record = {"group": f"PointMaze/learned/{base_name}_{architecture}/seed_{seed}", "architecture": f"pointmaze_{base_name}_{architecture}", "checkpoint": str(checkpoint.resolve()), "model_factors": {"latent_dimension": int(pca_basis.shape[0]), "training_stage": stage, "action_distribution": ACTION_DISTRIBUTION, "action_inverse_risk": calibration_action_risk, "adapter_mode": training.get("mode", "absolute_target")}, "diagnostics": {"prediction_mse": float(np.mean((fit_predicted - fit_target) ** 2)), "predictor_spectrum": spectrum, **{k: float(v) for k, v in geometry.items()}, "action_inverse_risk": calibration_action_risk}, "heldout": {"planning_loss": planning_loss, "planning_risk": planning_risk, "planning_details": planning_details, "planning_risk_conservative": planning_risk_conservative, "planning_details_conservative": planning_details_conservative}, "protocol": {"environment": "PointMaze", "independent_external_evidence": True, "fit_episode_ids": [int(r["episode_id"]) for r in fit_rows], "calibration_episode_ids": [int(r["episode_id"]) for r in calibration_rows], "outcome_episode_ids": [int(r["episode_id"]) for r in outcome_rows], "action_feature_source": f"{args.action_feature_chart} action chart", "action_chart_role": "formal shared physical chart" if args.action_feature_chart == "raw" else "exploratory export-native chart", "action_risk_split": "calibration rows never used to fit adapter", "planning_split": "outcome rows never used for fitting, calibration, metric, or selector", "task_direction_source": "train-only h=1 source-to-target displacement", "task_readout_source": "train rows only, horizon-specific", "prediction_center_source": "train target latent mean only, horizon-specific", "candidate_action_bank": list(HORIZONS), "runtime_state_access": False, "physical_state_role": "offline consequence scoring only", "adapter_mode": training.get("mode", "absolute_target"), "conservative_gate": "calibration_task_rmse_penalty_beta1 + transfer_permission_v1; fallback candidate h=1 retained", "claim_boundary": "independent held-out PointMaze planning consequence from fixed horizon candidate bank"}, "provenance": {"fit_export": str(fit_path), "outcome_export": str(out_path), "base_frozen_export": str(source_path.resolve()), "training": training}}
                    record_dir = args.output / "records" / f"{base_name}_{architecture}_seed{seed}_stage{stage:03d}"
                    _write_record(record_dir, record)
                    all_records.append(record)
    manifest = {"schema": "pointmaze_learned_adapter_panel.v1", "protocol": {"architectures": list(args.architectures), "seeds": list(SEEDS), "training_stages": sorted(set(args.stages)), "action_distribution": ACTION_DISTRIBUTION, "action_feature_chart": args.action_feature_chart, "rank": args.rank, "epochs": args.epochs, "max_rows": args.max_rows, "outcome_count": args.outcome_count, "support_matching": args.max_rows is not None, "allow_unmatched_support": args.allow_unmatched_support, "support_match_rule": "same deterministic prefix of every export" if args.max_rows is not None else ("equal export sizes" if len(set(source_lengths.values())) == 1 else "unequal exploratory support"), "pca_fit_only_on_train": True, "planning_candidate_bank": list(HORIZONS), "outcome_used_for_selection": False, "calibration_used_for_training": False, "task_loss_weight": args.task_loss_weight, "adapter_modes": {"linear": "absolute_target", "mlp": "absolute_target", "residual_mlp": "standardized_residual", "standardized_mlp": "standardized_absolute", "task_aligned_mlp": "standardized_absolute_plus_train_task_loss", "task_effect_mlp": "train_task_effect_only", "task_effect_linear": "train_task_effect_only_linear", "task_effect_lowrank_mlp": "train_task_effect_only_with_train_action_pca4"}}, "inventories": inventories, "record_count": len(all_records), "claim_boundary": "Real learned adapter panel with independent calibration action-risk and held-out PointMaze planning consequence; not a multi-seed claim about the frozen encoder."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
