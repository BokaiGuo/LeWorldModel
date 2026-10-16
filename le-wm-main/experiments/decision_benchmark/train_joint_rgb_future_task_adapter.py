"""Train a causal RGB future/task-effect adapter on one shared candidate bank.

This is the smallest WA-JEPA-inspired intervention for the current
LeWorldModel decision benchmark.  The input is a causal RGB projection
history and a complete candidate action sequence.  The shared trunk predicts
both (i) the realized RGB successor residual and (ii) the physical terminal
XY effect.  All normalization and fitting use train roots only.  Calibration
and outcome roots are used only for reporting.

The experiment is deliberately intervention-supervised: physical terminal
states are simulator labels used to train/evaluate the adapter, never runtime
inputs.  A positive result would support an action-conditioned joint-readout
mechanism, not frozen-LeWM counterfactual causality or closed-loop planning.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


SEEDS = (8411, 8412, 8413, 8414, 8415)


class JointFutureTaskMLP(nn.Module):
    """Shared causal future trunk with visual, task-effect, and cost heads."""

    def __init__(self, input_dim: int, rgb_dim: int = 8, task_dim: int = 2) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
        )
        self.rgb_head = nn.Linear(128, rgb_dim)
        self.task_head = nn.Linear(128, task_dim)
        self.cost_head = nn.Linear(128, 1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(value)
        return self.rgb_head(hidden), self.task_head(hidden), self.cost_head(hidden)


class AsymmetricFutureTaskMLP(nn.Module):
    """Minimal asymmetric future/action coupling intervention.

    The future branch receives a detached action representation, so future
    supervision cannot train an action-only shortcut.  The task branch sees
    both the predicted future representation and the live action
    representation, so task/ranking supervision can shape the decision path.
    This is a local WA-JEPA-inspired ablation, not flow matching.
    """

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        rgb_dim: int = 8,
        task_dim: int = 2,
        future_steps: int = 1,
        relative_task: bool = False,
        direct_future_readout: bool = False,
        reference_factorized: bool = False,
    ) -> None:
        super().__init__()
        if future_steps < 1:
            raise ValueError("future_steps must be positive")
        self.future_steps = int(future_steps)
        self.rgb_dim = int(rgb_dim)
        self.relative_task = bool(relative_task)
        self.direct_future_readout = bool(direct_future_readout)
        self.reference_factorized = bool(reference_factorized)
        if self.reference_factorized and not self.relative_task:
            raise ValueError("reference_factorized requires relative_task")
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
        )
        self.future_trunk = nn.Sequential(
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
        )
        if self.relative_task:
            # Split the physical effect into a root-shared context baseline
            # and an action-conditioned residual. Ranking candidates within a
            # root should depend on the residual, not on baseline error.
            self.task_base_trunk = nn.Sequential(
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
            )
            self.task_effect_trunk = nn.Sequential(
                nn.Linear(384, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
            )
            self.task_base_head = nn.Linear(128, task_dim)
            self.task_effect_head = nn.Linear(128, task_dim)
        else:
            self.task_trunk = nn.Sequential(
                nn.Linear(384 + (2 * self.rgb_dim if self.direct_future_readout else 0), 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
            )
            self.task_head = nn.Linear(128, task_dim)
        self.rgb_head = nn.Linear(128, self.future_steps * self.rgb_dim)
        self.cost_head = nn.Linear(128, 1)

    def _future_hidden(
        self,
        context_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
    ) -> torch.Tensor:
        return self.future_trunk(
            torch.cat([context_hidden, action_hidden.detach()], dim=1)
        )

    def forward(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
        reference_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_hidden = self.context_encoder(context)
        action_hidden = self.action_encoder(action)
        future_hidden = self._future_hidden(context_hidden, action_hidden)
        rgb = self.rgb_head(future_hidden).reshape(-1, self.future_steps, self.rgb_dim)
        if self.future_steps == 1:
            rgb = rgb[:, 0]
        if self.relative_task:
            base_hidden = self.task_base_trunk(context_hidden)
            effect_hidden = self.task_effect_trunk(
                torch.cat([context_hidden, future_hidden, action_hidden], dim=1)
            )
            effect = self.task_effect_head(effect_hidden)
            if self.reference_factorized:
                if reference_action is None:
                    raise ValueError("reference_action is required for reference_factorized mode")
                reference_action_hidden = self.action_encoder(reference_action)
                reference_future_hidden = self._future_hidden(
                    context_hidden, reference_action_hidden,
                )
                reference_effect_hidden = self.task_effect_trunk(
                    torch.cat(
                        [context_hidden, reference_future_hidden, reference_action_hidden],
                        dim=1,
                    )
                )
                reference_effect = self.task_effect_head(reference_effect_hidden)
                # The reference action is a structural zero of the learned
                # candidate effect. This prevents root difficulty from being
                # absorbed by the action residual.
                task = self.task_base_head(base_hidden) + effect - reference_effect
            else:
                task = self.task_base_head(base_hidden) + effect
            task_hidden = effect_hidden
        else:
            task_input = torch.cat([context_hidden, future_hidden, action_hidden], dim=1)
            if self.direct_future_readout:
                if rgb.ndim != 3:
                    raise ValueError("direct future readout requires a future sequence")
                future_summary = torch.cat(
                    [rgb[:, -1], rgb[:, -1] - rgb[:, 0]], dim=1,
                )
                task_input = torch.cat([task_input, future_summary], dim=1)
            task_hidden = self.task_trunk(
                task_input
            )
            task = self.task_head(task_hidden)
        return rgb, task, self.cost_head(task_hidden)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable; refusing CPU fallback")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _array_key(reference: str) -> str:
    parts = str(reference).split("::", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid array reference: {reference}")
    return parts[1]


def _load_rows(
    dataset: Path,
    rgb_report: Path,
    arrays: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]], Any, dict[str, Any]]:
    dataset_payload = json.loads(dataset.read_text(encoding="utf-8"))
    rgb_payload = json.loads(rgb_report.read_text(encoding="utf-8"))
    records = list(dataset_payload["records"])
    rgb_rows = {
        (str(row["instance_id"]), int(row["candidate_id"])): row
        for row in rgb_payload["rows"]
    }
    missing = [
        (str(row["instance_id"]), int(row["candidate_id"]))
        for row in records
        if (str(row["instance_id"]), int(row["candidate_id"])) not in rgb_rows
    ]
    if missing:
        raise ValueError(f"RGB report lacks {len(missing)} candidate rows")
    array_payload = np.load(arrays, allow_pickle=False)
    return records, rgb_rows, array_payload, dict(dataset_payload.get("metadata", {}))


def _goal_relative_action_sequence(
    action_sequence: np.ndarray,
    source_state: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    """Express a raw 2-D control sequence in the root's goal-relative frame."""
    source_xy = np.asarray(source_state, dtype=np.float64).reshape(-1)[:2]
    goal_xy = np.asarray(goal, dtype=np.float64).reshape(-1)[:2]
    displacement = goal_xy - source_xy
    norm = float(np.linalg.norm(displacement))
    if norm <= 1.0e-8:
        raise ValueError("goal-relative action frame requires a nonzero source-goal displacement")
    direct = displacement / norm
    perpendicular = np.asarray([-direct[1], direct[0]], dtype=np.float64)
    sequence = np.asarray(action_sequence, dtype=np.float64)
    if sequence.ndim != 2 or sequence.shape[1] != 2:
        raise ValueError(f"action_sequence must have shape [steps, 2], got {sequence.shape}")
    return np.stack([sequence @ direct, sequence @ perpendicular], axis=-1)


def _action_sequence_in_frame(record: dict[str, Any], action_frame: str) -> np.ndarray:
    """Return a candidate intervention in the model's declared action frame."""
    if action_frame not in {"raw_control", "goal_relative", "hybrid"}:
        raise ValueError(f"unsupported action_frame: {action_frame}")
    sequence = np.asarray(record["action_sequence"], dtype=np.float64)
    if action_frame == "raw_control":
        return sequence
    goal_relative = _goal_relative_action_sequence(
        sequence,
        np.asarray(record["source_state_raw"], dtype=np.float64),
        np.asarray(record["goal_raw"], dtype=np.float64),
    )
    if action_frame == "goal_relative":
        return goal_relative
    # The hybrid representation keeps the physical control chart while also
    # exposing the goal-relative semantics. This is the action analogue of
    # WA-JEPA's action plus ego-state conditioning: canonicalization alone can
    # erase root-specific maze geometry, while raw controls alone do not share
    # candidate semantics across roots.
    return np.concatenate([sequence, goal_relative], axis=-1)


def _reference_index_by_root(
    records: list[dict[str, Any]],
    reference_candidate_id: int | None,
) -> dict[str, int]:
    """Select a structural reference candidate without reading outcome labels."""
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(str(record["instance_id"]), []).append(index)
    selected: dict[str, int] = {}
    for root, indices in grouped.items():
        if reference_candidate_id is not None:
            matches = [
                index for index in indices
                if int(records[index]["candidate_id"]) == int(reference_candidate_id)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"reference candidate {reference_candidate_id} must occur exactly once in root {root}"
                )
            selected[root] = matches[0]
        else:
            selected[root] = min(
                indices,
                key=lambda index: float(np.mean(np.square(
                    np.asarray(records[index]["action_sequence"], dtype=np.float64)
                ))),
            )
    return selected


def _features(
    records: list[dict[str, Any]],
    rgb_rows: dict[tuple[str, int], dict[str, Any]],
    arrays: Any,
    *,
    history_windows: int,
    context_family: str,
    reference_candidate_id: int | None = None,
    action_frame: str = "raw_control",
) -> dict[str, Any]:
    if history_windows < 1:
        raise ValueError("history_windows must be positive")
    if context_family not in {
        "rgb_projection", "latent_model", "realized_future_projection",
        "reference_future_projection",
    }:
        raise ValueError(f"unsupported context_family: {context_family}")
    if action_frame not in {"raw_control", "goal_relative", "hybrid"}:
        raise ValueError(f"unsupported action_frame: {action_frame}")
    x_values: list[np.ndarray] = []
    context_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    rgb_sequence_targets: list[np.ndarray] = []
    rgb_targets: list[np.ndarray] = []
    task_targets: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    source_xy: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    actual_terminal_xy: list[np.ndarray] = []
    cost_targets: list[float] = []
    roots: list[str] = []
    reference_by_root = _reference_index_by_root(records, reference_candidate_id)
    for record in records:
        key = (str(record["instance_id"]), int(record["candidate_id"]))
        rgb = rgb_rows[key]
        source_rgb = np.asarray(rgb["source_projection"], dtype=np.float64)
        if context_family in {"realized_future_projection", "reference_future_projection"}:
            context_rgb = rgb
            if context_family == "reference_future_projection":
                context_rgb = rgb_rows[
                    (
                        str(record["instance_id"]),
                        int(records[reference_by_root[str(record["instance_id"])]]["candidate_id"]),
                    )
                ]
            if "realized_future_projection" not in context_rgb:
                raise ValueError("RGB report lacks realized_future_projection for probe context")
            future_projection = np.asarray(context_rgb["realized_future_projection"], dtype=np.float64)
            if future_projection.ndim != 2:
                raise ValueError("realized_future_projection must be a [steps, dim] array")
            context = future_projection.reshape(-1)
        elif context_family == "latent_model":
            if history_windows == 1:
                context = np.asarray(record["source_latent_model"], dtype=np.float64)
            else:
                if record.get("source_history_model") is None:
                    raise ValueError("candidate record lacks source_history_model")
                context = np.asarray(record["source_history_model"], dtype=np.float64)
                if context.ndim != 1:
                    context = context.reshape(-1)
        else:
            if history_windows == 1:
                context = source_rgb
            else:
                history = np.asarray(rgb["source_history_projection"], dtype=np.float64)
                if history.ndim != 2 or history.shape[0] != history_windows:
                    raise ValueError(
                        "source history shape does not match history_windows: "
                        f"{history.shape} vs {history_windows}"
                    )
                context = history.reshape(-1)
        action = _action_sequence_in_frame(record, action_frame).reshape(-1)
        terminal_state = np.asarray(
            arrays[_array_key(record["realized_terminal_state_ref"])],
            dtype=np.float64,
        ).reshape(-1)
        source_state = np.asarray(record["source_state_raw"], dtype=np.float64).reshape(-1)
        if terminal_state.shape[0] < 2 or source_state.shape[0] < 2:
            raise ValueError("source and terminal states must contain XY")
        x_values.append(np.concatenate([context, action]))
        context_values.append(context)
        action_values.append(action)
        rgb_targets.append(np.asarray(rgb["realized_terminal_projection"], dtype=np.float64) - source_rgb)
        if "realized_future_projection" not in rgb:
            rgb_sequence_targets.append(None)
        else:
            future_projection = np.asarray(rgb["realized_future_projection"], dtype=np.float64)
            if future_projection.ndim != 2 or future_projection.shape[1] != source_rgb.shape[0]:
                raise ValueError(
                    "realized_future_projection must have shape [future_steps, projection_dim], "
                    f"got {future_projection.shape} for source {source_rgb.shape}"
                )
            rgb_sequence_targets.append(future_projection - source_rgb[None, :])
        task_targets.append(terminal_state[:2] - source_state[:2])
        sources.append(source_rgb)
        source_xy.append(source_state[:2])
        goals.append(np.asarray(record["goal_raw"], dtype=np.float64)[:2])
        actual_terminal_xy.append(terminal_state[:2])
        cost_targets.append(float(np.linalg.norm(
            terminal_state[:2] - np.asarray(record["goal_raw"], dtype=np.float64)[:2]
        )))
        roots.append(str(record["instance_id"]))
    action_matrix = np.asarray(action_values, dtype=np.float32)
    reference_actions: list[np.ndarray] = []
    reference_task_targets: list[np.ndarray] = []
    for root in roots:
        # Select the reference from the action bank only. Task outcomes are
        # never consulted when defining the reference candidate.
        reference_record_index = reference_by_root[root]
        reference_actions.append(action_matrix[reference_record_index])
        reference_task_targets.append(np.asarray(task_targets[reference_record_index], dtype=np.float32))
    return {
        "x": np.asarray(x_values, dtype=np.float32),
        "context": np.asarray(context_values, dtype=np.float32),
        "action": np.asarray(action_values, dtype=np.float32),
        "reference_action": np.asarray(reference_actions, dtype=np.float32),
        "rgb_target": np.asarray(rgb_targets, dtype=np.float32),
        "rgb_sequence_target": (
            np.asarray(rgb_sequence_targets, dtype=np.float32)
            if all(value is not None for value in rgb_sequence_targets)
            else None
        ),
        "task_target": np.asarray(task_targets, dtype=np.float32),
        "reference_task_target": np.asarray(reference_task_targets, dtype=np.float32),
        "source_rgb": np.asarray(sources, dtype=np.float32),
        "source_xy": np.asarray(source_xy, dtype=np.float64),
        "goals": np.asarray(goals, dtype=np.float64),
        "actual_terminal_xy": np.asarray(actual_terminal_xy, dtype=np.float64),
        "cost_target": np.asarray(cost_targets, dtype=np.float32),
        "roots": roots,
        "action_frame": action_frame,
    }


def _fit_one(
    train: dict[str, Any],
    all_data: dict[str, Any],
    *,
    seed: int,
    epochs: int,
    device: torch.device,
    objective: str,
    rgb_weight: float,
    temporal_weight: float,
    ranking_weight: float,
    ranking_margin: float,
) -> dict[str, Any]:
    if objective not in {
        "joint", "joint_rank", "joint_cost", "cost_only", "task_only", "rgb_only",
        "asym_joint", "asym_rank", "asym_preference", "asym_sequence",
        "asym_sequence_temporal", "asym_relative", "asym_relative_pair",
        "asym_sequence_direct", "asym_reference", "asym_reference_supervised",
    }:
        raise ValueError(f"unsupported objective: {objective}")
    _seed(seed)
    x_train = np.asarray(train["x"], dtype=np.float64)
    x_all = np.asarray(all_data["x"], dtype=np.float64)
    sequence = objective in {
        "asym_sequence", "asym_sequence_temporal", "asym_sequence_direct",
    }
    if sequence:
        if train.get("rgb_sequence_target") is None or all_data.get("rgb_sequence_target") is None:
            raise ValueError("asym_sequence requires realized_future_projection in the RGB report")
        rgb_train = np.asarray(train["rgb_sequence_target"], dtype=np.float64)
        rgb_all = np.asarray(all_data["rgb_sequence_target"], dtype=np.float64)
    else:
        rgb_train = np.asarray(train["rgb_target"], dtype=np.float64)
        rgb_all = np.asarray(all_data["rgb_target"], dtype=np.float64)
    task_train = np.asarray(train["task_target"], dtype=np.float64)
    cost_train = np.asarray(train["cost_target"], dtype=np.float64)
    x_mean = x_train.mean(axis=0)
    x_scale = np.maximum(x_train.std(axis=0), 1.0e-3)
    rgb_mean = rgb_train.reshape(-1, rgb_train.shape[-1]).mean(axis=0)
    rgb_scale = np.maximum(rgb_train.reshape(-1, rgb_train.shape[-1]).std(axis=0), 1.0e-3)
    task_mean = task_train.mean(axis=0)
    task_scale = np.maximum(task_train.std(axis=0), 1.0e-3)
    cost_mean = float(cost_train.mean())
    cost_scale = float(max(cost_train.std(), 1.0e-3))
    asymmetric = objective in {
        "asym_joint", "asym_rank", "asym_preference", "asym_sequence",
        "asym_sequence_temporal", "asym_relative", "asym_relative_pair",
        "asym_sequence_direct", "asym_reference", "asym_reference_supervised",
    }
    relative_task = objective in {
        "asym_relative", "asym_relative_pair", "asym_reference", "asym_reference_supervised",
    }
    reference_factorized = objective in {"asym_reference", "asym_reference_supervised"}
    if asymmetric:
        context_train = np.asarray(train["context"], dtype=np.float64)
        context_all = np.asarray(all_data["context"], dtype=np.float64)
        action_train = np.asarray(train["action"], dtype=np.float64)
        action_all = np.asarray(all_data["action"], dtype=np.float64)
        reference_action_train = np.asarray(train["reference_action"], dtype=np.float64)
        reference_action_all = np.asarray(all_data["reference_action"], dtype=np.float64)
        context_mean = context_train.mean(axis=0)
        context_scale = np.maximum(context_train.std(axis=0), 1.0e-3)
        action_mean = action_train.mean(axis=0)
        action_scale = np.maximum(action_train.std(axis=0), 1.0e-3)
        model = AsymmetricFutureTaskMLP(
            context_train.shape[1], action_train.shape[1], rgb_dim=rgb_train.shape[-1],
            future_steps=rgb_train.shape[1] if sequence else 1,
            relative_task=relative_task,
            direct_future_readout=objective == "asym_sequence_direct",
            reference_factorized=reference_factorized,
        ).to(device)
    else:
        model = JointFutureTaskMLP(x_train.shape[1], rgb_dim=rgb_train.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    x_fit = torch.as_tensor((x_train - x_mean) / x_scale, dtype=torch.float32, device=device)
    if asymmetric:
        context_fit = torch.as_tensor(
            (context_train - context_mean) / context_scale,
            dtype=torch.float32, device=device,
        )
        action_fit = torch.as_tensor(
            (action_train - action_mean) / action_scale,
            dtype=torch.float32, device=device,
        )
        reference_action_fit = torch.as_tensor(
            (reference_action_train - action_mean) / action_scale,
            dtype=torch.float32, device=device,
        )
    rgb_fit = torch.as_tensor(
        (rgb_train - rgb_mean) / rgb_scale, dtype=torch.float32, device=device,
    )
    task_fit = torch.as_tensor((task_train - task_mean) / task_scale, dtype=torch.float32, device=device)
    reference_task_train = np.asarray(train["reference_task_target"], dtype=np.float64)
    reference_task_fit = torch.as_tensor(
        (reference_task_train - task_mean) / task_scale,
        dtype=torch.float32, device=device,
    )
    cost_fit = torch.as_tensor((cost_train - cost_mean) / cost_scale, dtype=torch.float32, device=device)
    source_xy_fit = torch.as_tensor(train["source_xy"], dtype=torch.float32, device=device)
    goals_fit = torch.as_tensor(train["goals"], dtype=torch.float32, device=device)
    actual_terminal_fit = torch.as_tensor(train["actual_terminal_xy"], dtype=torch.float32, device=device)
    rank_groups: list[torch.Tensor] = []
    for root in sorted(set(train["roots"])):
        rank_groups.append(torch.as_tensor(
            [index for index, candidate_root in enumerate(train["roots"]) if candidate_root == root],
            dtype=torch.long, device=device,
        ))
    true_cost_fit = torch.linalg.vector_norm(actual_terminal_fit - goals_fit, dim=1)
    history: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        if asymmetric:
            rgb_pred, task_pred, cost_pred = model(
                context_fit,
                action_fit,
                reference_action_fit if reference_factorized else None,
            )
        else:
            rgb_pred, task_pred, cost_pred = model(x_fit)
        rgb_loss = torch.mean((rgb_pred - rgb_fit) ** 2)
        temporal_loss = torch.zeros((), dtype=torch.float32, device=device)
        if sequence:
            # A pointwise MSE objective is a shrinkage-prone estimator for
            # multimodal/nonstationary future trajectories.  Match the
            # first finite difference explicitly so that a good terminal
            # average cannot be obtained by suppressing temporal amplitude.
            predicted_delta = torch.diff(rgb_pred, dim=1)
            target_delta = torch.diff(rgb_fit, dim=1)
            temporal_loss = torch.mean((predicted_delta - target_delta) ** 2)
        task_loss = torch.mean((task_pred - task_fit) ** 2)
        reference_loss = torch.zeros((), dtype=torch.float32, device=device)
        relative_effect_loss = torch.zeros((), dtype=torch.float32, device=device)
        if objective == "asym_reference_supervised":
            _, reference_pred, _ = model(
                context_fit, reference_action_fit, reference_action_fit,
            )
            reference_loss = torch.mean((reference_pred - reference_task_fit) ** 2)
            relative_effect_loss = torch.mean(
                ((task_pred - reference_pred) - (task_fit - reference_task_fit)) ** 2
            )
        cost_loss = torch.mean((cost_pred[:, 0] - cost_fit) ** 2)
        rank_loss = torch.zeros((), dtype=torch.float32, device=device)
        effect_loss = torch.zeros((), dtype=torch.float32, device=device)
        rank_pairs = 0
        if objective in {"joint_rank", "asym_rank", "asym_preference"}:
            if objective == "asym_preference":
                predicted_cost = cost_pred[:, 0] * cost_scale + cost_mean
            else:
                predicted_task = task_pred * torch.as_tensor(task_scale, dtype=torch.float32, device=device)
                predicted_task = predicted_task + torch.as_tensor(task_mean, dtype=torch.float32, device=device)
                predicted_terminal = source_xy_fit + predicted_task
                predicted_cost = torch.linalg.vector_norm(predicted_terminal - goals_fit, dim=1)
            rank_terms = []
            for group in rank_groups:
                if int(group.numel()) < 2:
                    continue
                pair_i, pair_j = torch.triu_indices(int(group.numel()), int(group.numel()), offset=1, device=device)
                global_i, global_j = group[pair_i], group[pair_j]
                true_delta = true_cost_fit[global_i] - true_cost_fit[global_j]
                valid = torch.abs(true_delta) > 1.0e-3
                if bool(valid.any()):
                    sign = torch.sign(true_delta[valid])
                    predicted_delta = predicted_cost[global_i[valid]] - predicted_cost[global_j[valid]]
                    rank_terms.append(torch.relu(float(ranking_margin) - sign * predicted_delta))
                    rank_pairs += int(valid.sum().item())
            if rank_terms:
                rank_loss = torch.cat(rank_terms).mean()
        if objective == "asym_relative_pair":
            effect_terms = []
            for group in rank_groups:
                if int(group.numel()) < 2:
                    continue
                pair_i, pair_j = torch.triu_indices(
                    int(group.numel()), int(group.numel()), offset=1, device=device,
                )
                global_i, global_j = group[pair_i], group[pair_j]
                predicted_delta = task_pred[global_i] - task_pred[global_j]
                true_delta = task_fit[global_i] - task_fit[global_j]
                effect_terms.append(torch.mean((predicted_delta - true_delta) ** 2))
            if effect_terms:
                effect_loss = torch.stack(effect_terms).mean()
        if objective in {
            "joint", "asym_joint", "asym_sequence", "asym_sequence_temporal",
            "asym_sequence_direct", "asym_reference", "asym_reference_supervised",
            "asym_relative",
        }:
            loss = task_loss + float(rgb_weight) * rgb_loss
            if objective == "asym_sequence_temporal":
                loss = loss + float(temporal_weight) * temporal_loss
            if objective == "asym_reference_supervised":
                loss = loss + float(ranking_weight) * (reference_loss + relative_effect_loss)
        elif objective == "asym_relative_pair":
            loss = task_loss + float(rgb_weight) * rgb_loss + float(ranking_weight) * effect_loss
        elif objective in {"joint_rank", "asym_rank", "asym_preference"}:
            loss = task_loss + float(rgb_weight) * rgb_loss + float(ranking_weight) * rank_loss
        elif objective == "joint_cost":
            loss = task_loss + float(rgb_weight) * rgb_loss + float(ranking_weight) * cost_loss
        elif objective == "cost_only":
            loss = cost_loss
        elif objective == "task_only":
            loss = task_loss
        else:
            loss = rgb_loss
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    with torch.inference_mode():
        if asymmetric:
            context_tensor = torch.as_tensor(
                (context_all - context_mean) / context_scale,
                dtype=torch.float32, device=device,
            )
            action_tensor = torch.as_tensor(
                (action_all - action_mean) / action_scale,
                dtype=torch.float32, device=device,
            )
            reference_action_tensor = torch.as_tensor(
                (reference_action_all - action_mean) / action_scale,
                dtype=torch.float32, device=device,
            )
            rgb_pred, task_pred, cost_pred = model(
                context_tensor,
                action_tensor,
                reference_action_tensor if reference_factorized else None,
            )
        else:
            all_tensor = torch.as_tensor(
                (x_all - x_mean) / x_scale, dtype=torch.float32, device=device,
            )
            rgb_pred, task_pred, cost_pred = model(all_tensor)
    rgb_output = rgb_pred.cpu().numpy() * rgb_scale + rgb_mean
    if sequence:
        rgb_sequence_output = np.asarray(rgb_output, dtype=np.float64)
        rgb_output = rgb_sequence_output[:, -1]
    else:
        rgb_sequence_output = None
    task_output = task_pred.cpu().numpy() * task_scale + task_mean
    return {
        "rgb_prediction": np.asarray(rgb_output, dtype=np.float64),
        "rgb_sequence_prediction": rgb_sequence_output,
        "task_prediction": np.asarray(task_output, dtype=np.float64),
        "cost_prediction": np.asarray(
            cost_pred[:, 0].cpu().numpy() * cost_scale + cost_mean,
            dtype=np.float64,
        ),
        "training": {
            "seed": int(seed),
            "objective": objective,
            "rgb_weight": float(rgb_weight),
            "temporal_weight": float(temporal_weight),
            "ranking_weight": float(ranking_weight),
            "ranking_margin": float(ranking_margin),
            "ranking_pairs_per_step": int(rank_pairs),
            "epochs": int(epochs),
            "final_loss": float(history[-1]),
            "device": str(device),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "architecture": (
                "asymmetric_future_action_sequence_direct" if objective == "asym_sequence_direct"
                else "asymmetric_future_action_reference_supervised" if objective == "asym_reference_supervised"
                else "asymmetric_future_action_reference_factorized" if objective == "asym_reference"
                else "asymmetric_future_action_relative_pair" if objective == "asym_relative_pair"
                else "asymmetric_future_action_relative" if objective == "asym_relative"
                else "asymmetric_future_action_sequence_temporal" if objective == "asym_sequence_temporal"
                else "asymmetric_future_action_sequence" if objective == "asym_sequence"
                else "asymmetric_future_action_preference" if objective == "asym_preference"
                else "asymmetric_future_action" if asymmetric else "shared_joint_mlp"
            ),
        },
    }


def _root_ranking(
    roots: list[str],
    predicted_terminal: np.ndarray,
    actual_terminal: np.ndarray,
    goals: np.ndarray,
    *,
    success_threshold: float,
    predicted_cost_override: np.ndarray | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = {}
    for index, root in enumerate(roots):
        grouped.setdefault(root, []).append(index)
    selected_regret: list[float] = []
    random_regret: list[float] = []
    selected_success: list[float] = []
    actual_costs: list[float] = []
    predicted_spread: list[float] = []
    actual_spread: list[float] = []
    for root, indices in sorted(grouped.items()):
        idx = np.asarray(indices, dtype=np.int64)
        if predicted_cost_override is None:
            pred_cost = np.linalg.norm(predicted_terminal[idx] - goals[idx], axis=1)
        else:
            pred_cost = np.asarray(predicted_cost_override[idx], dtype=np.float64)
        actual_cost = np.linalg.norm(actual_terminal[idx] - goals[idx], axis=1)
        selected = int(idx[int(np.argmin(pred_cost))])
        oracle = float(np.min(actual_cost))
        rng = np.random.default_rng(8411 + len(root))
        random_index = int(idx[int(rng.integers(0, len(idx)))])
        selected_regret.append(float(np.linalg.norm(actual_terminal[selected] - goals[selected]) - oracle))
        random_regret.append(float(np.linalg.norm(actual_terminal[random_index] - goals[random_index]) - oracle))
        selected_success.append(float(np.linalg.norm(actual_terminal[selected] - goals[selected]) <= success_threshold))
        actual_costs.extend(actual_cost.tolist())
        predicted_spread.append(float(np.mean(np.linalg.norm(predicted_terminal[idx] - np.mean(predicted_terminal[idx], axis=0), axis=1))))
        actual_spread.append(float(np.mean(np.linalg.norm(actual_terminal[idx] - np.mean(actual_terminal[idx], axis=0), axis=1))))
    paired_gain = np.asarray(random_regret) - np.asarray(selected_regret)
    return {
        "roots": int(len(grouped)),
        "selected_regret_mean": float(np.mean(selected_regret)),
        "random_regret_mean": float(np.mean(random_regret)),
        "paired_regret_gain_mean": float(np.mean(paired_gain)),
        "selected_success_rate": float(np.mean(selected_success)),
        "actual_candidate_cost_mean": float(np.mean(actual_costs)),
        "predicted_candidate_spread_mean": float(np.mean(predicted_spread)),
        "actual_candidate_spread_mean": float(np.mean(actual_spread)),
        "candidate_spread_ratio": float(np.mean(predicted_spread) / max(np.mean(actual_spread), 1.0e-8)),
        "selected_regret_values": selected_regret,
        "paired_regret_gain_values": paired_gain.tolist(),
        "direct_cost_head_used": bool(predicted_cost_override is not None),
    }


def _split_report(
    data: dict[str, Any],
    predictions: dict[str, Any],
    mask: np.ndarray,
    *,
    success_threshold: float,
    allow_cost_head: bool = False,
) -> dict[str, Any]:
    task_prediction = np.asarray(predictions["task_prediction"], dtype=np.float64)[mask]
    task_target = np.asarray(data["task_target"], dtype=np.float64)[mask]
    rgb_prediction = np.asarray(predictions["rgb_prediction"], dtype=np.float64)[mask]
    rgb_target = np.asarray(data["rgb_target"], dtype=np.float64)[mask]
    source_xy = np.asarray(data["source_xy"], dtype=np.float64)[mask]
    goals = np.asarray(data["goals"], dtype=np.float64)[mask]
    actual_terminal = np.asarray(data["actual_terminal_xy"], dtype=np.float64)[mask]
    predicted_terminal = source_xy + task_prediction
    cost_prediction = predictions.get("cost_prediction") if allow_cost_head else None
    if cost_prediction is not None:
        cost_prediction = np.asarray(cost_prediction, dtype=np.float64)[mask]
    report = {
        "rows": int(np.sum(mask)),
        "rgb_residual_rmse": float(np.sqrt(np.mean((rgb_prediction - rgb_target) ** 2))),
        "task_effect_rmse": float(np.sqrt(np.mean((task_prediction - task_target) ** 2))),
        "ranking": _root_ranking(
            [root for root, keep in zip(data["roots"], mask) if keep],
            predicted_terminal, actual_terminal, goals,
            success_threshold=success_threshold,
            predicted_cost_override=cost_prediction,
        ),
    }
    sequence_prediction = predictions.get("rgb_sequence_prediction")
    sequence_target = data.get("rgb_sequence_target")
    if sequence_prediction is not None and sequence_target is not None:
        predicted_sequence = np.asarray(sequence_prediction, dtype=np.float64)[mask]
        target_sequence = np.asarray(sequence_target, dtype=np.float64)[mask]
        predicted_delta = np.diff(predicted_sequence, axis=1)
        target_delta = np.diff(target_sequence, axis=1)
        predicted_change = float(np.mean(np.linalg.norm(predicted_delta, axis=-1)))
        target_change = float(np.mean(np.linalg.norm(target_delta, axis=-1)))
        report["future_sequence"] = {
            "steps": int(predicted_sequence.shape[1]),
            "rmse": float(np.sqrt(np.mean((predicted_sequence - target_sequence) ** 2))),
            "change_magnitude_ratio": float(predicted_change / max(target_change, 1.0e-8)),
            "target_change_magnitude": target_change,
            "prediction_change_magnitude": predicted_change,
        }
    return report


def run(
    dataset: Path,
    rgb_report: Path,
    arrays: Path,
    output: Path,
    *,
    epochs: int = 250,
    device: str = "cuda",
    history_windows: int = 4,
    context_family: str = "rgb_projection",
    reference_candidate_id: int | None = None,
    rgb_weight: float = 1.0,
    temporal_weight: float = 0.5,
    ranking_weight: float = 0.5,
    ranking_margin: float = 0.005,
    success_threshold: float = 0.25,
    action_frame: str = "raw_control",
) -> dict[str, Any]:
    runtime_device = _device(device)
    records, rgb_rows, array_payload, dataset_metadata = _load_rows(dataset, rgb_report, arrays)
    data = _features(
        records, rgb_rows, array_payload,
        history_windows=history_windows, context_family=context_family,
        reference_candidate_id=reference_candidate_id,
        action_frame=action_frame,
    )
    train_mask = np.asarray([str(row["split"]) == "train" for row in records])
    calibration_mask = np.asarray([str(row["split"]) == "calibration" for row in records])
    outcome_mask = np.asarray([str(row["split"]) == "outcome" for row in records])
    if int(np.sum(train_mask)) < 2 or int(np.sum(outcome_mask)) < 2:
        raise ValueError("train and outcome roots are required")
    train_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            train_data[key] = value[train_mask]
        elif value is None:
            train_data[key] = None
        else:
            train_data[key] = [v for v, keep in zip(value, train_mask) if keep]
    reports: dict[str, Any] = {}
    objectives = [
        "task_only", "joint", "joint_rank", "joint_cost", "cost_only",
        "asym_joint", "asym_rank", "asym_preference", "asym_relative", "asym_relative_pair",
        "asym_reference", "asym_reference_supervised",
    ]
    if data["rgb_sequence_target"] is not None:
        objectives.append("asym_sequence")
        objectives.append("asym_sequence_temporal")
        objectives.append("asym_sequence_direct")
    for objective in objectives:
        seed_reports = []
        predictions = []
        for seed in SEEDS:
            fit = _fit_one(
                train_data, data, seed=seed, epochs=epochs,
                device=runtime_device, objective=objective, rgb_weight=rgb_weight,
                temporal_weight=temporal_weight,
                ranking_weight=ranking_weight, ranking_margin=ranking_margin,
            )
            predictions.append(fit)
            seed_reports.append({
                "seed": int(seed),
                "training": fit["training"],
                "calibration": _split_report(
                    data, fit, calibration_mask, success_threshold=success_threshold,
                    allow_cost_head=objective in {"joint_cost", "cost_only", "asym_preference"},
                ),
                "outcome": _split_report(
                    data, fit, outcome_mask, success_threshold=success_threshold,
                    allow_cost_head=objective in {"joint_cost", "cost_only", "asym_preference"},
                ),
            })
        mean_rgb = np.mean(np.stack([item["rgb_prediction"] for item in predictions]), axis=0)
        mean_task = np.mean(np.stack([item["task_prediction"] for item in predictions]), axis=0)
        mean_cost = np.mean(np.stack([item["cost_prediction"] for item in predictions]), axis=0)
        ensemble = {
            "rgb_prediction": mean_rgb,
            "rgb_sequence_prediction": (
                np.mean(np.stack([item["rgb_sequence_prediction"] for item in predictions]), axis=0)
                if predictions[0]["rgb_sequence_prediction"] is not None else None
            ),
            "task_prediction": mean_task,
            "cost_prediction": mean_cost if objective in {"joint_cost", "cost_only", "asym_preference"} else None,
        }
        reports[objective] = {
            "seeds": seed_reports,
            "ensemble": {
                "calibration": _split_report(
                    data, ensemble, calibration_mask, success_threshold=success_threshold,
                    allow_cost_head=objective in {"joint_cost", "cost_only", "asym_preference"},
                ),
                "outcome": _split_report(
                    data, ensemble, outcome_mask, success_threshold=success_threshold,
                    allow_cost_head=objective in {"joint_cost", "cost_only", "asym_preference"},
                ),
            },
        }
    report = {
        "schema": "pointmaze_joint_rgb_future_task_adapter.v6",
        "dataset": str(dataset.resolve()),
        "rgb_report": str(rgb_report.resolve()),
        "arrays": str(arrays.resolve()),
        "history_windows": int(history_windows),
        "context_family": context_family,
        "action_frame": action_frame,
        "candidate_bank_coordinate_frame": dataset_metadata.get("candidate_bank_coordinate_frame"),
        "candidate_bank_hashes": sorted({str(record.get("candidate_bank_hash")) for record in records}),
        "objectives": reports,
        "protocol": {
            "fit_split": "train roots only",
            "normalization_fit_split": "train roots only",
            "epoch_selection": "fixed predeclared epoch count",
            "rgb_target": "realized RGB terminal projection minus source projection",
            "task_target": "physical terminal XY displacement minus source XY",
            "action_conditioning": "complete candidate action sequence",
            "action_frame": action_frame,
            "action_frame_transform": (
                "per-root raw_control to goal_relative coordinates"
                if action_frame == "goal_relative"
                else "raw_control concatenated with per-root goal_relative coordinates"
                if action_frame == "hybrid"
                else "identity raw_control coordinates"
            ),
            "reference_action": (
                "explicit action-bank candidate selected per root without outcome labels"
                if reference_candidate_id is not None
                else "minimum-energy action-bank candidate selected per root without outcome labels"
            ),
            "reference_candidate_id": reference_candidate_id,
            "causal_context": "source history context only; no future observation input",
            "context_family": context_family,
            "device": str(runtime_device),
            "rgb_weight": float(rgb_weight),
            "temporal_weight": float(temporal_weight),
            "ranking_weight": float(ranking_weight),
            "ranking_margin": float(ranking_margin),
            "success_threshold": float(success_threshold),
            "outcome_used_for_training_or_selection": False,
            "runtime_privileged_state": False,
            "future_observation_used_for_offline_oracle": context_family == "realized_future_projection",
            "active_reference_probe_context": context_family == "reference_future_projection",
            "runtime_future_observation_input": False,
            "asymmetric_objectives": [
                "asym_joint", "asym_rank", "asym_preference", "asym_sequence",
                "asym_sequence_temporal", "asym_relative", "asym_relative_pair",
                "asym_sequence_direct", "asym_reference", "asym_reference_supervised",
            ],
        },
        "gates": {
            "train_root_only_fit": True,
            "root_disjoint_outcome": True,
            "runtime_state_not_input": True,
            "closed_loop_online_planning": False,
        },
        "claim_boundary": "intervention-supervised joint action-conditioned RGB/task-effect adapter; reference_future_projection is an active-probe mechanism arm, not passive frozen-LeWM planning, counterfactual causality, or universal transfer",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rgb-report", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--history-windows", type=int, default=4)
    parser.add_argument(
        "--reference-candidate-id", type=int, default=None,
        help="fixed action-bank candidate used as the active reference probe; defaults to minimum energy",
    )
    parser.add_argument(
        "--context-family",
        choices=(
            "rgb_projection", "latent_model", "realized_future_projection",
            "reference_future_projection",
        ),
        default="rgb_projection",
    )
    parser.add_argument("--rgb-weight", type=float, default=1.0)
    parser.add_argument("--temporal-weight", type=float, default=0.5)
    parser.add_argument("--success-threshold", type=float, default=0.25)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--ranking-margin", type=float, default=0.005)
    parser.add_argument(
        "--action-frame", choices=("raw_control", "goal_relative", "hybrid"), default="raw_control",
        help="coordinate frame used by the action-conditioned adapter",
    )
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps({
        objective: value["ensemble"]["outcome"]
        for objective, value in report["objectives"].items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
