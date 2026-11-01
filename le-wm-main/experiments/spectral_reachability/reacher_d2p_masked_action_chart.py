"""Exact raw-torque subspaces without degenerate affine action boxes.

An inactive actuator is literal zero, not weak torque or a clipped normalized
coordinate. History remains the actual unrestricted observed history. Only
future command coordinates are reduced; normalization occurs exactly once.
"""
from __future__ import annotations

import torch

TASKS = {"full_drive": (0, 1), "shoulder_only": (0,), "wrist_only": (1,)}


def active_columns(task, horizon=1):
    if task not in TASKS or not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("declared actuator task and positive packet horizon required")
    return [2 * step + joint for step in range(5 * horizon) for joint in TASKS[task]]


def expand_commands(commands, task):
    if commands.ndim != 3 or commands.shape[1] < 1 or commands.shape[-1] != len(active_columns(task)):
        raise ValueError("batch-by-horizon reduced raw command packets required")
    columns = torch.tensor(active_columns(task), dtype=torch.long, device=commands.device)
    return torch.zeros((*commands.shape[:-1], 10), dtype=commands.dtype, device=commands.device).index_copy(-1, columns, commands)


def normalize_raw(raw, mean, std):
    if (raw.ndim != 3 or raw.shape[-1] != 10 or mean.shape != (10,) or std.shape != (10,)
            or raw.device != mean.device or raw.device != std.device or raw.dtype != mean.dtype or raw.dtype != std.dtype
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or bool((std < 1e-6).any())):
        raise ValueError("compatible raw packets and finite checkpoint statistics required")
    return (raw - mean) / std


def terminal_normalized(model, emb, history, future):
    if (future.ndim != 3 or future.shape[1] < 1 or future.shape[-1] != 10
            or emb.shape[:2] != history.shape[:2] or emb.shape[0] != future.shape[0] or history.shape[-1] != 10
            or emb.shape[1] < 2 or emb.device != history.device or emb.device != future.device):
        raise ValueError("root-anchored aligned latent/action/future context required")
    size = emb.shape[1]
    actions = torch.cat((history[:, 1:], future[:, :1]), dim=1)
    state = emb
    for i in range(future.shape[1]):
        value = model.predict(state[:, -size:], model.action_encoder(actions[:, -size:]))[:, -1]
        if i + 1 < future.shape[1]:
            state = torch.cat((state, value[:, None]), dim=1)
            actions = torch.cat((actions, future[:, i + 1:i + 2]), dim=1)
    return value


def terminal(model, emb, history, commands, *, task, mean, std):
    return terminal_normalized(model, emb, history, normalize_raw(expand_commands(commands, task), mean, std))
