"""Shared PointMaze action-coordinate transforms.

The released PointMaze exports store controls in world coordinates, while the
task readout is defined relative to the fixed legacy goal. This module keeps
the conversion identical between adapter training and external evaluation.
The physical simulator always receives the original raw controls; only the
learned adapter feature chart is transformed.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


DEFAULT_POINTMAZE_GOAL = np.asarray([1.2, 1.2], dtype=np.float64)


def goal_relative_action_blocks(
    rows: Sequence[dict[str, Any]],
    action_blocks: np.ndarray,
    *,
    goal: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate each row's action sequence into its source-to-goal frame."""

    values = np.asarray(action_blocks, dtype=np.float64)
    if values.ndim not in (2, 3) or values.shape[0] != len(rows):
        raise ValueError(
            "action_blocks must have shape [N,40] or [N,4,10], "
            f"got {values.shape} for {len(rows)} rows"
        )
    original_shape = values.shape
    flat = values.reshape(len(rows), -1)
    if flat.shape[1] % 2:
        raise ValueError(f"action feature width must be even, got {flat.shape[1]}")
    actions = flat.reshape(len(rows), -1, 2)
    default_goal = DEFAULT_POINTMAZE_GOAL if goal is None else np.asarray(goal, dtype=np.float64)
    if default_goal.shape != (2,):
        raise ValueError(f"goal must have shape [2], got {default_goal.shape}")

    output = np.empty_like(actions)
    for index, row in enumerate(rows):
        source = np.asarray(row["source_state_raw"], dtype=np.float64).reshape(-1)
        row_goal = row.get("goal_raw")
        target = default_goal if row_goal is None else np.asarray(row_goal, dtype=np.float64).reshape(-1)[:2]
        if source.size < 2 or target.shape != (2,):
            raise ValueError("source_state_raw and goal_raw must provide XY coordinates")
        direction = target - source[:2]
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-8:
            raise ValueError(f"source and goal coincide for row {index}")
        direction /= norm
        perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        output[index, :, 0] = actions[index] @ direction
        output[index, :, 1] = actions[index] @ perpendicular
    return output.reshape(original_shape).astype(np.float32)


def action_blocks_for_chart(
    rows: Sequence[dict[str, Any]],
    action_blocks: np.ndarray,
    chart: str,
) -> np.ndarray:
    """Return one of the declared physical action feature charts."""

    values = np.asarray(action_blocks, dtype=np.float32)
    if chart in {"raw", "export"}:
        return values
    if chart == "goal_relative":
        return goal_relative_action_blocks(rows, values)
    if chart == "hybrid":
        relative = goal_relative_action_blocks(rows, values)
        return np.concatenate([values, relative], axis=-1).astype(np.float32)
    raise ValueError(f"unsupported action feature chart: {chart}")
