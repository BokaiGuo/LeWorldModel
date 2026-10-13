"""Canonical state/pose helpers for the scalar-angle PushT environment.

``swm/PushT-v1`` exposes raw states as
``[agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]``.
Some learned-model interfaces deliberately convert this to a six/eight
dimensional sin/cos chart; those chart arrays must not be fed back into these
raw-state helpers.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(value: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle to ``[-pi, pi)`` without changing its scalar/array shape."""
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def raw_block_angle(state: np.ndarray) -> float:
    """Read the scalar block angle from a raw PushT state."""
    value = np.asarray(state, dtype=np.float64)
    if value.shape[-1] < 5:
        raise ValueError("raw PushT state must have at least five components")
    return float(value[4])


def raw_angle_error(value: float | np.ndarray, target: float | np.ndarray) -> float | np.ndarray:
    """Absolute wrapped error for scalar-angle raw states or scalar angles."""
    return np.abs(wrap_angle(np.asarray(value) - np.asarray(target)))


def raw_state_to_pose_chart(state: np.ndarray) -> np.ndarray:
    """Convert a raw state to ``[agent_xy, block_xy, sin, cos]``."""
    value = np.asarray(state, dtype=np.float64)
    return np.asarray(
        [value[0], value[1], value[2], value[3], np.sin(value[4]), np.cos(value[4])],
        dtype=np.float64,
    )
