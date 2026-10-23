"""Dependency-light immutable contracts shared by PushT oracle and learners."""

from __future__ import annotations

import hashlib

import numpy as np


def action_bank_fingerprint(actions: np.ndarray) -> str:
    """Hash an ordered candidate matrix; candidate order is part of the bank."""

    actions = np.ascontiguousarray(np.asarray(actions, dtype=np.float64))
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError("action bank must be non-empty [candidate, action_dim]")
    return hashlib.sha256(actions.tobytes()).hexdigest()


def root_fingerprint(root: dict[str, np.ndarray]) -> str:
    """Hash a complete PushT source/goal root without serialising it."""

    if set(root) != {"state", "goal"}:
        raise ValueError("root must contain exactly state and goal")
    state = np.asarray(root["state"], dtype=np.float64).reshape(-1)
    goal = np.asarray(root["goal"], dtype=np.float64).reshape(-1)
    if state.shape != (7,) or goal.shape != (7,):
        raise ValueError("PushT root state and goal must each have shape (7,)")
    return hashlib.sha256(np.ascontiguousarray(np.concatenate((state, goal))).tobytes()).hexdigest()
