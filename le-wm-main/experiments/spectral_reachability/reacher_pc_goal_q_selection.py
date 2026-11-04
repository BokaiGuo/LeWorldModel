"""Deterministic Q-score selector for a fixed candidate bank."""
from __future__ import annotations

import math


def select(rows, *, margin=1e-6):
    if len(rows) < 2 or margin < 0 or [r["proposal_index"] for r in rows] != list(range(len(rows))):
        raise ValueError("invalid ordered candidate rows")
    for row in rows:
        if not all(math.isfinite(float(row[k])) for k in ("immediate_goal_cost", "cem_future_goal_cost")):
            raise ValueError("nonfinite candidate costs")
    zero = rows[0]
    eligible = [r for r in rows[1:]
                if r["immediate_goal_cost"] >= zero["immediate_goal_cost"] + margin
                and r["cem_future_goal_cost"] <= zero["cem_future_goal_cost"] - margin]
    winner = min(eligible, key=lambda r: (r["cem_future_goal_cost"], r["proposal_index"])) if eligible else None
    return None if winner is None else winner["proposal_index"], len(eligible)
