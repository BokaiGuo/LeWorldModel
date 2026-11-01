"""Model-only strong-reference guard; physical outcomes never enter selection."""
from __future__ import annotations

import math
from typing import Any


FUTURE_REFERENCES = ("zero_prefix", "same_bank_greedy")


def future_reference_row(rows: list[dict[str, Any]], reference: str) -> dict[str, Any]:
    if reference not in FUTURE_REFERENCES or not rows:
        raise ValueError("unknown future reference or empty proposal rows")
    indices = [row["proposal_index"] for row in rows]
    if len(indices) != len(set(indices)) or 0 not in indices:
        raise ValueError("future reference needs unique proposals including zero")
    if not all(math.isfinite(float(row["immediate_goal_cost"])) for row in rows):
        raise ValueError("non-finite model immediate goal cost")
    if reference == "zero_prefix":
        return next(row for row in rows if row["proposal_index"] == 0)
    return min(rows, key=lambda row: (row["immediate_goal_cost"], row["proposal_index"]))


def improves_future_reference(candidate: dict[str, Any], reference: dict[str, Any], *, score: str, margin: float) -> bool:
    """Strict benefit is required to call a new strong-reference witness useful."""
    current, baseline = float(candidate[score]), float(reference[score])
    if not math.isfinite(current) or not math.isfinite(baseline) or not math.isfinite(margin) or margin < 0:
        raise ValueError("invalid future reference score or margin")
    return current < baseline - margin
