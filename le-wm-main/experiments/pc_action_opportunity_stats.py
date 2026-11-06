"""Statistics shared by action-opportunity descriptor diagnostics.

The historical descriptor panels used ordinal ranks with a stable tie break.
This module deliberately uses average ranks so repeated scores do not acquire
an arbitrary order-dependent advantage.
"""
from __future__ import annotations

import numpy as np


def average_rank(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, matching the usual Spearman definition."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks_sorted = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks_sorted[start:end] = 0.5 * (start + 1 + end)
        start = end
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Compute tie-corrected Spearman correlation or None if undefined."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 3:
        return None
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    xr, yr = average_rank(x), average_rank(y)
    if np.std(xr) == 0.0 or np.std(yr) == 0.0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def partial_rank_correlation(
    x: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
) -> float | None:
    """Partial Spearman via rank residuals against explicit controls.

    Controls are ranked column-wise before residualization. A constant column
    is harmless because the intercept is always included. This is a diagnostic
    quantity, not a substitute for a pre-registered task-level analysis.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim == 1:
        controls = controls[:, None]
    if x.size != y.size or controls.shape[0] != x.size or x.size < 3:
        return None
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    if controls.size and not np.all(np.isfinite(controls)):
        return None
    design = [np.ones(x.size, dtype=np.float64)]
    for column in range(controls.shape[1]):
        design.append(average_rank(controls[:, column]))
    design_matrix = np.column_stack(design)
    x_residual = average_rank(x) - design_matrix @ np.linalg.lstsq(
        design_matrix, average_rank(x), rcond=None
    )[0]
    y_residual = average_rank(y) - design_matrix @ np.linalg.lstsq(
        design_matrix, average_rank(y), rcond=None
    )[0]
    if np.std(x_residual) == 0.0 or np.std(y_residual) == 0.0:
        return None
    return float(np.corrcoef(x_residual, y_residual)[0, 1])
