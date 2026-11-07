"""Bounded line-search candidates along a goal gradient in physical action units."""
import numpy as np


def make_candidates(raw, gradient):
    raw = np.asarray(raw, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    if (raw.ndim != 2 or gradient.shape != raw.shape or not np.isfinite(raw).all()
            or not np.isfinite(gradient).all() or np.abs(raw).max() > 0.7000001):
        raise ValueError("invalid gradient or nominal action")
    size = float(np.max(np.abs(gradient)))
    if size <= 1e-12:
        raise ValueError("degenerate gradient")
    direction = -gradient / size
    actions = np.stack([raw, raw + .15 * direction, raw - .15 * direction,
                        raw + .30 * direction, raw - .30 * direction]).astype(np.float32)
    if not np.isfinite(actions).all() or np.abs(actions).max() > 1.0000001:
        raise ValueError("action bound violation")
    return actions, direction
