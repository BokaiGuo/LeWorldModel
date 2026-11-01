"""Evaluator-only joint distance respecting the actual joint topology."""
import numpy as np

CONTRACT = "periodic-shoulder-bounded-wrist-v1"


def distance(current, target, limited):
    current, target = (np.asarray(v, np.float64) for v in (current, target))
    if (current.shape != (2,) or target.shape != (2,)
            or not np.isfinite(current).all() or not np.isfinite(target).all()
            or len(limited) != 2 or any(type(v) is not bool for v in limited)):
        raise ValueError("finite two-joint states and explicit boolean joint topology required")
    delta = current-target
    for j, bounded in enumerate(limited):
        if not bounded:
            delta[j] = (delta[j]+np.pi) % (2*np.pi)-np.pi
    return float(np.linalg.norm(delta))
