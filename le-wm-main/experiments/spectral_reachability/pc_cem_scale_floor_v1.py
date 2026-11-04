"""Bounded iterative CEM with an explicit search-scale floor."""
import numpy as np


def optimize(prefix, horizon, count, rng, predict, scale_floor=0.05):
    prefix = np.asarray(prefix, dtype=float).reshape(-1, 2)
    remaining = horizon - len(prefix)
    if remaining <= 0 or not isinstance(count, int) or count < 1:
        raise ValueError("invalid optimizer allocation")
    if not np.isfinite(prefix).all() or np.any(np.abs(prefix) > 1):
        raise ValueError("invalid prefix")
    if not np.isfinite(scale_floor) or not 0 < scale_floor <= 1:
        raise ValueError("invalid scale floor")
    loc = np.zeros((remaining, 2))
    scale = np.full((remaining, 2), 1 / 3)
    best, best_score, scores = None, float("inf"), []
    for start in range(0, count, 12):
        size = min(12, count - start)
        samples = np.clip(loc + rng.normal(size=(size, remaining, 2)) * scale, -1, 1)
        samples[0] = loc
        chains = np.concatenate((np.broadcast_to(prefix, (size, len(prefix), 2)), samples), axis=1)
        values = np.asarray(predict(chains), dtype=float)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError("invalid prediction scores")
        for chain, value in zip(chains, values, strict=True):
            scores.append(float(value))
            if value < best_score:
                best, best_score = chain.copy(), float(value)
        if size > 1:
            elite = samples[np.argsort(values, kind="stable")[:min(4, size)]]
            loc = elite.mean(0)
            scale = np.maximum(elite.std(0, ddof=0), scale_floor)
    return best, best_score, scores
