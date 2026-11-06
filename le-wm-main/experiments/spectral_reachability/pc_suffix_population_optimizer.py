"""Bounded full-action suffix CEM with a fixed global-exploration fraction.

The caller owns candidate generation, the initial score evaluation, and
resource accounting. This optimizer never reads task state or privileged
information and spends exactly ``samples`` predictor evaluations.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    samples: int
    batch_size: int = 12
    elites: int = 4
    global_fraction: float = 0.25
    initial_scale: float = 0.35
    min_scale: float = 0.07


def optimize(predict, prefix, initial_suffix, initial_score, config, *, seed):
    """Optimize a legal suffix while preserving its supplied incumbent."""
    prefix = np.asarray(prefix, dtype=np.float32)
    suffix = np.asarray(initial_suffix, dtype=np.float32)
    if (prefix.ndim != 2 or suffix.ndim != 2 or not len(suffix)
            or prefix.shape[1] != suffix.shape[1] or not suffix.shape[1]
            or not np.isfinite(prefix).all() or not np.isfinite(suffix).all()
            or np.any(np.abs(prefix) > 1) or np.any(np.abs(suffix) > 1)
            or not np.isfinite(initial_score)):
        raise ValueError("invalid prefix, suffix or initial score")
    if (config.samples < 1 or config.batch_size < 2 or not 1 <= config.elites <= config.batch_size
            or not np.isfinite(config.global_fraction) or not 0 <= config.global_fraction < 1
            or not np.isfinite(config.initial_scale) or config.initial_scale <= 0
            or not np.isfinite(config.min_scale) or config.min_scale <= 0):
        raise ValueError("invalid optimizer configuration")
    rng = np.random.default_rng(seed)
    loc, scale = suffix.astype(np.float64), np.full(suffix.shape, config.initial_scale)
    best_suffix, best_score = suffix.copy(), float(initial_score)
    history = []
    for offset in range(0, config.samples, config.batch_size):
        size = min(config.batch_size, config.samples - offset)
        global_count = min(size - 1, int(np.floor(size * config.global_fraction)))
        local = size - global_count
        proposals = np.empty((size, *suffix.shape), dtype=np.float64)
        proposals[:local] = loc + rng.normal(size=(local, *suffix.shape)) * scale
        if global_count:
            proposals[local:] = rng.uniform(-1., 1., size=(global_count, *suffix.shape))
        proposals[0] = best_suffix
        proposals = np.clip(proposals, -1., 1.).astype(np.float32)
        chains = np.concatenate((np.broadcast_to(prefix, (size, *prefix.shape)), proposals), axis=1)
        result = predict(chains.copy())
        scores = np.asarray(result[1], dtype=np.float64)
        if scores.shape != (size,) or not np.isfinite(scores).all():
            raise ValueError("invalid predictor scores")
        index = int(np.argmin(scores))
        if scores[index] < best_score:
            best_suffix, best_score = proposals[index].copy(), float(scores[index])
        elite = proposals[np.argsort(scores, kind="stable")[:min(size, config.elites)]].astype(np.float64)
        before_loc, before_scale = loc, scale
        loc, scale = elite.mean(axis=0), np.maximum(elite.std(axis=0), config.min_scale)
        history.append({"size": size, "global_count": global_count, "mean_before": before_loc.tolist(),
                        "scale_before": before_scale.tolist(), "best_score": best_score})
    return {"chain": np.concatenate((prefix, best_suffix)).tolist(), "score": best_score,
            "samples": config.samples, "generations": history}
