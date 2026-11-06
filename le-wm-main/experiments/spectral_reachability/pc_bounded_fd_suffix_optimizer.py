"""Black-box projected finite-difference suffix baseline with exact accounting."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    samples: int
    fd_step: float = 0.05
    line_steps: tuple = (0.1, 0.25, 0.5, 1.0)


def optimize(predict, prefix, initial_suffix, initial_score, config):
    """Minimize the caller's score; preserve all scored full-chain incumbents."""
    prefix = np.asarray(prefix, np.float32)
    suffix = np.asarray(initial_suffix, np.float32)
    if (prefix.ndim != 2 or suffix.ndim != 2 or not len(suffix)
            or not suffix.shape[1] or prefix.shape[1] != suffix.shape[1]
            or not np.isfinite(prefix).all() or not np.isfinite(suffix).all()
            or np.any(np.abs(prefix) > 1) or np.any(np.abs(suffix) > 1)
            or not np.isfinite(initial_score)):
        raise ValueError("invalid prefix, suffix or initial score")
    if (config.samples < 1 or not np.isfinite(config.fd_step) or config.fd_step <= 0
            or not config.line_steps or any(not np.isfinite(x) or x <= 0 for x in config.line_steps)):
        raise ValueError("invalid finite-difference configuration")
    dim = suffix.size
    axis = np.eye(dim, dtype=np.float64).reshape(dim, *suffix.shape)
    best, best_score = suffix.copy(), float(initial_score)
    used, generations = 0, []
    while used + 2 * dim <= config.samples:
        center = best.copy()
        plus = np.clip(center[None] + config.fd_step * axis, -1., 1.).astype(np.float32)
        minus = np.clip(center[None] - config.fd_step * axis, -1., 1.).astype(np.float32)
        probes = np.concatenate((plus, minus))
        chains = np.concatenate((np.broadcast_to(prefix, (len(probes), *prefix.shape)), probes), axis=1)
        scores = np.asarray(predict(chains.copy())[1], np.float64)
        if scores.shape != (len(probes),) or not np.isfinite(scores).all():
            raise ValueError("invalid predictor scores")
        used += len(probes)
        denominator = plus.reshape(dim, dim)[np.arange(dim), np.arange(dim)] - minus.reshape(dim, dim)[np.arange(dim), np.arange(dim)]
        if np.any(denominator <= 0):
            raise ValueError("finite difference collapsed at action bounds")
        gradient = (scores[:dim] - scores[dim:]) / denominator.astype(np.float64)
        index = int(np.argmin(scores))
        if scores[index] < best_score:
            best, best_score = probes[index].copy(), float(scores[index])
        scale = float(np.max(np.abs(gradient)))
        remaining = config.samples - used
        steps = config.line_steps[:min(len(config.line_steps), remaining)]
        line_scores = []
        if scale > 1e-12 and steps:
            candidates = np.stack([np.clip(center - step * (gradient / scale).reshape(suffix.shape), -1., 1.)
                                   for step in steps]).astype(np.float32)
            chains = np.concatenate((np.broadcast_to(prefix, (len(candidates), *prefix.shape)), candidates), axis=1)
            values = np.asarray(predict(chains.copy())[1], np.float64)
            if values.shape != (len(candidates),) or not np.isfinite(values).all():
                raise ValueError("invalid line-search scores")
            used += len(candidates)
            line_scores = values.tolist()
            index = int(np.argmin(values))
            if values[index] < best_score:
                best, best_score = candidates[index].copy(), float(values[index])
        generations.append({"center": center.tolist(), "fd_samples": 2 * dim, "line_samples": len(line_scores),
                            "gradient": gradient.tolist(), "line_scores": line_scores, "best_score": best_score})
        if scale <= 1e-12:
            break
    return {"chain": np.concatenate((prefix, best)).tolist(), "score": best_score,
            "samples": used, "unused_samples": config.samples - used, "generations": generations}
