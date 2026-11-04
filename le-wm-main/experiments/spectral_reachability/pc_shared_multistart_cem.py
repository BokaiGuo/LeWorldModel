"""Budgeted CEM shared by compiler refinement and prefix controls.

The callback receives complete bounded float32 action chains. Every seed and
every generated sample is evaluated and charged to the caller's own meter.
This module does not inspect environment state or supply privileged seeds.
"""

import numpy as np


def optimize(evaluate, prefix, initial_tails, sample_budget, *, seed=0,
             batch_size=12, elites=4, initial_scale=0.35, min_scale=0.07):
    prefix = np.asarray(prefix, dtype=np.float32)
    tails = np.asarray(initial_tails, dtype=np.float32)
    if (prefix.ndim != 2 or tails.ndim != 3 or not len(tails)
            or not tails.shape[1] or prefix.shape[1] != tails.shape[2]
            or not np.isfinite(prefix).all() or not np.isfinite(tails).all()
            or np.any(np.abs(prefix) > 1) or np.any(np.abs(tails) > 1)):
        raise ValueError("invalid prefix or initial tails")
    if (not isinstance(sample_budget, (int, np.integer))
            or not isinstance(batch_size, (int, np.integer))
            or not isinstance(elites, (int, np.integer))
            or batch_size < 2 or not 1 <= elites <= batch_size
            or sample_budget < len(tails) * (1 + 2 * batch_size)
            or not np.isfinite(initial_scale) or initial_scale <= 0
            or not np.isfinite(min_scale) or min_scale <= 0):
        raise ValueError("invalid CEM configuration or insufficient budget")

    def score(samples):
        samples = np.asarray(samples, dtype=np.float32)
        chains = np.concatenate(
            (np.broadcast_to(prefix, (len(samples), *prefix.shape)), samples), axis=1)
        scores = np.asarray(evaluate(chains.copy()), dtype=np.float64)
        if scores.shape != (len(samples),) or not np.isfinite(scores).all():
            raise ValueError("invalid scores")
        return scores

    seed_scores = score(tails)
    left = sample_budget - len(tails)
    allocations = [left // len(tails) + (i < left % len(tails))
                   for i in range(len(tails))]
    best_index = int(np.argmin(seed_scores))
    best, best_score = tails[best_index].copy(), float(seed_scores[best_index])
    starts = []
    for index, allocation in enumerate(allocations):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), index]))
        local, local_score = tails[index].copy(), float(seed_scores[index])
        mean = local.astype(np.float64)
        scale = np.full(mean.shape, initial_scale, dtype=np.float64)
        generations = []
        remaining = int(allocation)
        while remaining:
            size = min(batch_size, remaining)
            samples = np.clip(mean + rng.normal(size=(size, *mean.shape)) * scale,
                              -1, 1).astype(np.float32)
            samples[0] = local
            scores = score(samples)
            winner = int(np.argmin(scores))
            if scores[winner] < local_score:
                local, local_score = samples[winner].copy(), float(scores[winner])
            if local_score < best_score:
                best, best_score = local.copy(), local_score
            elite = samples[np.argsort(scores, kind="stable")[:min(size, elites)]]
            mean = elite.astype(np.float64).mean(axis=0)
            scale = np.maximum(elite.astype(np.float64).std(axis=0), min_scale)
            generations.append({"samples": size, "best_score": local_score})
            remaining -= size
        starts.append({"seed_score": float(seed_scores[index]), "samples": int(allocation) + 1,
                       "best_score": local_score, "generations": generations})
    return {"actions": np.concatenate((prefix, best)), "score": best_score,
            "samples": int(sample_budget), "starts": starts}
