"""Outcome-independent trajectory allocation and bounded, partial-batch CEM."""
import numpy as np


def allocate_samples(token_cap, consumed, candidates, tokens_per_sample):
    """Allocate affordable complete samples equally, deterministically."""
    if (not all(isinstance(x, int) for x in (token_cap, consumed, candidates, tokens_per_sample))
            or token_cap < 0 or consumed < 0 or consumed > token_cap
            or candidates < 1 or tokens_per_sample < 1):
        raise ValueError("invalid allocation")
    total = (token_cap - consumed) // tokens_per_sample
    quotient, remainder = divmod(total, candidates)
    return [quotient + (i < remainder) for i in range(candidates)]


def optimize(prefix, horizon, count, rng, predict):
    """CEM with the same 12-sample maximum and a smaller final batch."""
    prefix = np.asarray(prefix, dtype=float).reshape(-1, 2)
    remaining = horizon - len(prefix)
    if remaining <= 0 or not isinstance(count, int) or count < 1:
        raise ValueError("invalid optimizer allocation")
    loc = np.zeros((remaining, 2))
    scale = np.full((remaining, 2), 1 / 3)
    best, score, records = None, float("inf"), []
    for start in range(0, count, 12):
        size = min(12, count - start)
        samples = np.clip(loc + rng.normal(size=(size, remaining, 2)) * scale, -1, 1)
        samples[0] = loc
        chains = np.concatenate((np.broadcast_to(prefix, (size, len(prefix), 2)), samples), axis=1)
        values = np.asarray(predict(chains))
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError("invalid prediction scores")
        for chain, value in zip(chains, values, strict=True):
            records.append(float(value))
            if value < score:
                best, score = chain.copy(), float(value)
        # A singleton contains no information for updating dispersion.
        if size > 1:
            elite = samples[np.argsort(values, kind="stable")[:min(4, size)]]
            loc = elite.mean(0)
            scale = np.maximum(elite.std(0, ddof=0), 1e-5)
    return best, score, records
