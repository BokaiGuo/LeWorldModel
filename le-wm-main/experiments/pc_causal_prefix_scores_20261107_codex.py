"""Canonicalize numerically different scores for exactly identical action prefixes.

This is a numerical-consistency utility, not a new objective or an optimizer.
Only candidates already available to the calling planner may be passed in.
"""
import numpy as np


def canonicalize(actions, scores, *, packet_steps, max_disagreement):
    """Return copied scores with identical causal prefixes assigned one score.

    At each time the score of the lexicographically smallest complete action
    byte string is the deterministic representative. Sorting candidates does not
    affect the result. A spread exceeding the caller's declared tolerance
    raises rather than concealing a model, preprocessing, or clock mismatch.
    """
    actions = np.asarray(actions)
    scores = np.asarray(scores)
    if (actions.ndim != 3 or scores.ndim != 2 or len(actions) == 0
            or len(actions) != len(scores) or packet_steps <= 0
            or actions.shape[1] != packet_steps * scores.shape[1]
            or not np.issubdtype(actions.dtype, np.floating)
            or not np.issubdtype(scores.dtype, np.floating)
            or not np.isfinite(actions).all() or not np.isfinite(scores).all()
            or np.any(np.abs(actions) > 1) or not np.isfinite(max_disagreement)
            or max_disagreement < 0):
        raise ValueError("invalid actions, scores, time base, or tolerance")
    actions = np.ascontiguousarray(actions)
    result = scores.copy()
    full_keys = [x.tobytes() for x in actions]
    group_count, max_spread = 0, 0.0
    for step in range(scores.shape[1]):
        groups = {}
        for i in range(len(actions)):
            key = actions[i, :packet_steps * (step + 1)].tobytes()
            groups.setdefault(key, []).append(i)
        for group in groups.values():
            if len(group) < 2:
                continue
            group_count += 1
            values = scores[group, step].astype(np.float64)
            spread = float(values.max() - values.min())
            max_spread = max(max_spread, spread)
            if spread > max_disagreement:
                raise ValueError(f"shared-prefix score spread {spread} exceeds tolerance")
            representative = min(group, key=lambda i: (full_keys[i], float(scores[i, step])))
            result[group, step] = scores[representative, step]
    return result, {"duplicate_prefix_groups": group_count, "max_spread": max_spread}


def select(actions, scores, *, packet_steps, max_disagreement):
    """Minimize temporal score with action-byte tie-break after canonicalization."""
    actions = np.asarray(actions)
    normalized, audit = canonicalize(actions, scores, packet_steps=packet_steps,
                                     max_disagreement=max_disagreement)
    best_by_chain = normalized.min(axis=1)
    best = float(best_by_chain.min())
    equal = np.flatnonzero(best_by_chain == best)
    index = min(equal, key=lambda i: np.ascontiguousarray(actions[i]).tobytes())
    return int(index), best, audit
