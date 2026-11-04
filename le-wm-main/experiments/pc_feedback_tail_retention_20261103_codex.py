"""Budgeted model-score comparison of an old tail and a new feedback plan.

The caller supplies one scorer already bound to the refreshed observation,
action history, target and shared meter. This module does not execute actions
or certify that a model-score improvement translates to physical improvement.
"""
import numpy as np

from experiments.pc_budget_allocation_20261103_codex import optimize


def select_feedback_tail(old_tail, *, total_samples, rng, predict):
    """Charge one old-tail score and give the remaining samples to CEM.

    ``predict`` receives a batch of equally long action sequences and returns
    one finite scalar cost per sequence. The caller must account for the cost
    at its actual prediction entry point and call this for every comparison arm.
    """
    tail = np.asarray(old_tail, dtype=float)
    if (tail.ndim != 2 or tail.shape[1] != 2 or len(tail) == 0
            or not np.isfinite(tail).all() or np.any(np.abs(tail) > 1)
            or not isinstance(total_samples, int) or total_samples < 2):
        raise ValueError("invalid tail or feedback budget")
    old_score = np.asarray(predict(tail[None]), dtype=float)
    if old_score.shape != (1,) or not np.isfinite(old_score).all():
        raise ValueError("invalid old-tail score")
    candidate, score, scores = optimize(
        [], len(tail), total_samples - 1, rng, predict)
    use_old = float(old_score[0]) <= score
    return {
        "tail": (tail if use_old else candidate).tolist(),
        "source": "old" if use_old else "new",
        "old_score": float(old_score[0]),
        "new_score": float(score),
        "new_tail": candidate.tolist(),
        "new_scores": scores,
        "old_tail_samples": 1,
        "new_search_samples": total_samples - 1,
    }
