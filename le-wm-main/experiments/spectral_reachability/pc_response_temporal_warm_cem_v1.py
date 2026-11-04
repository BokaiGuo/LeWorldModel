"""Separate carry-aware CEM control for fixed-terminal v3 feedback studies."""
import numpy as np

from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_response_temporal_feedback_v3 as v3

METHOD = "fixed_terminal_carry_aware_cem_dev_v1"


def decide(arm, root, step, target, predict, meter, carry=None):
    """Score zero and carry, initialize CEM at the better, and retain incumbent."""
    if arm not in ("smooth_cem", "action_cem") or step not in (0, 1, 2) or meter.token_samples:
        raise ValueError("invalid arm, step, or nonempty meter")
    target = np.asarray(target, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("invalid target")
    if step == 0:
        return v3.decide(arm, root, step, target, predict, meter, carry)
    horizon = 15 - 5 * step
    carry = np.asarray(carry, dtype=float)
    if carry.shape != (horizon, 2) or not np.isfinite(carry).all() or np.any(np.abs(carry) > 1):
        raise ValueError("invalid carry")
    seeds = np.stack((np.zeros((horizon, 2)), carry))
    _, scores = predict(seeds)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (2,) or not np.isfinite(scores).all():
        raise ValueError("invalid seed scores")
    index = int(np.argmin(scores))
    per = horizon // 5
    count = (min(core.CAP, meter.token_cap) - meter.token_samples) // per
    if count < 1:
        raise ValueError("no optimization budget")
    seed = 2027110400 + root * 100 + step * 10000
    chain, score, generations = core.cem(seeds[index], [], count, np.random.default_rng(seed + 1),
                                         predict, float(scores[index]),
                                         4 if arm == "smooth_cem" else horizon)
    return {"selected": {"name": "carry_aware_cem", "chain": chain.tolist(), "terminal_score": score},
            "carry": carry.tolist(), "carry_score": float(scores[1]), "zero_score": float(scores[0]),
            "initialization": "zero" if index == 0 else "carry", "generations": generations,
            "structure": {"svd": 0, "fd_action_samples": 0, "svd_seconds": 0.0},
            "abstained": None}
