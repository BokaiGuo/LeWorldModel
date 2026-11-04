"""Budget-split zero/carry feedback CEM baseline, separate from v3."""
import numpy as np

from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_response_temporal_warm_cem_v1 as single

METHOD = "fixed_terminal_zero_carry_two_start_cem_dev_v1"


def decide(arm, root, step, target, predict, meter, carry=None):
    if arm not in ("smooth_cem", "action_cem") or step not in (0, 1, 2) or meter.token_samples:
        raise ValueError("invalid arm, step, or nonempty meter")
    target = np.asarray(target, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("invalid target")
    if step == 0:
        return single.decide(arm, root, step, target, predict, meter, carry)
    horizon = 15 - 5 * step
    carry = np.asarray(carry, dtype=float)
    if carry.shape != (horizon, 2) or not np.isfinite(carry).all() or np.any(np.abs(carry) > 1):
        raise ValueError("invalid carry")
    seeds = np.stack((np.zeros((horizon, 2)), carry))
    _, scores = predict(seeds)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (2,) or not np.isfinite(scores).all():
        raise ValueError("invalid seed scores")
    per = horizon // 5
    available = (min(core.CAP, meter.token_cap) - meter.token_samples) // per
    counts = (available // 2, available - available // 2)
    if min(counts) < 1:
        raise ValueError("insufficient optimization budget")
    seed = 2027110400 + root * 100 + step * 10000
    best_index = int(np.argmin(scores))
    best_chain, best_score = seeds[best_index].copy(), float(scores[best_index])
    restarts = []
    for index, count in enumerate(counts):
        chain, value, generations = core.cem(
            seeds[index], [], count, np.random.default_rng(seed + 1 + index),
            predict, float(scores[index]), 4 if arm == "smooth_cem" else horizon)
        restarts.append({"name": ("zero", "carry")[index], "samples": count,
                         "score": value, "generations": generations})
        if value < best_score:
            best_chain, best_score = chain, value
    return {"selected": {"name": "two_start_cem", "chain": best_chain.tolist(), "terminal_score": best_score},
            "carry": carry.tolist(), "carry_score": float(scores[1]), "zero_score": float(scores[0]),
            "initialization": ("zero", "carry")[best_index], "restarts": restarts,
            "structure": {"svd": 0, "fd_action_samples": 0, "svd_seconds": 0.0}, "abstained": None}
