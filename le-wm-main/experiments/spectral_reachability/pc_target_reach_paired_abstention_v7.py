"""v6 search with budgeted same-batch final abstention comparison."""

import numpy as np

from experiments.spectral_reachability import pc_target_reach_action_dim_v6 as v6

METHOD = "target_reach_paired_abstention_dev_v7"


class _ReservedMeter:
    def __init__(self, meter, reserved):
        self._meter = meter
        self.token_cap = min(v6.CAP, meter.token_cap) - reserved
        if self.token_cap <= 0:
            raise ValueError("insufficient budget for paired comparison")

    @property
    def token_samples(self):
        return self._meter.token_samples


def decide(strategy, arm, root, step, target, predict, meter, nominals, carry=None):
    if step not in (0, 1, 2):
        raise ValueError("invalid step")
    horizon = 15 - 5 * step
    reserved = 2 * (horizon // 5)
    proxy = _ReservedMeter(meter, reserved)
    result = v6.decide(strategy, arm, root, step, target, predict, proxy, nominals, carry)
    first = np.asarray(result["precomparison"]["chain"], dtype=np.float64)
    zero = np.asarray(result["comparison_chain"], dtype=np.float64)
    if first.shape != zero.shape or first.ndim != 2 or first.shape[0] != horizon:
        raise ValueError("invalid comparison chains")
    _, scores = predict(np.stack((first, zero)))
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (2,) or not np.isfinite(scores).all():
        raise ValueError("invalid paired scores")
    identical = np.array_equal(first.astype(np.float32), zero.astype(np.float32))
    abstained = bool(identical or scores[1] <= scores[0])
    result["v6_selection_before_paired_comparison"] = result["selected"]
    result["v6_abstained_before_paired_comparison"] = result["abstained"]
    result["paired_comparison"] = {
        "incumbent_score": float(scores[0]), "zero_prefix_score": float(scores[1]),
        "same_actual_float32_actions": bool(identical), "reserved_token_samples": reserved,
    }
    result["selected"] = {
        "name": "zero_prefix" if abstained else "global_scored_incumbent",
        "chain": (zero if abstained else first).tolist(),
        "terminal_score": float(scores[1] if abstained else scores[0]),
    }
    result["abstained"] = abstained
    return result
