"""Feedback v2: preserve all scored full-horizon probes, then recheck abstention."""
import numpy as np

from experiments.spectral_reachability import pc_response_temporal_feedback_v1 as prior
from experiments.spectral_reachability import pc_response_temporal_search_v1 as initial

METHOD = "response_temporal_fixed_terminal_dev_v2"


class _ReservedMeter:
    """Expose a reserved terminal comparison to v1 without charging fake calls."""

    def __init__(self, meter, reserve):
        self.meter, self.reserve = meter, reserve

    @property
    def token_samples(self):
        used = self.meter.token_samples
        return used + self.reserve if used else 0


def decide(arm, root, step, target, predict, meter, carry=None):
    if step == 0 or arm in ("smooth_cem", "action_cem"):
        return prior.decide(arm, root, step, target, predict, meter, carry)
    if arm not in ("response", "matched_random") or step not in (1, 2) or meter.token_samples:
        raise ValueError("invalid arm, step or meter")
    horizon, per = 15 - 5 * step, (15 - 5 * step) // 5
    best = {"chain": None, "score": float("inf"), "full_samples": 0}

    def tracked(actions):
        actions = np.asarray(actions, dtype=float)
        positions, scores = predict(actions)
        scores = np.asarray(scores)
        if actions.ndim != 3 or scores.shape != (len(actions),) or not np.isfinite(scores).all():
            raise ValueError("invalid model output")
        if actions.shape[1] == horizon:
            best["full_samples"] += len(actions)
            index = int(np.argmin(scores))
            if float(scores[index]) < best["score"]:
                best["chain"], best["score"] = actions[index].copy(), float(scores[index])
        return positions, scores

    proxy = _ReservedMeter(meter, per)
    old = prior.decide(arm, root, step, target, tracked, proxy, carry)
    if best["chain"] is None or meter.token_samples + per > initial.CAP:
        raise ValueError("missing scored incumbent or terminal comparison budget")
    chain, score = best["chain"].copy(), best["score"]
    comparator = chain.copy()
    comparator[:5] = 0
    comparator_score = float(tracked(comparator[None])[1][0])
    abstained = comparator_score <= score
    selected = ({"name": "zero_prefix", "chain": comparator.tolist(), "terminal_score": comparator_score}
                if abstained else {"name": "global_scored_incumbent", "chain": chain.tolist(),
                                   "terminal_score": score})
    return {**old, "selected": selected, "abstained": bool(abstained),
            "v1_selected_before_global_check": old["selected"], "v1_abstained": old["abstained"],
            "global_incumbent_pre_abstention": {"chain": chain.tolist(), "terminal_score": score},
            "comparison_score": comparator_score, "comparison_chain": comparator.tolist(),
            "full_samples_tracked": best["full_samples"], "reserved_terminal_tokens": per}
