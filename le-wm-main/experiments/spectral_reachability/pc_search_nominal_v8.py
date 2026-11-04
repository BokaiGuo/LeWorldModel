"""Budgeted full-chain warm search followed by v7 response search."""
import numpy as np

from experiments.spectral_reachability import pc_target_reach_action_dim_v6 as v6
from experiments.spectral_reachability import pc_target_reach_paired_abstention_v7 as v7

METHOD = "search_induced_nominal_dev_v8"
WARM_SAMPLES = 48


class _RemainingMeter:
    def __init__(self, meter, initial, reserve):
        self.meter, self.initial = meter, initial
        self.token_cap = min(v6.CAP, meter.token_cap) - initial - reserve
        if self.token_cap <= 0:
            raise ValueError("no budget remains for v7")

    @property
    def token_samples(self):
        return self.meter.token_samples - self.initial


def decide(strategy, arm, root, step, target, predict, meter, nominals, carry=None):
    nominals = np.asarray(nominals, dtype=float)
    if (strategy != "response" or arm != "response" or step not in (0, 1, 2)
            or meter.token_samples or nominals.ndim != 3 or nominals.shape[:2] != (2, 5)
            or not np.isfinite(nominals).all() or np.any(np.abs(nominals) > .95)):
        raise ValueError("invalid v8 inputs")
    dimension = nominals.shape[2]
    if dimension < 1:
        raise ValueError("action dimension required")
    horizon, per = 15 - 5*step, 3 - step
    bases = [np.zeros((horizon, dimension))]
    if carry is not None:
        carry = np.asarray(carry, dtype=float)
        if step == 0 or carry.shape != bases[0].shape or not np.isfinite(carry).all() or np.any(np.abs(carry) > 1):
            raise ValueError("invalid carry")
        bases.append(carry)
    elif step != 0:
        raise ValueError("missing carry")
    _, base_scores = predict(np.asarray(bases))
    base_scores = np.asarray(base_scores, dtype=float)
    if base_scores.shape != (len(bases),) or not np.isfinite(base_scores).all():
        raise ValueError("invalid base scores")
    base_index = int(np.argmin(base_scores))
    seed = 2027113000 + root*100 + step*10000
    warm_chain, warm_score, generations = v6.cem(
        bases[base_index], [], WARM_SAMPLES, np.random.default_rng(seed), predict,
        float(base_scores[base_index]), 4)
    induced = np.stack((warm_chain[:5], warm_chain[5:10] if horizon >= 10 else nominals[1]))
    induced = np.clip(induced, -.95, .95)
    initial = meter.token_samples
    inner = _RemainingMeter(meter, initial, 3*per)
    compiled = v7.decide(strategy, arm, root, step, target, predict, inner, induced, carry)
    zero_warm = warm_chain.copy()
    zero_warm[:5] = 0
    chains = np.stack((zero_warm, np.asarray(compiled["selected"]["chain"]), warm_chain))
    _, values = predict(chains)
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or meter.token_samples > min(v6.CAP, meter.token_cap):
        raise ValueError("invalid final comparison")
    choice = int(np.argmin(values))
    return {"method": METHOD, "selected": {"name": ("zero_warm", "v7", "warm")[choice],
                                            "chain": chains[choice].tolist(),
                                            "terminal_score": float(values[choice])},
            "warm": {"base_index": base_index, "base_scores": base_scores.tolist(),
                     "chain": warm_chain.tolist(), "terminal_score": warm_score,
                     "samples": WARM_SAMPLES, "rng_seed": seed, "generations": generations,
                     "tokens_before_v7": initial},
            "induced_nominals": induced.tolist(), "v7": compiled,
            "final_comparison": {"chains": chains.tolist(), "scores": values.tolist(),
                                 "choice": choice, "reserved_tokens": 3*per}}
