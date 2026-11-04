"""Shared-bank, budget-bounded full-action multi-start CEM baseline."""
import numpy as np

from experiments.spectral_reachability import pc_recursive_response_v13 as v13
from experiments.spectral_reachability import pc_response_temporal_search_v1 as temporal


def plan(predict, config, *, seed, starts=4, carry=None):
    horizon = config.segment_length * config.segments
    if (config.action_dim != 2 or config.model_packet_length != 5 or horizon % 5
            or starts < 1 or config.token_cap < 1):
        raise ValueError("invalid multistart configuration")
    bank = v13.proposal_bank(config, carry)
    _, scores = predict(bank)
    scores = np.asarray(scores, np.float64)
    if scores.shape != (len(bank),) or not np.isfinite(scores).all() or starts > len(bank):
        raise ValueError("invalid bank scores or starts")
    packets = horizon // 5
    count = (config.token_cap - len(bank) * packets - 2 * packets) // packets
    if count < starts:
        raise ValueError("insufficient multistart budget")
    counts = [count // starts + int(i < count % starts) for i in range(starts)]
    indices = np.argsort(scores, kind="stable")[:starts].tolist()
    best_chain, best_score, results = None, float("inf"), []
    for i, (index, samples) in enumerate(zip(indices, counts, strict=True)):
        chain, score, generations = temporal.cem(bank[index], [], samples, np.random.default_rng(seed + i),
                                                 predict, float(scores[index]), horizon)
        results.append({"index": index, "samples": samples, "score": float(score), "generations": generations})
        if score < best_score:
            best_chain, best_score = np.asarray(chain, np.float32), float(score)
    zero = best_chain.copy()
    zero[:config.segment_length] = 0
    pair = np.stack((best_chain, zero))
    _, pair_scores = predict(pair)
    pair_scores = np.asarray(pair_scores, np.float64)
    choice = int(np.argmin(pair_scores))
    return {"selected": pair[choice].tolist(), "selected_score": float(pair_scores[choice]),
            "bank_indices": indices, "bank_scores": scores.tolist(), "starts": results,
            "pair_scores": pair_scores.tolist(), "pair_choice": choice,
            "prediction_tokens": (len(bank) + count + 2) * packets}
