"""Shared-bank baseline using the same CEM implementation as v23 refinement."""
import numpy as np

from experiments.spectral_reachability import pc_recursive_response_v13 as proposals
from experiments.spectral_reachability import pc_shared_multistart_cem as cem


def plan(predict, config, *, seed, carry=None):
    horizon = config.segment_length * config.segments
    if (config.model_packet_length < 1 or horizon % config.model_packet_length
            or config.refine_starts < 1):
        raise ValueError("invalid horizon or number of starts")
    packets = horizon // config.model_packet_length
    bank = proposals.proposal_bank(config, carry)
    _, scores = predict(bank.copy())
    scores = np.asarray(scores, np.float64)
    if scores.shape != (len(bank),) or not np.isfinite(scores).all():
        raise ValueError("invalid bank scores")
    sample_budget = config.token_cap // packets - len(bank) - 2
    indices = np.argsort(scores, kind="stable")[:config.refine_starts]
    if len(indices) != config.refine_starts:
        raise ValueError("insufficient bank seeds")

    def evaluate(actions):
        return predict(actions)[1]

    optimized = cem.optimize(evaluate, np.empty((0, config.action_dim), np.float32),
                             bank[indices], sample_budget, seed=seed,
                             batch_size=config.refine_batch, elites=config.refine_elites)
    incumbent = optimized["actions"]
    comparison = incumbent.copy()
    comparison[:config.segment_length] = 0
    pair = np.stack((incumbent, comparison))
    _, pair_scores = predict(pair.copy())
    pair_scores = np.asarray(pair_scores, np.float64)
    if pair_scores.shape != (2,) or not np.isfinite(pair_scores).all():
        raise ValueError("invalid paired scores")
    choice = int(np.argmin(pair_scores))
    return {"selected": pair[choice].tolist(), "selected_score": float(pair_scores[choice]),
            "incumbent": incumbent.tolist(), "comparison": comparison.tolist(),
            "paired_scores": pair_scores.tolist(), "abstained": bool(choice),
            "bank_indices": indices.tolist(), "bank_scores": scores.tolist(),
            "starts": optimized["starts"],
            "prediction_tokens": (len(bank) + sample_budget + 2) * packets}
