"""Development portfolio: v13 and a bounded full-action CEM share a cap."""
from dataclasses import replace

import numpy as np

from experiments.spectral_reachability import pc_recursive_response_v13 as v13
from experiments.spectral_reachability import pc_response_temporal_search_v1 as temporal


def compile_program(predict, target, config, carry=None):
    """Compile on the fixed 15-step, two-action development contract."""
    if (config.action_dim != 2 or config.segment_length != 5 or config.segments != 3
            or config.model_packet_length != 5 or config.token_cap != 2409):
        raise ValueError("v15 development contract requires 15 steps and 2409 tokens")
    child = v13.compile_program(predict, target, replace(config, token_cap=1920), carry=carry)
    bank = child["global_proposals"]
    base = np.asarray(bank["actions"][bank["choice"]], dtype=np.float32)
    chain, score, generations = temporal.cem(
        base, [], 160, np.random.default_rng(config.seed + 6100), predict,
        float(bank["scores"][bank["choice"]]), 15,
    )
    comparison = np.asarray(chain, np.float32).copy()
    comparison[:5] = 0
    candidates = np.stack((np.asarray(child["selected"], np.float32),
                           np.asarray(chain, np.float32), comparison))
    features, scores = predict(candidates)
    features, scores = np.asarray(features), np.asarray(scores, np.float64)
    if (features.shape != (3, np.asarray(target).size) or scores.shape != (3,)
            or not np.isfinite(features).all() or not np.isfinite(scores).all()):
        raise ValueError("invalid final comparison")
    choice = int(np.argmin(scores))
    used = child["resource"]["prediction_tokens"] + 480 + 9
    if used > config.token_cap:
        raise ValueError("budget exceeded")
    return {"method": "recursive_response_v15_portfolio_development",
            "child": child, "cem_generations": generations, "cem_score": float(score),
            "candidates": candidates.tolist(), "scores": scores.tolist(), "choice": choice,
            "selected": candidates[choice].tolist(), "selected_score": float(scores[choice]),
            "resource": {"prediction_tokens": used, "token_cap": config.token_cap,
                         "svd_calls": child["resource"]["svd_calls"],
                         "svd_seconds": child["resource"]["svd_seconds"]}}
