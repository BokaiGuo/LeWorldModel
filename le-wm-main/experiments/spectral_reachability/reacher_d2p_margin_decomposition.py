"""Exact evaluator-only decomposition of candidate/zero RGB cost misranking.

No fitted coefficients or selector. The oracle anchor uses a future zero-action
successor and MUST NOT be used as a deployment prediction or P1 admission.
"""
from __future__ import annotations

import torch


def decompose_margin(predicted, predicted_zero, observed, observed_zero, goal):
    values = (predicted, predicted_zero, observed, observed_zero, goal)
    if any(value.ndim != 1 or value.shape != goal.shape or value.device != goal.device
           or not torch.isfinite(value).all() for value in values):
        raise ValueError("finite matching latent vectors on one device required")
    # Float64 arithmetic isolates the algebraic identity from float32 score
    # subtraction. This never changes the model forward or its planner cost.
    p, pz, o, oz, g = [value.detach().to(torch.float64) for value in values]
    dp, do = p - pz, o - oz
    predicted_margin = (p - g).square().mean() - (pz - g).square().mean()
    observed_margin = (o - g).square().mean() - (oz - g).square().mean()
    baseline_bias = 2 * ((oz - pz) * dp).mean()
    response_linear = 2 * ((oz - g) * (do - dp)).mean()
    response_quadratic = do.square().mean() - dp.square().mean()
    error = observed_margin - predicted_margin
    cosine = torch.nn.functional.cosine_similarity(dp, do, dim=0, eps=1e-12)
    return {"predicted_margin": float(predicted_margin), "observed_margin": float(observed_margin),
        "margin_error": float(error), "zero_successor_bias_contribution": float(baseline_bias),
        "response_linear_error_contribution": float(response_linear),
        "response_quadratic_error_contribution": float(response_quadratic),
        "decomposition_abs_error": float((error - baseline_bias - response_linear - response_quadratic).abs()),
        "oracle_zero_anchored_margin": float(predicted_margin + baseline_bias),
        "predicted_action_response_l2": float(dp.norm()), "observed_action_response_l2": float(do.norm()),
        "action_response_error_l2": float((do - dp).norm()), "action_response_cosine": float(cosine),
        "oracle_future_zero_successor_used": True,
        "claim_boundary": "Evaluator-only algebra; future zero successor is unavailable to a gold-free deployed planner."}
