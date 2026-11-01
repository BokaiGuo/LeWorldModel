"""One-step, parameter-free innovation from an already observed transition.

This is an explicit diagnostic alternative to the raw frozen predictor. It
does not select actions, fit a readout, use future successors, or alter model
parameters. The full predecessor RGB/action window must be retained; a
three-frame root-anchored window alone cannot recover its own predecessor.
"""
from __future__ import annotations

import torch

from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0

RULE = "unit_additive_last_observed_transition_innovation_v1"


def innovation_from_executed_history(model, predecessor_pixels, executed_history, observed_root_latent):
    if predecessor_pixels.ndim != 5 or executed_history.ndim != 3:
        raise ValueError("predecessor RGB/history ranks differ")
    if predecessor_pixels.shape[0] != 1 or executed_history.shape[0] != 1:
        raise ValueError("innovation requires exactly one observed root")
    if predecessor_pixels.shape[1] < 2 or predecessor_pixels.shape[1] != executed_history.shape[1]:
        raise ValueError("full causal predecessor RGB/action window required")
    if observed_root_latent.ndim != 1 or observed_root_latent.device != predecessor_pixels.device or executed_history.device != predecessor_pixels.device:
        raise ValueError("observed root and predecessor device/shape differ")
    with torch.no_grad():
        embeddings = p0._latent_context(model, predecessor_pixels, executed_history)
        # These are the three actions FROM the predecessor images, including
        # the last executed action reaching the observed root. Do not apply
        # the current-root candidate recurrence's history[:, 1:] shift here.
        predicted_root = model.predict(embeddings, model.action_encoder(executed_history))[0, -1]
        if predicted_root.shape != observed_root_latent.shape:
            raise ValueError("predicted and observed root latent shapes differ")
        innovation = observed_root_latent.detach() - predicted_root.detach()
    if not torch.isfinite(innovation).all():
        raise ValueError("nonfinite causal innovation")
    return innovation, {"rule": RULE, "coefficient": 1., "future_labels_used": False,
        "historical_predict_calls": 1, "historical_predict_samples": 1,
        "innovation_l2": float(torch.linalg.vector_norm(innovation)),
        "innovation_mse": float(innovation.square().mean()),
        "claim_boundary": "Temporary observation-based diagnostic correction, not raw-model fidelity or formal P1 admission."}


def corrected_prediction(prediction, innovation):
    if prediction.shape != innovation.shape or prediction.device != innovation.device:
        raise ValueError("prediction/innovation shape or device differs")
    if not torch.isfinite(prediction).all() or not torch.isfinite(innovation).all():
        raise ValueError("nonfinite prediction or innovation")
    return prediction + innovation
