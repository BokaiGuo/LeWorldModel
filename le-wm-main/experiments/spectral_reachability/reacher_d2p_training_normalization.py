"""Explicit normalization contract for diagnostic dynamics-only repair training.

Candidate, neutral and source-transition forwards have different batch
distributions. Updating one shared prediction-head BN running state across
them silently changes the singleton deployment chart. Keep base statistics
fixed unless reproducing the explicitly named legacy training protocol.
"""
from __future__ import annotations

import torch

BN_POLICIES = ("frozen-base-statistics", "legacy-domain-updates")


def dynamics_mode(model, training, policy="frozen-base-statistics"):
    if policy not in BN_POLICIES:
        raise ValueError("explicit supported BatchNorm policy required")
    for module in (model.encoder, model.projector):
        module.eval()
    for module in (model.predictor, model.action_encoder, model.pred_proj):
        module.train(training)
        if policy == "frozen-base-statistics":
            for child in module.modules():
                if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
                    if not child.track_running_stats or child.running_mean is None or child.running_var is None:
                        raise ValueError("repair BatchNorm requires a fixed deployment running state")
                    child.eval()


def normalization_snapshot(model):
    return {k: v.detach().clone() for k, v in model.named_buffers()
        if k.endswith(("running_mean", "running_var", "num_batches_tracked"))}


def verify_normalization(model, expected, policy="frozen-base-statistics"):
    if policy not in BN_POLICIES:
        raise ValueError("unsupported BatchNorm policy")
    actual = normalization_snapshot(model)
    if set(actual) != set(expected):
        raise RuntimeError("normalization buffer layout changed")
    unchanged = all(torch.equal(actual[k], expected[k]) for k in actual)
    if policy == "frozen-base-statistics" and not unchanged:
        raise RuntimeError("frozen base normalization statistics drifted during repair")
    return {"batchnorm_policy": policy, "base_normalization_buffers_unchanged": unchanged,
        "normalization_buffer_names": sorted(expected),
        "scope": "Diagnostic repaired checkpoint only; new CUDA P0/P1 admission required, not training-free mainline."}
