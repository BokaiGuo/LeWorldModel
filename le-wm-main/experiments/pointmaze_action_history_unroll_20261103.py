"""Unroll a frozen JEPA predictor with explicitly bound historical actions.

The caller supplies already encoded observations and normalized action packets.
This function performs no privileged-state access and does not alter weights.
"""
from __future__ import annotations

import torch


def unroll_with_action_history(model, context, history, future, window):
    """Return time-major visual/proprio tensors including the observed prefix.

    context tensors are [1, context_time, ...]; history is normalized
    [context_time-1, batch, action_dim]; future is [horizon, batch, action_dim].
    Each action at index t leaves the matching observation at index t.
    """
    if set(context.keys()) != {"visual", "proprio"}:
        raise ValueError("visual and proprio contexts are required")
    visual, proprio = context["visual"], context["proprio"]
    if visual.ndim != 6 or proprio.ndim != 4 or visual.shape[:2] != proprio.shape[:2]:
        raise ValueError("invalid encoded observation dimensions")
    if visual.shape[0] != 1 or visual.shape[1] < 1:
        raise ValueError("expected one nonempty common observation history")
    if future.ndim != 3 or history.ndim != 3 or future.shape[0] < 1:
        raise ValueError("invalid action dimensions")
    if history.shape[0] != visual.shape[1] - 1:
        raise ValueError("history must contain one packet per observed transition")
    if history.shape[1:] != future.shape[1:] or future.shape[1] < 1 or future.shape[2] < 1:
        raise ValueError("action batch or packet dimensions differ")
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    if not torch.isfinite(history).all() or not torch.isfinite(future).all():
        raise ValueError("nonfinite normalized actions")
    if model.proprio_mode != "predict_proprio":
        raise ValueError("only predictor-generated proprio is supported")
    batch = future.shape[1]
    video = visual.expand(batch, *visual.shape[1:])
    prop = proprio.expand(batch, *proprio.shape[1:])
    for step in range(future.shape[0]):
        # Current observation and its outgoing candidate occupy the final slot.
        actions = torch.cat((history, future[:step + 1]), dim=0)
        n = min(window, video.shape[1])
        if actions.shape[0] != video.shape[1]:
            raise RuntimeError("observation/action alignment lost")
        act = actions[-n:].transpose(0, 1)
        prediction, _, prop_prediction = model.model.forward_pred(
            video[:, -n:], model.model.encode_act(act), prop[:, -n:])
        if prediction.shape[:2] != (batch, n) or prop_prediction is None or prop_prediction.shape[:2] != (batch, n):
            raise ValueError("invalid model output shape")
        video = torch.cat((video, prediction[:, -1:]), dim=1)
        prop = torch.cat((prop, prop_prediction[:, -1:]), dim=1)
    return {"visual": video.transpose(0, 1), "proprio": prop.transpose(0, 1)}
