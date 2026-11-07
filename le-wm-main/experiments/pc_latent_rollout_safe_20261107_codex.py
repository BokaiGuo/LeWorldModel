"""Single-observation cached rollout for the installed stable-worldmodel LeWM."""

import torch
from stable_worldmodel.wm.lewm import LeWM


def rollout_single_latent(model, initial, actions, history_size=3):
    """Return (samples, horizon + 1, features), preserving action gradients."""
    if initial.ndim != 1 or actions.ndim != 3:
        raise ValueError("expected initial (features,) and actions (samples, horizon, action_dim)")
    samples, horizon, _ = actions.shape
    if samples < 1 or horizon < 1 or history_size < 1:
        raise ValueError("empty rollout or invalid history")
    if initial.device != actions.device:
        raise ValueError("latent and actions must share a device")
    if type(model) is not LeWM:
        raise TypeError("cached embedding contract is only certified for stable-worldmodel LeWM")
    # The installed LeWM caches info['emb'] and uses pixels only for H.
    # Match its native encoder path, which detaches the initial embedding.
    info = {"pixels": torch.empty((1, 1, 1, 1), device=initial.device),
            "emb": initial.detach()[None, None, None, :].expand(1, samples, -1, -1)}
    return model.rollout(info, actions[None], history_size=history_size)["predicted_emb"][0]
