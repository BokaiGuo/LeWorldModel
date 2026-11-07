"""Differentiable LeWM continuation with aligned latent/action history."""
import torch
from stable_worldmodel.wm.lewm import LeWM


def rollout_history(model, embeddings, past_actions, future_actions, history_size=3):
    """Continue native rollout without detaching history.

    embeddings: (N,H,D), ordered initial observation followed by predicted
      successors. past_actions: (N,H-1,A), the actions that led to those
      successors. future_actions: (N,K,A), K>=1.
    The caller is responsible for matching the histories to the same root and
    action clock. Outputs include all H historical embeddings and K successors.
    """
    if type(model) is not LeWM:
        raise TypeError("only installed stable-worldmodel LeWM is supported")
    if embeddings.ndim != 3 or past_actions.ndim != 3 or future_actions.ndim != 3:
        raise ValueError("history, past actions and future actions must be rank three")
    n, h, d = embeddings.shape
    if (n < 1 or h < 1 or d < 1 or history_size < 1 or future_actions.shape[1] < 1
            or past_actions.shape[:2] != (n, h - 1)
            or future_actions.shape[0] != n
            or past_actions.shape[2] != future_actions.shape[2]
            or future_actions.shape[2] < 1):
        raise ValueError("latent/action history or horizon mismatch")
    if (embeddings.device != past_actions.device or embeddings.device != future_actions.device
            or past_actions.dtype != future_actions.dtype):
        raise ValueError("history tensors must have matching devices and action dtypes")
    info = {"pixels": torch.empty((1, 1, h, 1), device=embeddings.device),
            "emb": embeddings.unsqueeze(0)}
    actions = torch.cat((past_actions, future_actions), dim=1)
    return model.rollout(info, actions.unsqueeze(0), history_size=history_size)["predicted_emb"][0]
