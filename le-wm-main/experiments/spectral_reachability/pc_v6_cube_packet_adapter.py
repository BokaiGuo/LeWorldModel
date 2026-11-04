"""Five-dimensional Cube action bridge for the v6 development compiler."""

import numpy as np
import torch

from experiments.spectral_reachability.pc_response_temporal_v3_reacher_adapter import (
    cached_terminal,
    goal_chart,
)

ACTION_DIM = 5
PACKET = 5


def raw_to_packets(actions, mean, std, *, device):
    """Validate raw bounded future actions and normalize time-major packets."""
    raw = np.asarray(actions, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if (raw.ndim != 3 or raw.shape[0] < 1 or raw.shape[2] != ACTION_DIM
            or raw.shape[1] not in (5, 10, 15)
            or mean.shape != (25,) or std.shape != (25,)
            or not np.isfinite(raw).all() or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)
            or np.any(np.abs(raw) > 1)):
        raise ValueError("invalid Cube future actions or normalization")
    tensor = torch.as_tensor(raw.reshape(len(raw), -1, 25), device=device)
    return (tensor - torch.as_tensor(mean, device=device)) / torch.as_tensor(std, device=device)


def make_predictor(model, encoded, history, goal, mean, std, *, chart_seed):
    """Return projected positions and full-latent MSE for bounded raw actions.

    Encoded frames and history must already be bound to the same observed root.
    Goal and chart use only frozen latent representations, not physical state.
    """
    if encoded.ndim != 3 or encoded.shape[0] != 1 or encoded.shape[1] < 2:
        raise ValueError("expected one causal encoded history")
    if history.shape != (1, encoded.shape[1], 25):
        raise ValueError("history must contain one 25-wide token per frame")
    if goal.shape != (encoded.shape[-1],) or not torch.isfinite(goal).all():
        raise ValueError("invalid goal latent")
    if not torch.isfinite(encoded).all() or not torch.isfinite(history).all():
        raise ValueError("invalid encoded history")
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    # Validate statistics before computing normalized physical bounds.
    raw_to_packets(np.zeros((1, PACKET, ACTION_DIM)), mean, std, device=encoded.device)
    lower = torch.as_tensor((-1 - mean) / std, device=encoded.device)
    upper = torch.as_tensor((1 - mean) / std, device=encoded.device)
    chart = goal_chart(encoded, goal, chart_seed)

    def predict(actions):
        future = raw_to_packets(actions, mean, std, device=encoded.device)
        with torch.no_grad():
            terminal = cached_terminal(model, encoded, history, future, lower, upper)
            positions = terminal @ chart
            scores = torch.mean((terminal - goal) ** 2, dim=-1)
        if not torch.isfinite(positions).all() or not torch.isfinite(scores).all():
            raise ValueError("non-finite Cube model prediction")
        return positions.cpu().numpy(), scores.cpu().numpy()

    return predict, (goal @ chart).detach().cpu().numpy()
