"""Training-free raw-action and latent projection adapter for the v3 compiler."""
import numpy as np
import torch


def raw_to_packets(actions, mean, std, *, device):
    raw = np.asarray(actions, dtype=np.float32)
    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    if (raw.ndim != 3 or raw.shape[2] != 2 or raw.shape[1] not in (5, 10, 15)
            or mean.shape != (10,) or std.shape != (10,) or not np.isfinite(raw).all()
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)
            or np.any(np.abs(raw) > 1 + 1e-6)):
        raise ValueError("invalid raw action or packet statistics")
    packets = torch.as_tensor(raw.reshape(len(raw), -1, 10), device=device)
    return (packets - torch.as_tensor(mean, device=device)) / torch.as_tensor(std, device=device)


def cached_terminal(model, encoded, history, future, lower, upper):
    if encoded.ndim != 3 or encoded.shape[0] != 1 or history.shape != (1, encoded.shape[1], future.shape[-1]):
        raise ValueError("invalid encoded or action history")
    if future.ndim != 3 or future.shape[1] < 1:
        raise ValueError("invalid future packets")
    if torch.any(future < lower - 1e-5) or torch.any(future > upper + 1e-5):
        raise ValueError("normalized packets outside physical bounds")
    future = torch.maximum(torch.minimum(future, upper), lower)
    batch, history_size = future.shape[0], encoded.shape[1]
    action_context = torch.cat((history[0, 1:].unsqueeze(0).expand(batch, -1, -1), future[:, :1]), dim=1)
    emb_context = encoded.expand(batch, -1, -1)
    for index in range(future.shape[1]):
        action_embedding = model.action_encoder(action_context[:, -history_size:])
        terminal = model.predict(emb_context[:, -history_size:], action_embedding)[:, -1]
        if index + 1 < future.shape[1]:
            emb_context = torch.cat((emb_context, terminal[:, None]), dim=1)
            action_context = torch.cat((action_context, future[:, index + 1:index + 2]), dim=1)
    return terminal


def goal_chart(encoded, goal, seed):
    """Two orthonormal axes using only goal and observed-root frozen latents."""
    root = encoded[0, -1]
    difference = goal - root
    norm = torch.linalg.vector_norm(difference)
    if not torch.isfinite(norm) or norm < 1e-8:
        raise ValueError("degenerate root-goal direction")
    first = difference / norm
    random = np.random.default_rng(seed).normal(size=first.numel()).astype(np.float32)
    second = torch.as_tensor(random, device=first.device, dtype=first.dtype)
    second -= torch.dot(first, second) * first
    norm2 = torch.linalg.vector_norm(second)
    if not torch.isfinite(norm2) or norm2 < 1e-8:
        raise ValueError("degenerate orthogonal chart direction")
    return torch.stack((first, second / norm2), dim=1)
