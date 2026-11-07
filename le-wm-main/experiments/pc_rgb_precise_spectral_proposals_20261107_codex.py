"""Legal spectral candidate generation under scoped higher-order precision."""
import numpy as np
import torch

from experiments.pc_rgb_derivative_backend_20261107_codex import precise_action_derivatives


def jacobian_and_directions(rollout, raw, count=2):
    if raw.ndim != 2 or not raw.is_floating_point() or not torch.isfinite(raw).all():
        raise ValueError("finite matrix of raw actions required")
    if not 1 <= count <= raw.numel():
        raise ValueError("invalid direction count")
    columns = []
    flat_size = raw.numel()
    with precise_action_derivatives():
        for i in range(flat_size):
            tangent = torch.zeros_like(raw)
            tangent.reshape(-1)[i] = 1
            _, value = torch.autograd.functional.jvp(rollout, raw, tangent, strict=True)
            columns.append(value.detach().reshape(-1))
        matrix = torch.stack(columns, dim=1)
        if not torch.isfinite(matrix).all():
            raise ValueError("nonfinite Jacobian")
        _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    if count > vh.shape[0] or singular[count - 1] <= 1e-10:
        raise ValueError("insufficient Jacobian rank")
    return matrix.detach(), singular.detach(), vh[:count].reshape(count, *raw.shape).detach()


def make_candidates(raw, directions, amplitude=0.3):
    raw, directions = np.asarray(raw, dtype=np.float64), np.asarray(directions, dtype=np.float64)
    if raw.ndim != 2 or directions.shape != (2, *raw.shape) or not np.isfinite(raw).all() or (
            not np.isfinite(directions).all()) or not 0 < amplitude <= 1:
        raise ValueError("invalid candidates")
    if np.abs(raw).max() > 1 - amplitude + 1e-7:
        raise ValueError("insufficient action margin")
    norms = np.max(np.abs(directions), axis=(1, 2))
    if (norms <= 1e-12).any():
        raise ValueError("zero direction")
    normalized = directions / norms[:, None, None]
    chains = np.stack([raw, raw + amplitude * normalized[0], raw - amplitude * normalized[0],
                       raw + amplitude * normalized[1], raw - amplitude * normalized[1]])
    if np.abs(chains).max() > 1 + 1e-7:
        raise ValueError("physical action bounds exceeded")
    return chains.astype(np.float32)
