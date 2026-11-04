"""PointMaze model callback for the AP-compatible normalized-action CEM.

The caller owns RGB encoding, target retrieval, observation refresh, the CUDA
generator, and physical execution.  This adapter does not certify full AP.
"""
from __future__ import annotations

import torch


class PointMazeAPCallback:
    """Score bounded normalized five-step packets against a frozen target token."""

    def __init__(self, model, context, target_proprio, device):
        self.model = model
        self.context = context
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("neural AP planning requires host CUDA")
        if not isinstance(target_proprio, torch.Tensor):
            raise TypeError("target proprio must be a tensor")
        if (target_proprio.device.type != "cuda"
                or (self.device.index is not None and target_proprio.device.index != self.device.index)
                or not torch.isfinite(target_proprio).all()):
            raise ValueError("target proprio must be finite and on the planning device")
        self.device = target_proprio.device
        self.target = target_proprio.detach()
        self.batches = 0
        self.samples = 0

    def __call__(self, packets):
        if (not isinstance(packets, torch.Tensor) or packets.device != self.device
                or packets.dtype != torch.float32 or packets.ndim != 4
                or packets.shape[0] != 1 or packets.shape[2:] != (1, 10)
                or packets.shape[1] < 1 or not torch.isfinite(packets).all()):
            raise ValueError("expected finite CUDA float32 packets [1,N,1,10]")
        # Packet is already normalized by the AP kernel: do not normalize twice.
        suffix = packets[:, :, 0, :]
        with torch.no_grad():
            predicted = self.model.unroll(self.context, act_suffix=suffix)["proprio"][-1]
            if predicted.shape[0] != packets.shape[1] or not torch.isfinite(predicted).all():
                raise ValueError("invalid predicted proprio batch")
            try:
                difference = predicted - self.target
            except RuntimeError as exc:
                raise ValueError("target and predicted proprio are incompatible") from exc
            if difference.shape != predicted.shape:
                raise ValueError("target broadcasting changed the prediction shape")
            costs = difference.reshape(packets.shape[1], -1).square().mean(dim=1)
        if not torch.isfinite(costs).all():
            raise ValueError("nonfinite AP costs")
        self.batches += 1
        self.samples += packets.shape[1]
        return costs.unsqueeze(0)
