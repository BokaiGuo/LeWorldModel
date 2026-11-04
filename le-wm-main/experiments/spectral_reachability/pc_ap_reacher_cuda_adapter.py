"""Bind AP's normalized five-action blocks to a frozen LeWM predictor.

The caller supplies a causally aligned, real-observation context and is
responsible for action normalization and provenance. No simulator state is
accepted here. This adapter does not implement a physical control loop.
"""
from __future__ import annotations

import numpy as np
import torch


class ReacherCudaPredictor:
    def __init__(self, model, encoded_context: torch.Tensor, action_history: torch.Tensor):
        if encoded_context.device.type != "cuda" or action_history.device.type != "cuda":
            raise ValueError("host CUDA context and history required")
        if encoded_context.ndim != 3 or action_history.ndim != 3:
            raise ValueError("expected batched sequence tensors")
        if (encoded_context.shape[0] != 1 or action_history.shape[0] != 1
                or encoded_context.shape[1] != action_history.shape[1]
                or encoded_context.shape[1] < 2 or action_history.shape[2] != 10):
            raise ValueError("expected aligned Reacher history with five 2D actions per token")
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("model must be frozen and in evaluation mode")
        if not torch.isfinite(encoded_context).all() or not torch.isfinite(action_history).all():
            raise ValueError("nonfinite context or history")
        self._model = model
        self._emb = encoded_context.detach()
        self._history = action_history.detach()
        self.predicted_blocks = 0
        self.prediction_batches = 0

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions)
        if actions.ndim != 3 or actions.shape[1:] != (5, 2) or len(actions) < 1 or not np.isfinite(actions).all():
            raise ValueError("expected finite [batch,5,2] normalized actions")
        action = torch.as_tensor(actions, dtype=self._history.dtype, device=self._history.device).reshape(len(actions), 1, 10)
        context_actions = torch.cat((self._history[:, 1:].expand(len(actions), -1, -1), action), dim=1)
        context_emb = self._emb.expand(len(actions), -1, -1)
        # Count attempted work before invoking the model, including errors.
        self.predicted_blocks += len(actions)
        self.prediction_batches += 1
        with torch.no_grad():
            output = self._model.predict(context_emb, self._model.action_encoder(context_actions))
        if output.ndim != 3 or output.shape[:2] != context_emb.shape[:2] or not torch.isfinite(output).all():
            raise RuntimeError("invalid predicted embedding")
        return output[:, -1].detach().cpu().numpy()
