"""Opt-in RGB history boundary with explicit physical-clock declarations.

Time values and history are caller declarations, not independent proof of
physical dispatch. The existing history assembler remains the prediction path.
"""
from numbers import Integral

import torch

from experiments.pc_rgb_history_contract_20261107_codex import score_future
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB


class TimedHistoryRGB(SingleFrameRGB):
    def __init__(self, *args, max_frames, **kwargs):
        if not isinstance(max_frames, Integral) or isinstance(max_frames, bool) or max_frames < 1:
            raise ValueError("positive integral history window required")
        self.max_frames = int(max_frames)
        super().__init__(*args, **kwargs)

    def score_timed(self, *, pixels, goal, history, future, frame_steps, current_step):
        tensors = (pixels, goal, history, future)
        if not all(isinstance(x, torch.Tensor) and x.is_floating_point() for x in tensors):
            raise ValueError("floating tensor inputs required")
        if (pixels.ndim != 6 or pixels.shape[:2] != (1, 1) or pixels.shape[3] != 3
                or not 1 <= pixels.shape[2] <= self.max_frames or min(pixels.shape[-2:]) < 1
                or goal.shape != (1, 1, 1, *pixels.shape[3:])):
            raise ValueError("invalid RGB frame window or goal")
        n = pixels.shape[2]
        if (not isinstance(frame_steps, (tuple, list)) or len(frame_steps) != n
                or not isinstance(current_step, Integral) or isinstance(current_step, bool)
                or any(not isinstance(x, Integral) or isinstance(x, bool) or x < 0 for x in frame_steps)
                or frame_steps[-1] != current_step
                or any(b - a != self.packet_steps for a, b in zip(frame_steps, frame_steps[1:]))):
            raise ValueError("history frame clock mismatch")
        if history.shape != (1, 1, n - 1, self.packet_dim):
            raise ValueError("history packet count must match frame clock")
        if (future.ndim != 4 or future.shape[0] != 1 or future.shape[1] < 1
                or future.shape[2] < 1 or future.shape[3] != self.packet_dim):
            raise ValueError("invalid future packet shape")
        if len({x.device for x in tensors}) != 1 or history.dtype != future.dtype:
            raise ValueError("device or action dtype mismatch")
        if not all(torch.isfinite(x).all().item() for x in tensors):
            raise ValueError("nonfinite input")
        if any(m.training for m in self.model.modules()) or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("model must remain frozen")
        for actions in (history, future):
            raw = actions.reshape(*actions.shape[:-1], self.packet_steps, len(self._mean))
            mean = torch.as_tensor(self._mean, dtype=raw.dtype, device=raw.device)
            scale = torch.as_tensor(self._scale, dtype=raw.dtype, device=raw.device)
            low = torch.as_tensor(self._low, dtype=raw.dtype, device=raw.device)
            high = torch.as_tensor(self._high, dtype=raw.dtype, device=raw.device)
            physical = raw * scale + mean
            if ((physical < low - 1e-6) | (physical > high + 1e-6)).any().item():
                raise ValueError("history or future exceeds physical action bounds")
        values, context = score_future(self._wrapper, pixels=pixels, goal=goal, history=history, future=future)
        if values.shape != future.shape[:2] or not torch.isfinite(values).all().item():
            raise ValueError("invalid model score")
        return values, context
