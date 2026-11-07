"""Explicit single-frame LeWM scoring contract; not an adversarial sandbox.

LeWM's current rollout replaces ``info['action']`` with candidate packets.
Therefore this adapter rejects nonempty action history rather than claiming it
is consumed. Historical experiments using their original wrapper are unchanged.
"""
import hashlib
import struct

import numpy as np
import torch

from experiments.pc_input_permissions import RGB, RGBLeWM


def model_digest(model):
    """Hash state names, dtypes, shapes, and bytes, including buffers."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        for item in (name, str(value.dtype), repr(tuple(value.shape))):
            encoded = item.encode()
            digest.update(struct.pack(">Q", len(encoded)))
            digest.update(encoded)
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


class SingleFrameRGB:
    def __init__(self, model, audit, *, expected_model_digest, action_mean, action_scale,
                 packet_steps, physical_low, physical_high):
        if audit.track_id != RGB or audit.family != "lewm":
            raise ValueError("RGB LeWM track required")
        if not isinstance(packet_steps, int) or isinstance(packet_steps, bool) or packet_steps < 1:
            raise ValueError("positive integral packet steps required")
        mean, scale = np.asarray(action_mean, float), np.asarray(action_scale, float)
        if (mean.ndim != 1 or not len(mean) or scale.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError("invalid action normalizer")
        low, high = np.broadcast_to(physical_low, mean.shape), np.broadcast_to(physical_high, mean.shape)
        if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low >= high):
            raise ValueError("invalid physical action bounds")
        self.model, self.audit = model, audit
        self._wrapper = RGBLeWM(model, audit)
        self._expected_digest = expected_model_digest
        self._mean, self._scale = mean.copy(), scale.copy()
        self._low, self._high = np.array(low, float), np.array(high, float)
        self.packet_steps = packet_steps
        self.packet_dim = packet_steps * len(mean)
        self.verify_unchanged()

    def verify_unchanged(self):
        self.audit.verify()
        if (any(module.training for module in self.model.modules())
                or any(p.requires_grad for p in self.model.parameters())
                or model_digest(self.model) != self._expected_digest):
            raise ValueError("model frozen state or fingerprint changed")
        return True

    def score(self, inputs, actions):
        if set(inputs) != {"pixels", "goal", "action"}:
            raise ValueError("RGB, goal and explicitly empty history required")
        pixels, goal, history = (inputs[k] for k in ("pixels", "goal", "action"))
        if not all(isinstance(x, torch.Tensor) for x in (pixels, goal, history, actions)):
            raise ValueError("tensor inputs required")
        if not all(x.is_floating_point() for x in (pixels, goal, history, actions)):
            raise ValueError("floating point tensors required")
        if (pixels.ndim != 6 or pixels.shape[:4] != (1, 1, 1, 3) or goal.shape != pixels.shape
                or pixels.shape[-1] < 1 or pixels.shape[-2] < 1):
            raise ValueError("single-frame batch RGB and matching goal required")
        if tuple(history.shape) != (1, 1, 0, self.packet_dim):
            raise ValueError("nonempty or malformed action history is not supported")
        if (actions.ndim != 4 or actions.shape[0] != 1 or actions.shape[1] < 1
                or actions.shape[2] < 1 or actions.shape[3] != self.packet_dim):
            raise ValueError("invalid normalized candidate packet shape")
        if not all(torch.isfinite(x).all().item() for x in (pixels, goal, history, actions)):
            raise ValueError("nonfinite input")
        if len({x.device for x in (pixels, goal, history, actions)}) != 1:
            raise ValueError("inputs must share device")
        if any(module.training for module in self.model.modules()) or any(
                p.requires_grad for p in self.model.parameters()):
            raise ValueError("model must remain frozen")
        raw = actions.reshape(*actions.shape[:-1], self.packet_steps, -1)
        mean = torch.as_tensor(self._mean, device=raw.device, dtype=raw.dtype)
        scale = torch.as_tensor(self._scale, device=raw.device, dtype=raw.dtype)
        low = torch.as_tensor(self._low, device=raw.device, dtype=raw.dtype)
        high = torch.as_tensor(self._high, device=raw.device, dtype=raw.dtype)
        physical = raw * scale + mean
        if ((physical < low - 1e-6) | (physical > high + 1e-6)).any().item():
            raise ValueError("candidate exceeds physical action bounds")
        values, context = self._wrapper.score(inputs, actions)
        if values.shape != actions.shape[:2] or not torch.isfinite(values).all().item():
            raise ValueError("invalid model score")
        return values, context
