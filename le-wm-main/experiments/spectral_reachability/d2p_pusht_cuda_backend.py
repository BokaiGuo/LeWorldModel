"""Differentiable frozen PushT backend for the reusable D2P compiler.

The backend is deliberately narrow: official five-step PushT packets,
host-CUDA execution, the published action normalization contract, and the
three-latent/two-action-history context used by the existing replay adapter.
It exposes no simulator state or reward to the compiler.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .d2p_composition_witness_v2 import _context


class PushTCudaBackend:
    """Adapt a frozen LeWM checkpoint to ``LocalCompiler``."""

    def __init__(self, model, device, *, emit=None):
        import torch

        if getattr(device, "type", None) != "cuda":
            raise ValueError("host CUDA only; CPU neural fallback forbidden")
        self.torch = torch
        self.model = model
        self.device = device
        self.emit = emit if emit is not None else lambda event: None
        self.counters = {
            "rollout_calls": 0,
            "jacobian_calls": 0,
            "packet_transitions": 0,
            "feedback_encode_attempted": 0,
            "feedback_encode_returned": 0,
        }
        self._validate_model()

    def _validate_model(self):
        if (any(module.training for module in self.model.modules())
                or any(parameter.requires_grad or parameter.device != self.device
                       for parameter in self.model.parameters())
                or any(buffer.device != self.device for buffer in self.model.buffers())):
            raise ValueError("frozen eval model must be entirely on the requested CUDA device")

    def _validate_context(self, context):
        _context(context)
        if (context["context_length"] != 3
                or context["latent_dim"] != 192
                or context["action_dim"] != 10):
            raise ValueError("official PushT compiler context is required")

    @staticmethod
    def _validate_raw_actions(actions):
        raw = np.asarray(actions)
        if (raw.dtype.kind not in "fi" or raw.ndim != 2 or raw.shape[1] != 2
                or raw.shape[0] < 5 or raw.shape[0] % 5
                or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1)):
            raise ValueError("raw actions must be bounded [N,2] five-step packets")
        return raw.astype(np.float32, copy=True)

    def _validate_actions(self, actions):
        raw = self._validate_raw_actions(actions)
        return self.torch.as_tensor(raw, dtype=self.torch.float32, device=self.device)

    def _normalize(self, packet):
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0

        mean = self.torch.as_tensor(
            p0.PUBLIC_ACTION_MEAN, dtype=packet.dtype, device=self.device
        )
        std = self.torch.as_tensor(
            p0.PUBLIC_ACTION_STD, dtype=packet.dtype, device=self.device
        )
        return p0.packets.normalize_live_packet(
            packet.reshape(1, 5, 2), mean=mean, std=std
        )

    def _tensors(self, context):
        latent = self.torch.tensor(
            context["latent_history"], dtype=self.torch.float32, device=self.device
        )[None]
        history = self.torch.tensor(
            context["action_history"], dtype=self.torch.float32, device=self.device
        )[None]
        return latent, history

    def _transition(self, latent, history, raw):
        self.counters["packet_transitions"] += 1
        self.emit({"kind": "pusht_packet_transition",
                   "counts": dict(self.counters)})
        token = self._normalize(raw)
        outgoing = self.torch.cat((history, token[:, None]), dim=1)
        successor = self.model.predict(
            latent, self.model.action_encoder(outgoing)
        )[:, -1]
        if not bool(self.torch.isfinite(successor).all()):
            raise ValueError("nonfinite frozen-model successor")
        return (
            self.torch.cat((latent[:, 1:], successor[:, None]), dim=1),
            outgoing[:, 1:],
        )

    def _rollout_tensor(self, context, actions, *, grad):
        latent, history = self._tensors(context)
        packets = actions.reshape(-1, 5, 2)
        if grad:
            for packet in packets:
                latent, history = self._transition(latent, history, packet)
        else:
            with self.torch.no_grad():
                for packet in packets:
                    latent, history = self._transition(latent, history, packet)
        return latent, history

    def rollout(self, context, raw_actions):
        self._validate_model()
        self._validate_context(context)
        actions = self._validate_actions(raw_actions)
        self.counters["rollout_calls"] += 1
        latent, history = self._rollout_tensor(context, actions, grad=False)
        result = deepcopy(context)
        result["latent_history"] = latent[0].cpu().tolist()
        result["action_history"] = history[0].cpu().tolist()
        _context(result)
        return result

    def jacobian(self, context, raw_actions):
        self._validate_model()
        self._validate_context(context)
        actions = self._validate_actions(raw_actions)
        self.counters["jacobian_calls"] += 1
        torch = self.torch
        flat = actions.reshape(-1).detach().requires_grad_(True)

        def terminal(vector):
            latent, _ = self._rollout_tensor(
                context, vector.reshape(-1, 2), grad=True
            )
            return latent[0, -1]

        value = torch.autograd.functional.jacobian(
            terminal, flat, vectorize=True, create_graph=False
        )
        if value.shape != (192, flat.numel()) or not bool(torch.isfinite(value).all()):
            raise ValueError("invalid PushT action Jacobian")
        return value.detach().cpu().numpy().astype(np.float64)

    def observe(self, context, actual_actions, rgb):
        """Update one boundary using executed actions and an actual RGB frame.

        The caller must authenticate execution/RGB provenance. This method
        performs no prediction and cannot repair a partial execution packet.
        """
        self._validate_model()
        self._validate_context(context)
        raw = self._validate_raw_actions(actual_actions)
        image = np.asarray(rgb)
        if raw.shape != (5, 2):
            raise ValueError("feedback requires exactly one complete packet")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("actual uint8 RGB frame required")
        from experiments.run_pusht_counterfactual import image_transform
        from . import preflight_pusht_d2p_p0 as p0

        self.counters["feedback_encode_attempted"] += 1
        self.emit({"kind": "feedback_encode_attempt", "counts": dict(self.counters)})
        with self.torch.no_grad():
            latent = self.model.encode({"pixels": p0._image_tensor(
                image.copy(), transform=image_transform(), device=self.device)})["emb"][0, 0]
            if latent.shape != (192,) or not bool(self.torch.isfinite(latent).all()):
                raise ValueError("invalid actual RGB encoding")
            token = self._normalize(self.torch.tensor(raw, device=self.device))[0]
        self.counters["feedback_encode_returned"] += 1
        result = deepcopy(context)
        result["latent_history"] = context["latent_history"][1:] + [latent.cpu().tolist()]
        result["action_history"] = context["action_history"][1:] + [token.cpu().tolist()]
        _context(result)
        self.emit({"kind": "feedback_encode_return", "counts": dict(self.counters)})
        return result
