"""Opt-in, checkpoint-bound LeWM RGB boundary for new experiments.

This verifies tensor values and dependency hashes; it is not a sandbox and
does not certify upstream origin, code equivalence, or scientific validity.
"""
import hashlib
from pathlib import Path

import torch

from experiments.checkpoint_compat import remap_legacy_vit_keys
from experiments.pc_input_permissions import RGBLeWM, file_hash


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("non-tensor model state")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class CheckpointBoundRGB(RGBLeWM):
    def __init__(self, model, audit, weights, config):
        super().__init__(model, audit)
        self._bound_model = model
        self._weights = Path(weights).resolve()
        self._config = Path(config).resolve()
        dependencies = dict(audit.dependencies)
        if any(str(p) not in dependencies for p in (self._weights, self._config)):
            raise ValueError("checkpoint or configuration is not an audited dependency")
        if self._weights == self._config:
            raise ValueError("checkpoint and configuration must be distinct")
        self._verify_weights()
        self._initial_hash = state_hash(model.state_dict())

    def _verify_weights(self):
        self.audit.verify()
        if self._model is not self._bound_model:
            raise ValueError("model object replaced")
        if any(m.training for m in self._model.modules()) or any(p.requires_grad for p in self._model.parameters()):
            raise ValueError("model no longer frozen and in eval mode")
        raw = torch.load(self._weights, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict):
            raise ValueError("checkpoint must be a state dictionary")
        expected, _ = remap_legacy_vit_keys(raw)
        actual = self._model.state_dict()
        if set(expected) != set(actual):
            raise ValueError("checkpoint model state keys mismatch")
        for name, tensor in actual.items():
            ref = expected[name]
            if (not isinstance(ref, torch.Tensor) or ref.shape != tensor.shape or ref.dtype != tensor.dtype
                    or not torch.equal(ref.detach().cpu(), tensor.detach().cpu())):
                raise ValueError(f"weight mismatch: {name}")

    def finish(self):
        if not self.receipts:
            raise ValueError("no runtime receipts")
        self._verify_weights()
        final_hash = state_hash(self._model.state_dict())
        if final_hash != self._initial_hash:
            raise ValueError("model state changed")
        return {"schema": "pc.checkpoint_bound_rgb.v1",
                "track_id": self.audit.track_id, "input_audit_sha256": self.audit.fingerprint,
                "weights_sha256": file_hash(self._weights), "config_sha256": file_hash(self._config),
                "initial_model_state_sha256": self._initial_hash, "model_state_sha256": final_hash,
                "runtime_receipts": len(self.receipts),
                "scope": "tensor and dependency binding only; not origin or scientific certification"}
