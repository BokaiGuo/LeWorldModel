"""Bridge numerical sampling kernels to the existing frozen CUDA P0 objective.

Only proposal arrays and scalar costs cross the host boundary. All model
encodes/predictions remain on CUDA. This is not an MPC feedback schedule.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress import _cost_fn


def _require_frozen_eval(model, device):
    """Reject partially trained or wrong-device models before delegated calls."""
    if any(module.training for module in model.modules()):
        raise ValueError("every model module must be in evaluation mode")
    for parameter in model.parameters():
        if parameter.requires_grad or parameter.device != device:
            raise ValueError("frozen parameters on the input CUDA device required")


def _require_real_cost_dtype(costs):
    if not isinstance(costs, torch.Tensor) or not costs.is_floating_point():
        raise ValueError("real floating-point tensor costs required; complex/bool/integer costs forbidden")


class CudaSamplingCost:
    def __init__(self, model, pixels, goal, history, *, action_lower, action_upper,
                 future_horizon, cost_fn=_cost_fn):
        values = (pixels, goal, history, action_lower, action_upper)
        if any(v.device.type != "cuda" for v in values):
            raise ValueError("all model inputs must use host CUDA; CPU fallback forbidden")
        if len({v.device for v in values}) != 1:
            raise ValueError("all inputs must share the same CUDA device")
        if type(future_horizon) is not int or future_horizon < 1:
            raise ValueError("positive future horizon required")
        if history.ndim != 3 or history.shape[0] != 1:
            raise ValueError("single-root [1,history,actions] context required")
        if (action_lower.shape != (history.shape[-1],) or action_upper.shape != action_lower.shape
                or not torch.isfinite(action_lower).all() or not torch.isfinite(action_upper).all()
                or not torch.all(action_lower < action_upper)):
            raise ValueError("finite ordered normalized action bounds required")
        _require_frozen_eval(model, history.device)
        self.model, self.pixels, self.goal, self.history = model, pixels, goal, history
        self.lower, self.upper = action_lower, action_upper
        self.horizon, self.cost_fn = future_horizon, cost_fn
        self.calls = self.samples = 0
        self.completed_calls = self.completed_samples = self.failed_calls = 0

    def __call__(self, candidates):
        _require_frozen_eval(self.model, self.history.device)
        candidates = np.asarray(candidates)
        expected = (self.horizon, self.history.shape[-1])
        if (candidates.ndim != 3 or candidates.shape[0] < 1 or candidates.shape[1:] != expected
                or candidates.dtype.kind not in "fi" or not np.isfinite(candidates).all()):
            raise ValueError("finite [batch,horizon,action] parameter proposals required")
        values = torch.as_tensor(candidates, device=self.history.device, dtype=self.history.dtype)
        if not torch.isfinite(values).all():
            raise ValueError("candidate conversion overflow")
        self.calls += 1
        self.samples += len(candidates)
        try:
            with torch.no_grad():
                costs = self.cost_fn(self.model, self.pixels, self.goal, self.history, values,
                                     action_lower=self.lower, action_upper=self.upper)
            _require_real_cost_dtype(costs)
            if costs.device != self.history.device or costs.shape != (len(candidates),) or not torch.isfinite(costs).all():
                raise ValueError("finite CUDA per-candidate cost vector required")
            exported = costs.detach().cpu().numpy().astype(np.float64)
            if exported.shape != (len(candidates),) or not np.isfinite(exported).all():
                raise ValueError("finite exported per-candidate cost vector required")
        except BaseException:
            self.failed_calls += 1
            raise
        self.completed_calls += 1
        self.completed_samples += len(candidates)
        return exported

    def receipt(self):
        return {"candidate_cost_calls": self.calls, "candidate_cost_samples": self.samples,
                "completed_candidate_cost_calls": self.completed_calls,
                "completed_candidate_cost_samples": self.completed_samples,
                "failed_candidate_cost_calls": self.failed_calls,
                "counter_semantics": "candidate_cost counters are submitted after CUDA conversion; completed counters require valid output",
                "model_device": str(self.history.device), "horizon_tokens": self.horizon,
                "normalized_action_lower": self.lower.detach().cpu().tolist(),
                "normalized_action_upper": self.upper.detach().cpu().tolist(),
                "model_input_dtype": str(self.history.dtype),
                "coordinate": "unconstrained_parameters_tanh_mapped_to_normalized_physical_box",
                "scope": "candidate scoring only; not predict/encoder/backward or total-compute accounting"}
