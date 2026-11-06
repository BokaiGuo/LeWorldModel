"""Development CEM with physical-action constraints before model scoring."""
import json

import numpy as np
import torch
from stable_worldmodel.solver.utils import prepare_init_action

from experiments.pc_native_internal_response_20261105_codex import NativeInternalResponse


def normalized_bounds(mean, scale, action_dim, margin=1e-6, device="cpu", dtype=torch.float32):
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if mean.ndim != 1 or mean.shape != scale.shape or not len(mean):
        raise ValueError("action normalizer must be nonempty matching vectors")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("invalid action normalizer")
    if not 0 < margin < 1 or action_dim % len(mean):
        raise ValueError("invalid margin or action block dimension")
    low = np.tile((-1.0 + margin - mean) / scale, action_dim // len(mean))
    high = np.tile((1.0 - margin - mean) / scale, action_dim // len(mean))
    return (torch.as_tensor(low, device=device, dtype=dtype),
            torch.as_tensor(high, device=device, dtype=dtype))


class BoundedCEM(NativeInternalResponse):
    def __init__(self, *args, action_mean, action_scale, margin=1e-6, **kwargs):
        super().__init__(*args, mode="response", **kwargs)
        self.action_mean = list(action_mean)
        self.action_scale = list(action_scale)
        self.margin = margin

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        if self.callbacks:
            raise ValueError("callbacks are unsupported")
        total = len(next(iter(info_dict.values())))
        initial = prepare_init_action(self.model, info_dict, init_action, self.horizon,
                                      n_envs=total, action_dim=self.action_dim)
        mean, var = self.init_action_distrib(total, initial)
        mean, var = mean.to(self.device), var.to(self.device)
        low, high = normalized_bounds(self.action_mean, self.action_scale, self.action_dim,
                                      self.margin, self.device, self.dtype)
        mean = mean.clamp(min=low, max=high)
        audit, final_costs = [], []
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            size = end - start
            batch_mean, batch_var = mean[start:end], var[start:end]
            expanded = self._infos(info_dict, start, end, self.num_samples)
            for step in range(self.n_steps):
                noise = torch.randn(size, self.num_samples, self.horizon, self.action_dim,
                                    generator=self.torch_gen, device=self.device, dtype=self.dtype)
                raw = noise * batch_var[:, None] + batch_mean[:, None]
                raw[:, 0] = batch_mean
                projected = raw.clamp(min=low, max=high)
                costs = self._cost(expanded, projected)
                values, indices = torch.topk(costs, k=self.topk, dim=1, largest=False)
                batch = torch.arange(size, device=self.device)[:, None].expand(-1, self.topk)
                elite = projected[batch, indices]
                batch_mean, batch_var = elite.mean(dim=1), elite.std(dim=1)
                if not torch.isfinite(batch_mean).all() or not torch.isfinite(batch_var).all():
                    raise ValueError("nonfinite CEM distribution")
                if self.audit_path is not None:
                    audit.append({"batch_start": start, "batch_end": end, "step": step,
                                  "projected_candidates": (raw != projected).flatten(2).any(-1).sum(1).cpu().tolist(),
                                  "scored_per_environment": self.num_samples})
            mean[start:end], var[start:end] = batch_mean, batch_var
            final_costs.extend(values.mean(1).cpu().tolist())
        if ((mean < low) | (mean > high)).any():
            raise ValueError("mean escaped action bounds")
        if self.audit_path is not None:
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"solve": self.solve_index, "iterations": audit},
                                        allow_nan=False) + "\n")
        self.solve_index += 1
        return {"actions": mean.cpu(), "mean": [mean.cpu()], "var": [var.cpu()],
                "costs": final_costs}
