"""Fixed-budget CEM candidate insertion for real-feedback tail carryover."""
from __future__ import annotations

import torch


def cem_with_carried_tail(cost_fn, tail, *, samples, steps, elite, seed):
    if tail.ndim != 3 or tail.shape[0] != 1 or not torch.isfinite(tail).all():
        raise ValueError("tail must be a finite singleton plan")
    if samples < 3 or steps < 1 or not 1 < elite < samples:
        raise ValueError("invalid CEM budget")
    device, dtype = tail.device, tail.dtype
    generator = torch.Generator(device=device).manual_seed(int(seed))
    mean = torch.zeros_like(tail[0])
    std = torch.ones_like(mean)
    old_costs, best_costs = [], []
    selected = None
    final_index = None
    for _ in range(steps):
        candidates = torch.randn((samples, *tail.shape[1:]), generator=generator,
                                 device=device, dtype=dtype)
        candidates = candidates * std.unsqueeze(0) + mean.unsqueeze(0)
        candidates[0] = mean
        candidates[1] = tail[0]
        costs = cost_fn(candidates)
        if costs.shape != (samples,) or not torch.isfinite(costs).all():
            raise ValueError("invalid candidate costs")
        values, indices = torch.topk(costs, k=elite, largest=False)
        chosen = candidates[indices]
        mean = chosen.mean(dim=0)
        std = chosen.std(dim=0).clamp_min(1.0e-4)
        final_index = int(torch.argmin(costs))
        selected = candidates[final_index:final_index + 1].detach().clone()
        old_costs.append(float(costs[1]))
        best_costs.append(float(costs[final_index]))
    return selected, {
        "old_tail_model_costs": old_costs,
        "iteration_best_model_costs": best_costs,
        "final_selected_index": final_index,
        "selected_old_tail": final_index == 1,
        "forward_rollout_evaluations": samples * steps,
        "model_predict_sample_evaluations": samples * steps * tail.shape[1],
        "backward_vjp_rows": 0,
        "candidate_policy": "mean index 0, carried tail index 1, remaining samples seeded CEM",
    }
