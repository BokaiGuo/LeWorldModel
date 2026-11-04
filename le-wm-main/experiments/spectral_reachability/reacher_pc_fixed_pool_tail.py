"""Evaluate a carried tail after an unchanged CEM candidate population."""
from __future__ import annotations

import torch


def cem_fixed_pool_with_tail(cost_fn, tail, *, samples, steps, elite, seed):
    if tail.ndim != 3 or tail.shape[0] != 1 or not torch.isfinite(tail).all():
        raise ValueError("tail must be a finite singleton program")
    if samples < 3 or steps < 1 or not 1 < elite < samples:
        raise ValueError("invalid CEM budget")
    device, dtype = tail.device, tail.dtype
    generator = torch.Generator(device=device).manual_seed(int(seed))
    mean = torch.zeros_like(tail[0])
    std = torch.ones_like(mean)
    best = None
    best_cost = None
    for _ in range(steps):
        candidates = torch.randn((samples, *tail.shape[1:]), generator=generator,
                                 device=device, dtype=dtype)
        candidates = candidates * std.unsqueeze(0) + mean.unsqueeze(0)
        candidates[0] = mean
        costs = cost_fn(candidates)
        if costs.shape != (samples,) or not torch.isfinite(costs).all():
            raise ValueError("invalid CEM candidate costs")
        _, indices = torch.topk(costs, k=elite, largest=False)
        chosen = candidates[indices]
        mean = chosen.mean(dim=0)
        std = chosen.std(dim=0).clamp_min(1e-4)
        index = int(torch.argmin(costs))
        best = candidates[index:index + 1].detach().clone()
        best_cost = float(costs[index])
    tail_costs = cost_fn(tail)
    if tail_costs.shape != (1,) or not torch.isfinite(tail_costs).all():
        raise ValueError("invalid carried-tail cost")
    tail_cost = float(tail_costs[0])
    # Batch-size dependent floating point noise must not be reported as a
    # plan replacement when both parameter witnesses are exactly identical.
    identical_parameters = torch.equal(tail, best)
    selected_old_tail = not identical_parameters and tail_cost < best_cost
    result = tail.detach().clone() if selected_old_tail else best
    return result, {
        "old_tail_model_costs": [tail_cost],
        "iteration_best_model_costs": [best_cost],
        "selected_old_tail": selected_old_tail,
        "tail_equals_cem_parameters": identical_parameters,
        "original_best_plan_params": best.cpu().tolist(),
        "baseline_model_cost": best_cost,
        "tail_model_cost": tail_cost,
        "forward_rollout_evaluations": samples * steps + 1,
        "model_predict_sample_evaluations": (samples * steps + 1) * tail.shape[1],
        "backward_vjp_rows": 0,
        "candidate_policy": "original CEM population unchanged, plus one separately evaluated tail; ties keep CEM",
    }
