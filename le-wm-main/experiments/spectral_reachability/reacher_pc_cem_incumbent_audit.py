"""CEM replay retaining the best sample without changing the sampling stream."""
from __future__ import annotations

import torch


def cem_incumbent(cost_fn, *, horizon, action_dim, device, samples, steps, elite, seed):
    if not 0 < elite < samples or steps < 1 or horizon < 1:
        raise ValueError("invalid CEM configuration")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    mean = torch.zeros((horizon, action_dim), device=device)
    std = torch.ones_like(mean)
    best_plan = None
    best_cost = float("inf")
    best_iteration = -1
    trace = []
    final_plan = None
    final_cost = None
    for iteration in range(steps):
        candidates = torch.randn((samples, horizon, action_dim), generator=generator, device=device)
        candidates = candidates * std.unsqueeze(0) + mean.unsqueeze(0)
        candidates[0] = mean
        costs = cost_fn(candidates)
        if costs.shape != (samples,) or not torch.isfinite(costs).all():
            raise ValueError("invalid CEM costs")
        values, indices = torch.topk(costs, k=elite, largest=False)
        chosen = candidates[indices]
        mean = chosen.mean(dim=0)
        std = chosen.std(dim=0).clamp_min(1.0e-4)
        idx = int(torch.argmin(costs))
        current_cost = float(costs[idx])
        trace.append(current_cost)
        final_plan = candidates[idx:idx + 1].detach().clone()
        final_cost = current_cost
        if current_cost < best_cost:
            best_plan = final_plan.clone()
            best_cost = current_cost
            best_iteration = iteration
    return best_plan, final_plan, {
        "best_iteration": best_iteration,
        "best_cost": best_cost,
        "final_iteration_cost": final_cost,
        "iteration_best_costs": trace,
        "forward_rollout_evaluations": samples * steps,
        "predict_samples_horizon_weighted": samples * steps * horizon,
    }
