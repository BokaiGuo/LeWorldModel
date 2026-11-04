"""Fixed-budget suffix CEM; a caller supplies the Q-weighted model objective."""
from __future__ import annotations

import torch


def optimize(evaluate, *, horizon, action_dim, device, seed,
             samples=16, steps=2, elite=4):
    """Return best final sample, consistent with the established Reacher CEM."""
    if (not callable(evaluate) or not all(isinstance(x, int) for x in
            (horizon, action_dim, seed, samples, steps, elite))
            or horizon < 1 or action_dim < 1 or samples < 2 or steps < 1
            or not 2 <= elite < samples):
        raise ValueError("invalid CEM configuration")
    generator = torch.Generator(device=device).manual_seed(seed)
    mean = torch.zeros((horizon, action_dim), dtype=torch.float32, device=device)
    std = torch.ones_like(mean)
    trace = []
    for _ in range(steps):
        candidates = torch.randn((samples, horizon, action_dim), generator=generator,
                                 device=device) * std.unsqueeze(0) + mean.unsqueeze(0)
        candidates[0] = mean
        costs = evaluate(candidates)
        if costs.shape != (samples,) or not torch.isfinite(costs).all():
            raise ValueError("objective must return finite per-sample costs")
        values, indices = torch.topk(costs, k=elite, largest=False)
        selected = candidates[indices]
        mean = selected.mean(dim=0)
        std = selected.std(dim=0, correction=1).clamp_min(1e-4)
        trace.append(float(values.mean()))
    best = int(torch.argmin(costs))
    return candidates[best:best + 1].detach(), {
        "trace": trace, "forward_rollout_evaluations": samples * steps,
        "backward_vjp_rows": 0, "final_selection": "best_sample",
        "best_final_cost": float(costs[best]), "seed": seed,
    }
