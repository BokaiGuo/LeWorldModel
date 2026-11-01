"""CEM with unchanged proposal trajectory and a best-ever output incumbent.

Retaining the best visited sample changes only the final output selection.
It does not inject an incumbent into subsequent proposal generations, add
world-model evaluations, or use physical outcomes.
"""
from __future__ import annotations

import torch


def search_with_incumbent(cost_fn, *, future_horizon, action_dim, device, samples, steps, elite, seed):
    if min(future_horizon, action_dim, steps, elite) < 1 or elite < 2 or elite >= samples:
        raise ValueError("positive dimensions/steps and 2 <= elite < samples required")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    mean = torch.zeros((future_horizon, action_dim), device=device)
    std = torch.ones_like(mean)
    best, best_cost, selected_iteration, selected_index = None, None, None, None
    trace, iteration_best_costs = [], []
    final_candidates = final_costs = None
    with torch.no_grad():
        for iteration in range(steps):
            candidates = torch.randn((samples, future_horizon, action_dim), generator=generator, device=device)
            candidates = candidates * std.unsqueeze(0) + mean.unsqueeze(0)
            candidates[0] = mean
            costs = cost_fn(candidates)
            if costs.shape != (samples,) or not torch.isfinite(costs).all():
                raise ValueError("one finite cost per CEM proposal required")
            current_index = int(torch.argmin(costs))
            current_cost = float(costs[current_index])
            if best_cost is None or current_cost < best_cost:
                best, best_cost = candidates[current_index:current_index + 1].detach().clone(), current_cost
                selected_iteration, selected_index = iteration, current_index
            values, indices = torch.topk(costs, k=elite, largest=False)
            chosen = candidates[indices]
            mean = chosen.mean(dim=0)
            std = chosen.std(dim=0).clamp_min(1e-4)
            trace.append(float(values.mean()))
            iteration_best_costs.append(current_cost)
            final_candidates, final_costs = candidates, costs
    legacy_index = int(torch.argmin(final_costs))
    legacy = final_candidates[legacy_index:legacy_index + 1].detach().clone()
    return best, legacy, {"program_type": "unchanged_cem_trajectory_best_ever_output_v1",
        "trace": trace, "iteration_best_costs": iteration_best_costs,
        "selected_iteration": selected_iteration, "selected_sample_index": selected_index,
        "best_ever_goal_cost": best_cost, "legacy_last_iteration_goal_cost": float(final_costs[legacy_index]),
        "forward_rollout_evaluations": int(samples * steps), "backward_vjp_rows": 0,
        "selection_uses_physical_labels": False, "proposal_trajectory_modified": False,
        "final_selection": "best_visited_sample", "legacy_final_selection": "best_last_generation_sample"}
