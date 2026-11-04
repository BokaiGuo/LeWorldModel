"""Development-only bounded nonlinear refinement of spectral prefixes."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


def refine(initial, evaluate, neutral_cost, *, iterations=4, lr=0.08, radius=0.25,
           temperature=0.01, penalty=10.0):
    """Refine one prefix using a caller's differentiable immediate/future costs."""
    if (initial.ndim != 1 or not torch.isfinite(initial).all()
            or iterations < 1 or lr <= 0 or radius <= 0 or temperature <= 0 or penalty < 0):
        raise ValueError("invalid refinement contract")
    variable = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([variable], lr=lr)
    trace = []
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        immediate, future = evaluate(variable)
        if immediate.ndim != 0 or future.ndim != 1 or future.numel() == 0:
            raise ValueError("invalid cost shape")
        if not torch.isfinite(immediate) or not torch.isfinite(future).all():
            raise ValueError("nonfinite costs")
        softmin = -temperature * torch.logsumexp(-future / temperature, dim=0)
        loss = softmin + penalty * F.relu(neutral_cost - immediate).square()
        loss.backward()
        if variable.grad is None or not torch.isfinite(variable.grad).all():
            raise ValueError("invalid prefix gradient")
        optimizer.step()
        with torch.no_grad():
            delta = variable - initial
            variable.copy_(initial + delta * min(1.0, radius / max(float(delta.norm()), 1e-12)))
        trace.append({"loss": float(loss.detach()), "immediate": float(immediate.detach()),
                      "future_costs": future.detach().cpu().tolist()})
    return variable.detach(), trace


def augment(model, pixels, goal, history, bank, *, lower, upper, maximum=4):
    if bank.ndim != 3 or bank.shape[0] < 2 or bank.shape[1:] != (1, 10):
        raise ValueError("invalid prefix packet bank")
    if maximum < 0 or not torch.isfinite(bank).all():
        raise ValueError("invalid candidate bank")
    probes = bank.new_tensor([[0., 0.], [0.3, 0.], [-0.3, 0.],
                              [0., 0.3], [0., -0.3]]).repeat(1, 5).unsqueeze(1)
    goal_coordinate = p0._planning_coordinate(model, p0._goal_latent(model, goal), role="goal").detach()

    def evaluate(flat):
        embeddings, actions = p1._state_after_prefix(
            model, pixels, history, flat.reshape(1, 1, 10),
            action_lower=lower, action_upper=upper,
        )
        immediate_coordinate = p0._planning_coordinate(model, embeddings[:, -1], role="predicted")
        future = p1._terminal_from_state(
            model, embeddings.expand(5, -1, -1), actions.expand(5, -1, -1), probes,
            action_lower=lower, action_upper=upper,
        )
        future_coordinate = p0._planning_coordinate(model, future, role="predicted")
        return ((immediate_coordinate - goal_coordinate).square().mean(),
                (future_coordinate - goal_coordinate).square().mean(dim=1))

    with torch.no_grad():
        neutral_cost, _ = evaluate(bank[0, 0])
        neutral_cost = neutral_cost.detach()
    new, rows = [], []
    for idx in range(2, min(len(bank), 2 + maximum)):
        result, trace = refine(bank[idx, 0], evaluate, neutral_cost)
        new.append(result.reshape(1, 10))
        rows.append({"source_index": idx, "trace": trace,
                     "parameter_displacement_l2": float((result - bank[idx, 0]).norm())})
    combined = torch.cat([bank.detach(), torch.stack(new)], dim=0) if new else bank.detach()
    return combined, {
        "neutral_immediate_cost": float(neutral_cost), "refinements": rows,
        "refinement_forward_prefix_samples": 1 + 4 * len(rows),
        "refinement_forward_suffix_samples": 5 + 20 * len(rows),
        "refinement_backward_calls": 4 * len(rows),
        "scope": "model predictor samples; excludes encoder and wall time",
    }
