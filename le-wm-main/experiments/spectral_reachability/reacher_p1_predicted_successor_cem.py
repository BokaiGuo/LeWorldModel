"""Development-only CEM at a predicted successor; no stage admission."""

from __future__ import annotations

from typing import Any

import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


def cem_from_predicted_successor(
    model: torch.nn.Module,
    goal_latent: torch.Tensor,
    emb_context: torch.Tensor,
    action_context: torch.Tensor,
    *,
    future_horizon: int,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
    samples: int,
    steps: int,
    elite: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Use the P0 CEM sampling rule on a single predicted latent context.

    This only evaluates an imagined successor. Deployment must independently
    re-observe RGB and replan, and every invocation must be charged.
    """
    if emb_context.ndim != 3 or action_context.ndim != 3:
        raise ValueError("successor contexts must be rank-three tensors")
    if emb_context.shape[0] != 1 or action_context.shape[0] != 1:
        raise ValueError("exactly one successor context is required")
    if emb_context.shape[1] != action_context.shape[1] or emb_context.shape[1] < 2:
        raise ValueError("successor embedding and action histories must align")
    if min(future_horizon, samples, steps) < 1 or elite < 2 or elite >= samples:
        raise ValueError("invalid CEM budget")
    if not torch.isfinite(emb_context).all() or not torch.isfinite(action_context).all():
        raise ValueError("successor context must be finite")
    device, dtype = action_context.device, action_context.dtype
    generator = torch.Generator(device=device).manual_seed(int(seed))
    mean = torch.zeros((future_horizon, action_context.shape[-1]), device=device, dtype=dtype)
    std = torch.ones_like(mean)
    goal_coordinate = p0._planning_coordinate(model, goal_latent, role="goal")
    trace = []
    final_candidates = final_costs = None
    with torch.no_grad():
        for _ in range(steps):
            candidates = torch.randn(
                (samples, future_horizon, action_context.shape[-1]),
                generator=generator, device=device, dtype=dtype,
            )
            candidates = candidates * std.unsqueeze(0) + mean.unsqueeze(0)
            candidates[0] = mean
            terminal = p1._terminal_from_state(
                model, emb_context.expand(samples, -1, -1),
                action_context.expand(samples, -1, -1), candidates,
                action_lower=action_lower, action_upper=action_upper,
            )
            coordinate = p0._planning_coordinate(model, terminal, role="predicted")
            costs = (coordinate - goal_coordinate).square().mean(dim=1)
            if costs.shape != (samples,) or not torch.isfinite(costs).all():
                raise RuntimeError("CEM candidate costs must be finite")
            values, indices = torch.topk(costs, k=elite, largest=False)
            chosen = candidates[indices]
            mean = chosen.mean(dim=0)
            std = chosen.std(dim=0).clamp_min(1.0e-4)
            trace.append(float(values.mean()))
            final_candidates, final_costs = candidates, costs
    best = int(torch.argmin(final_costs))
    return final_candidates[best:best + 1].detach(), {
        "trace": trace,
        "best_cost": float(final_costs[best]),
        "forward_rollout_evaluations": samples * steps,
        "model_predict_sample_evaluations": samples * steps * future_horizon,
        "backward_vjp_rows": 0,
        "final_selection": "best_sample",
        "scope": "predicted-successor CEM only; excludes prefix, encoder and physical costs",
    }


def select_prefix_with_cem_suffix(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    goal: torch.Tensor,
    history: torch.Tensor,
    bank: torch.Tensor,
    *,
    future_horizon: int,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
    samples: int,
    steps: int,
    elite: int,
    seed: int,
    non_greedy_margin: float,
    future_margin: float,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Development selector with identical CEM budget at every successor.

    The first bank member is the neutral-parameter control. The selector uses
    frozen model outputs only; simulator state and outcomes are never inputs.
    Non-eligible roots return None, not an outcome-selected replacement.
    """
    if bank.ndim != 3 or bank.shape[0] < 2 or bank.shape[1] < 1:
        raise ValueError("candidate bank must contain at least two action programs")
    if bank.shape[2] != history.shape[-1] or not torch.isfinite(bank).all():
        raise ValueError("candidate bank has invalid action coordinates")
    if torch.count_nonzero(bank[0]) != 0:
        raise ValueError("first candidate must be neutral parameters")
    if non_greedy_margin < 0 or future_margin < 0:
        raise ValueError("margins must be non-negative")
    with torch.no_grad():
        goal_latent = p0._goal_latent(model, goal)
        immediate = p0._cost_fn(
            model, pixels, goal, history, bank,
            action_lower=action_lower, action_upper=action_upper,
        )
        if immediate.shape != (len(bank),) or not torch.isfinite(immediate).all():
            raise RuntimeError("candidate immediate costs must be finite")
        rows = []
        for index in range(len(bank)):
            embeddings, actions = p1._state_after_prefix(
                model, pixels, history, bank[index:index + 1],
                action_lower=action_lower, action_upper=action_upper,
            )
            suffix, receipt = cem_from_predicted_successor(
                model, goal_latent, embeddings, actions,
                future_horizon=future_horizon, action_lower=action_lower,
                action_upper=action_upper, samples=samples, steps=steps,
                elite=elite, seed=seed,
            )
            if suffix.shape != (1, future_horizon, bank.shape[2]) or not torch.isfinite(suffix).all():
                raise RuntimeError("CEM suffix witness has invalid shape or values")
            rows.append({
                "proposal_index": index,
                "immediate_goal_cost": float(immediate[index]),
                "cem_future_goal_cost": receipt["best_cost"],
                "cem_receipt": receipt,
                # These are unconstrained planner parameters, not raw env actions.
                "suffix_parameters": suffix[0].detach().cpu().tolist(),
            })
    zero = rows[0]
    eligible = [
        row for row in rows[1:]
        if row["immediate_goal_cost"] >= zero["immediate_goal_cost"] + non_greedy_margin
        and row["cem_future_goal_cost"] <= zero["cem_future_goal_cost"] - future_margin
    ]
    chosen = min(eligible, key=lambda row: (row["cem_future_goal_cost"], row["proposal_index"])) if eligible else None
    count = len(rows)
    return (None if chosen is None else bank[chosen["proposal_index"]:chosen["proposal_index"] + 1].detach()), {
        "scope": "development model-only prefix selector; not P1 admission or physical evidence",
        "reference": "neutral parameter prefix",
        "cem_seed_policy": "same seed at every candidate",
        "selected_proposal_index": None if chosen is None else chosen["proposal_index"],
        "selected_suffix_parameters": None if chosen is None else chosen["suffix_parameters"],
        "suffix_parameter_contract": "unconstrained CEM parameters; apply the existing bounded action transform",
        "eligible_count": len(eligible),
        "proposal_rows": rows,
        "predict_sample_evaluations": (
            2 * count * bank.shape[1] + count * samples * steps * future_horizon
        ),
        "suffix_cem_calls": count,
        "suffix_cem_rollout_evaluations": count * samples * steps,
        "backward_vjp_rows": 0,
        "cost_scope": "model.predict samples only; excludes encoder, action encoder, wall time and physical execution",
    }
