"""Model-only numerical suffix candidates; not a new P1 mechanism.

Keep the observed-RGB cold/warm incumbent, refine in two numerical charts,
and search a separate CEM basin. Selection never reads physical outcomes.
This explicitly adds compute and makes no physical monotonicity guarantee.
"""
from __future__ import annotations

import math

import torch

from experiments.spectral_reachability import reacher_d2p_box_action_suffix as box
from experiments.spectral_reachability import reacher_d2p_iterated_suffix as solver
from experiments.spectral_reachability import reacher_d2p_suffix_step_rescue as rescue
from experiments.spectral_reachability import reacher_d2p_cem_incumbent as cem
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1


NAMES = ("incumbent", "tanh_extra_8", "tanh_extra_32", "box_extra_8", "box_extra_32", "same_prefix_cem")


def choose_candidate(costs):
    if set(costs) != set(NAMES) or any(not math.isfinite(float(v)) or float(v) < 0 for v in costs.values()):
        raise ValueError("all six finite nonnegative model costs required")
    return min(costs, key=lambda name: (costs[name], name != "incumbent", name))


def compile_candidates(model, pixels, goal, history, *, incumbent, action_lower, action_upper, seed):
    """Return bounded commands, not unbounded tanh parameters."""
    emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
    # Validation precedes CEM, and the legacy default solver is not modified.
    legacy = solver.compile_snapshots(model, goal_latent, emb, history, future_horizon=2,
        action_lower=action_lower, action_upper=action_upper, damping=.01, checkpoints=(8, 32),
        accept_update=rescue.accept_with_fine_rescue, initial_params=incumbent)
    boxed = box.compile_box_snapshots(model, goal_latent, emb, history, initial_commands=incumbent.tanh().detach(),
        action_lower=action_lower, action_upper=action_upper, checkpoints=(8, 32))
    restart, _, cem_receipt = cem.search_with_incumbent(lambda params: p0._cost_fn(model, pixels, goal, history, params,
        action_lower=action_lower, action_upper=action_upper), future_horizon=2, action_dim=history.shape[-1],
        device=history.device, samples=128, steps=3, elite=16, seed=seed)
    goal_coordinate = p0._planning_coordinate(model, goal_latent, role="goal")
    costs, commands = {}, {"incumbent": incumbent.detach().tanh(), "same_prefix_cem": restart.detach().tanh()}
    with torch.no_grad():
        for name, params in (("incumbent", incumbent), ("same_prefix_cem", restart)):
            terminal = p1._terminal_from_state(model, emb, history, params,
                action_lower=action_lower, action_upper=action_upper)[0]
            costs[name] = float((p0._planning_coordinate(model, terminal, role="predicted") - goal_coordinate).square().mean())
    receipts = {"same_prefix_cem": cem_receipt}
    for step in (8, 32):
        commands[f"tanh_extra_{step}"] = legacy[step][0].detach().tanh()
        commands[f"box_extra_{step}"] = boxed[step][0].detach().clone()
        for prefix, result in (("tanh", legacy), ("box", boxed)):
            key = f"{prefix}_extra_{step}"
            costs[key] = result[step][1]["trace"][-1]["cost_after"]
            receipts[key] = result[step][1]
    selected = choose_candidate(costs)
    # Each search and local refinement is counted once; 8 is a snapshot of32.
    samples = legacy[32][1]["model_predict_sample_evaluations"] + boxed[32][1]["model_predict_sample_evaluations"] + 768 + 4
    if any(not torch.isfinite(value).all() or bool((value.abs() > 1 + 1e-6).any()) for value in commands.values()):
        raise RuntimeError("finite bounded command witnesses required")
    return commands, {"policy": "observed_suffix_two_chart_cem_multistart_v1", "selected": selected, "model_costs": costs,
        "receipts": receipts, "model_predict_sample_evaluations": samples,
        "backward_vjp_rows": legacy[32][1]["backward_vjp_rows"] + boxed[32][1]["backward_vjp_rows"],
        "physical_labels_used": False, "trained_modules": False, "model_cost_not_above_incumbent": costs[selected] <= costs["incumbent"],
        "scope": "Numerical suffix repair, added compute, no physical monotonicity or P1 admission."}
