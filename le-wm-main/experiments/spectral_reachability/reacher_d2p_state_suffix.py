"""Same numerical suffix contract for imagined and observed RGB latents.

The historical six-candidate optimizer is left immutable. This state-native
implementation is checked against its exact commands/costs on real roots.
Optional carried commands stay in the physical box, including exact +/-1;
no atanh clipping or hidden zero restart is used to transport the plan.
"""
from __future__ import annotations

import math

import torch

from experiments.spectral_reachability import reacher_d2p_suffix_multistart as historical
from experiments.spectral_reachability import reacher_d2p_box_action_suffix as box
from experiments.spectral_reachability import reacher_d2p_iterated_suffix as solver
from experiments.spectral_reachability import reacher_d2p_suffix_step_rescue as rescue
from experiments.spectral_reachability import reacher_d2p_cem_incumbent as cem
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1


POLICY = "state_native_two_chart_cem_exact_command_continuation_v1"
CARRY_NAMES = ("carry", "carry_box_8", "carry_box_32")


def choose_state_candidate(costs):
    names = set(costs)
    if names not in (set(historical.NAMES), set(historical.NAMES) | set(CARRY_NAMES)):
        raise ValueError("complete six-candidate bank, optionally all three carried variants required")
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in costs.values()):
        raise ValueError("finite nonnegative model costs required")
    return min(costs, key=lambda name: (costs[name], name != "incumbent", name))


def state_costs(model, goal_latent, emb_context, action_context, params, *, action_lower, action_upper):
    """Batch the same frozen suffix recurrence without inventing RGB frames."""
    if params.ndim != 3 or emb_context.shape[0] != 1 or action_context.shape[0] != 1:
        raise ValueError("singleton cached state and batched suffix parameters required")
    n = params.shape[0]
    terminal = p1._terminal_from_state(model, emb_context.expand(n, -1, -1), action_context.expand(n, -1, -1),
        params, action_lower=action_lower, action_upper=action_upper)
    predicted = p0._planning_coordinate(model, terminal, role="predicted")
    goal = p0._planning_coordinate(model, goal_latent.unsqueeze(0), role="goal")
    return (predicted - goal).square().mean(dim=1)


def compile_state_candidates(model, goal_latent, emb_context, action_context, *, reference_params,
                             action_lower, action_upper, seed, planned_commands=None):
    if planned_commands is not None:
        if (not isinstance(planned_commands, torch.Tensor) or planned_commands.shape != reference_params.shape
                or planned_commands.device != emb_context.device or planned_commands.dtype != emb_context.dtype
                or not torch.isfinite(planned_commands).all() or bool((planned_commands.abs() > 1).any())):
            raise ValueError("finite exact-shape/device/dtype carried commands in [-1,1] required")
    legacy = solver.compile_snapshots(model, goal_latent, emb_context, action_context, future_horizon=2,
        action_lower=action_lower, action_upper=action_upper, damping=.01, checkpoints=(8, 32),
        accept_update=rescue.accept_with_fine_rescue, initial_params=reference_params)
    boxed = box.compile_box_snapshots(model, goal_latent, emb_context, action_context,
        initial_commands=reference_params.detach().tanh(), action_lower=action_lower, action_upper=action_upper, checkpoints=(8, 32))
    restart, _, cem_receipt = cem.search_with_incumbent(lambda params: state_costs(model, goal_latent,
        emb_context, action_context, params, action_lower=action_lower, action_upper=action_upper),
        future_horizon=2, action_dim=action_context.shape[-1], device=action_context.device,
        samples=128, steps=3, elite=16, seed=seed)
    goal_coordinate = p0._planning_coordinate(model, goal_latent, role="goal")
    commands = {"incumbent": reference_params.detach().tanh(), "same_prefix_cem": restart.detach().tanh()}
    costs = {}
    with torch.no_grad():
        for name, params in (("incumbent", reference_params), ("same_prefix_cem", restart)):
            terminal = p1._terminal_from_state(model, emb_context, action_context, params,
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
    samples = legacy[32][1]["model_predict_sample_evaluations"] + boxed[32][1]["model_predict_sample_evaluations"] + 768 + 4
    vjp = legacy[32][1]["backward_vjp_rows"] + boxed[32][1]["backward_vjp_rows"]
    if planned_commands is not None:
        commands["carry"] = planned_commands.detach().clone()
        with torch.no_grad():
            terminal = box.terminal_commands(model, emb_context, action_context, commands["carry"],
                action_lower=action_lower, action_upper=action_upper)[0]
            costs["carry"] = float((p0._planning_coordinate(model, terminal, role="predicted") - goal_coordinate).square().mean())
        warm = box.compile_box_snapshots(model, goal_latent, emb_context, action_context,
            initial_commands=commands["carry"], action_lower=action_lower, action_upper=action_upper, checkpoints=(8, 32))
        for step in (8, 32):
            key = f"carry_box_{step}"
            commands[key], receipts[key] = warm[step]
            costs[key] = receipts[key]["trace"][-1]["cost_after"]
        samples += 2 + warm[32][1]["model_predict_sample_evaluations"]
        vjp += warm[32][1]["backward_vjp_rows"]
    selected = choose_state_candidate(costs)
    if any(not torch.isfinite(value).all() or bool((value.abs() > 1 + 1e-6).any()) for value in commands.values()):
        raise RuntimeError("finite bounded command witnesses required")
    return commands, {"policy": POLICY, "selected": selected, "model_costs": costs, "receipts": receipts,
        "model_predict_sample_evaluations": samples, "backward_vjp_rows": vjp,
        "physical_labels_used": False, "trained_modules": False,
        "scope": "Numerical score/execution consistency; no physical guarantee or P1 admission."}
