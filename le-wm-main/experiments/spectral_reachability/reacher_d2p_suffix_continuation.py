"""Preserve a compiled suffix as an incumbent across real RGB observations.

Only RGB-derived latent/action contexts and the already planned action witness
are inputs. Both cold and warm starts receive the identical fine8 solver.
Choose by the observed-state nonlinear frozen-model objective, not outcomes.
"""
from __future__ import annotations

import math

from experiments.spectral_reachability import reacher_d2p_iterated_suffix as solver
from experiments.spectral_reachability import reacher_d2p_suffix_step_rescue as rescue


def choose_incumbent(costs):
    if set(costs) != {"cold", "warm"} or any(not math.isfinite(float(value)) for value in costs.values()):
        raise ValueError("finite cold/warm model costs required")
    return min(("cold", "warm"), key=lambda name: (costs[name], name != "cold"))


def compile_continuation(model, goal_latent, emb_context, action_context, *, planned_suffix,
                         action_lower, action_upper, damping=.01):
    # The solver itself validates exact shape/device/dtype and finite values.
    # Warm first makes invalid carried plans fail before doing a cold search.
    warm, warm_receipt = solver.compile_snapshots(model, goal_latent, emb_context, action_context,
        future_horizon=2, action_lower=action_lower, action_upper=action_upper, damping=damping,
        checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue, initial_params=planned_suffix)[8]
    cold, cold_receipt = solver.compile_snapshots(model, goal_latent, emb_context, action_context,
        future_horizon=2, action_lower=action_lower, action_upper=action_upper, damping=damping,
        checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
    costs = {"cold": cold_receipt["trace"][-1]["cost_after"], "warm": warm_receipt["trace"][-1]["cost_after"]}
    chosen = choose_incumbent(costs)
    carry_cost = warm_receipt["trace"][0]["cost_before"]
    if costs[chosen] > carry_cost or costs[chosen] > costs["cold"]:
        raise RuntimeError("model-only suffix incumbent increased reference objective")
    variants = {"cold": cold, "carry": planned_suffix.detach().clone(), "warm": warm}
    return variants, {"policy": "preserve_planned_suffix_cold_warm_fine8_incumbent_v1",
        "selected": chosen, "model_costs": costs, "carry_model_cost": carry_cost,
        "cold_receipt": cold_receipt, "warm_receipt": warm_receipt,
        "model_predict_sample_evaluations": sum(item["model_predict_sample_evaluations"] for item in (cold_receipt, warm_receipt)),
        "backward_vjp_rows": sum(item["backward_vjp_rows"] for item in (cold_receipt, warm_receipt)),
        "physical_labels_used": False, "trained_modules": False,
        "scope": "Model-objective incumbent guarantee only, not a physical guarantee or P1 admission."}
