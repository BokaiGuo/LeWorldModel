"""A single supplied task metric for spectral initialization and GN suffix."""
from __future__ import annotations

import torch

from experiments.spectral_reachability import preflight_pusht_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import pusht_d2p_full_metric_suffix as solver


def refine_in_metric(terminal, goal, initial, projection):
    if (goal.device.type != "cuda" or initial.device.type != "cuda" or projection.device.type != "cuda"
            or projection.shape != (goal.numel(), goal.numel()) or not torch.isfinite(projection).all()):
        raise ValueError("finite square metric and CUDA-only goal/action tensors required")
    return solver.refine(lambda value: projection@terminal(value), projection@goal, initial,
                         damping=.01, iterations=4)


def solve(model, emb, actions, goal, projection):
    if emb.device.type != "cuda" or actions.device.type != "cuda":
        raise RuntimeError("CPU neural fallback forbidden")
    linear = p1._reachability(model, goal, emb, actions, horizon=1, damping=.01, projection=projection)
    initial = linear.pop("dls_params")[0]
    terminal = lambda value: p1._suffix_terminal(model, emb, actions, value[None])[0]
    params, nonlinear = refine_in_metric(terminal, goal, initial, projection)
    with torch.no_grad():
        predicted = terminal(params)
        cost = (projection@(predicted-goal)).square().mean()
    return torch.tanh(params).detach().cpu().numpy(), {
        "linear": linear, "linear_legacy_raw_cost_fields_not_used_for_selection": True,
        "nonlinear": nonlinear, "metric_same_for_spectrum_and_GN": True,
        "predicted_terminal_vector": predicted.cpu().tolist(), "predicted_metric_cost": float(cost),
        "raw_predicted_goal_cost": float((predicted-goal).square().mean()),
        "physical_labels_used": False}
