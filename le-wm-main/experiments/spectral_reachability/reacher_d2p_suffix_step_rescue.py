"""Model-only rescue of a coarse-grid DLS stop, never an outcome selector."""
from __future__ import annotations

import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1


def accept_with_fine_rescue(terminal, update, predicted, goal_coordinate, *, step_rule):
    delta, accepted, receipt = p1._accept_suffix_update(terminal, update, predicted, goal_coordinate, step_rule=step_rule)
    receipt.update({"coarse_stop": receipt["accepted_step_scale"] == 0., "fine_rescue_attempted": False,
        "fine_rescue_accepted": False, "physical_labels_used": False})
    if step_rule != "backtracking" or not receipt["coarse_stop"]:
        return delta, accepted, receipt
    receipt["fine_rescue_attempted"] = True
    scales = tuple(2. ** -exponent for exponent in range(5, 17))
    with torch.no_grad():
        trials = [terminal(update * scale) for scale in scales]
        costs = torch.stack([(trial - goal_coordinate).square().mean() for trial in trials])
        if not torch.isfinite(costs).all():
            raise RuntimeError("non-finite fine backtracking costs")
        before = float((predicted - goal_coordinate).square().mean())
        index = int(costs.argmin())
        receipt["fine_trial_scales"] = list(scales)
        receipt["fine_trial_costs"] = [float(value) for value in costs]
        # Exact objective check, no numerical relaxation or physical labels.
        if float(costs[index]) < before:
            delta, accepted = update.detach() * scales[index], trials[index]
            receipt["accepted_step_scale"] = scales[index]
            receipt["fine_rescue_accepted"] = True
    return delta, accepted, receipt
