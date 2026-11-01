"""Candidate-local residual / response-swap diagnostic, not a P1 override.

The proposal's Vreach(z_B, g) recomputes BOTH residual and response at B.
A response need not improve for a different branch's residual to be useful.
Here each candidate's residual stays fixed while its response is replaced by
each control response, separating a change in geometry from passive drift.
This finite local counterfactual is not a physical controllability theorem.
"""
from __future__ import annotations

import math

import torch

from experiments.spectral_reachability import bounded_response_qp as qp
from experiments.spectral_reachability import reacher_d2p_geometry_dispatch as geometry

POLICY = "candidate_local_residual_certified_response_swap_diagnostic_v1"


def response_swap_metrics(responses, endpoints, goal, *, greedy, damping=.01):
    required = set(responses)
    if not {0, greedy} <= required or required != set(endpoints):
        raise ValueError("complete matching response/endpoints and both controls required")
    if goal.ndim != 1 or not torch.isfinite(goal).all():
        raise ValueError("finite goal coordinate required")
    for index in required:
        response, endpoint = responses[index], endpoints[index]
        if (response.ndim != 2 or response.shape[0] != goal.numel() or response.shape[1] < 1
                or endpoint.shape != goal.shape or response.device != goal.device or endpoint.device != goal.device
                or response.dtype != goal.dtype or endpoint.dtype != goal.dtype or goal.dtype != torch.float64
                or not torch.isfinite(response).all() or not torch.isfinite(endpoint).all()):
            raise ValueError("finite aligned float64 responses/endpoints/goal required")
    if len({tuple(response.shape) for response in responses.values()}) != 1:
        raise ValueError("identical action coordinate dimensions required for response swapping")
    result = {}
    for index in sorted(required):
        residual = goal - endpoints[index]
        # Every cell in this triple uses EXACTLY this candidate's residual.
        cells = {name: qp.certified_response_gain(responses[other], residual, damping=damping)
                 for name, other in (("own", index), ("zero_response", 0), ("greedy_response", greedy))}
        threshold = max(cells[name]["bounded_quadratic_gain_upper_bound"] for name in ("zero_response", "greedy_response"))
        result[index] = {"residual": residual.tolist(), "response_swap": cells,
            "gain_lower_above_control_uppers": cells["own"]["bounded_quadratic_gain"] - threshold,
            "physical_labels_used": False}
    return result


def select_goal_local(receipt, metrics):
    greedy, winners = geometry.endpoint_winners(receipt)
    required = set(winners) | {0, greedy}
    if set(metrics) != required:
        raise ValueError("every endpoint winner and both controls required")
    for index, metric in metrics.items():
        cells = metric["response_swap"]
        if set(cells) != {"own", "zero_response", "greedy_response"} or metric.get("physical_labels_used") is not False:
            raise ValueError("complete model-only response swap required")
        baseline = None
        for cell in cells.values():
            lo, hi = cell["bounded_quadratic_gain"], cell["bounded_quadratic_gain_upper_bound"]
            norm = cell["baseline_squared_residual"]
            if (not cell.get("objective_certificate_passed") or not all(math.isfinite(x) for x in (lo, hi, norm))
                    or hi < lo or norm < 0):
                raise ValueError("finite certified gain intervals required")
            if baseline is not None and norm != baseline:
                raise ValueError("response swap must hold candidate residual fixed")
            baseline = norm
        threshold = max(cells[name]["bounded_quadratic_gain_upper_bound"] for name in ("zero_response", "greedy_response"))
        if metric["gain_lower_above_control_uppers"] != cells["own"]["bounded_quadratic_gain"] - threshold:
            raise ValueError("stored response swap advantage cannot be reproduced")
    qualified = [index for index in winners if metrics[index]["gain_lower_above_control_uppers"] > 0]
    chosen = min(qualified, key=lambda index: (receipt["proposal_rows"][index]["refined_goal_cost"], index)) if qualified else greedy
    return {"policy": POLICY, "selected_program_index": chosen, "same_bank_greedy_index": greedy,
        "endpoint_winner_indices": winners, "geometry_required_indices": sorted(required),
        "geometry_qualified_indices": qualified, "reconfiguration_witness_found": bool(qualified),
        "route": "candidate_local_response_swap" if qualified else "same_bank_goal_progress",
        "physical_labels_used": False, "objective_comparison_certified": True}
