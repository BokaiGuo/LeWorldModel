"""Explicit proposal-native stage + own-state reachability objective.

Not a replacement for the historical common-residual diagnostic or P1 gates.
All costs use mean squared latent units: V_reach is divided by latent width.
The chosen program minimizes prefix_cost + V_reach / width (beta=1).
Numerical certificates are not physical controllability certificates.
"""
from __future__ import annotations

import math


CONTRACT = {
    "schema": "reacher_d2p_proposal_native_selection.v1",
    "stage_cost": "predicted prefix mean squared goal distance",
    "reach_cost": "own-state bounded quadratic objective divided by 192",
    "beta": 1., "latent_width": 192, "damping": .01,
    "selector": "minimum stage plus reach cost; ties by original prefix index",
    "witness": "selected prefix is non-greedy versus zero, terminal better than same-bank greedy, and certified own reach and total scores better than zero and greedy",
    "non_greedy_margin": 1e-6, "terminal_margin": 1e-6,
    "physical_labels_used": False, "P1_admission": False,
}


def select(bank, variant):
    if len(bank) != 17 or [b["index"] for b in bank] != list(range(17)):
        raise ValueError("all 17 original ordered model programs required")
    upper, lower, objectives, gaps, costs = [], [], [], [], []
    for b in bank:
        g = b["own_response_gain"]
        certificate = g["qp_certificate"]
        objective = float(g["bounded_quadratic_objective"])
        gap = float(certificate["objective_duality_gap_upper_bound"])
        stage = float(b["prefix_cost"])
        terminal = float(b["variants"][variant]["terminal_cost"])
        baseline = float(g["baseline_squared_residual"])
        if (g.get("objective_certificate_passed") is not True or g.get("physical_labels_used") is not False
                or certificate.get("active_set_terminated") is not True
                or certificate.get("box_feasibility_violation") != 0
                or not str(certificate.get("device", "")).startswith("cuda")
                or any(not math.isfinite(x) or x < 0 for x in (objective, gap, stage, terminal, baseline))
                or gap > 1e-8 * max(1., baseline) or gap > objective + 1e-8):
            raise ValueError("finite own-state CUDA-certified model-only objective required")
        objectives.append(objective)
        gaps.append(gap)
        costs.append(terminal)
        upper.append(stage + objective / 192.)
        lower.append(stage + max(0., objective-gap) / 192.)
    native = min(range(17), key=lambda i: (upper[i], i))
    greedy = min(range(17), key=lambda i: (bank[i]["prefix_cost"], i))
    controls = (0, greedy)
    witness = (native != 0 and bank[native]["prefix_cost"] > bank[0]["prefix_cost"]+1e-6
        and costs[native] < costs[greedy]-1e-6
        and objectives[native] < min(max(0., objectives[i]-gaps[i]) for i in controls)
        and upper[native] < min(lower[i] for i in controls))
    return {"native_index": native, "greedy_index": greedy,
        "total_score_upper": upper, "total_score_lower": lower,
        "reconfiguration_witness_found": bool(witness), "physical_labels_used": False,
        "P1_admission": False, "contract": CONTRACT["schema"]}
