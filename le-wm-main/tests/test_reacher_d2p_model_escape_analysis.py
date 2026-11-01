import copy

import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_model_escape_cohort import check_task


def fixture_plan():
    costs = dict(zero=1., gn4=.5, gn8=.25, gn16=.25, cem=.125)
    plan = {name: {"cost": cost, "weighted_vector": [np.sqrt(cost)] * 192,
                   "raw_actions": np.zeros((15, 2)).tolist()}
            for name, cost in costs.items()}
    plan["gn_receipts"] = {}
    before = 1.
    for name, count in (("gn4", 4), ("gn8", 4), ("gn16", 8)):
        trace = []
        for i in range(count):
            after = costs[name] if i == 0 else before
            trace.append(dict(iteration=i+1, cost_before=before, cost_after=after,
                              accepted_scale=1., qp_certificate=dict(
                                  active_set_terminated=True, box_feasibility_violation=0,
                                  device="cuda:0", objective_duality_gap_upper_bound=0.)))
            before = after
        plan["gn_receipts"][name] = dict(iterations=count, physical_labels_used=False, trace=trace)
    plan.update(cem_receipt=dict(samples=128, steps=3, elite=16, flat_gn_incumbent=True,
                                singleton_incumbent_guard=True, physical_labels_used=False),
                physical_future_labels_read=False, actual_predict_samples=1554,
                logical_action_vjp_rows=3073, gn16_gradient=[0.] * 15,
                gn16_projected_gradient_inf=0.)
    return plan


def test_independent_plateau_and_escape():
    result = check_task("shoulder_only", fixture_plan(), np.zeros(192))
    assert result["eligible"]
    assert result["normalized_escape"] == pytest.approx(.125)


@pytest.mark.parametrize("corruption", ["mask", "bound", "cost", "trace", "device",
                                        "gradient", "outcome", "budget", "incumbent"])
def test_corrupt_receipts_fail_closed(corruption):
    plan = copy.deepcopy(fixture_plan())
    if corruption == "mask": plan["gn4"]["raw_actions"][0][1] = .1
    if corruption == "bound": plan["gn4"]["raw_actions"][0][0] = 1.1
    if corruption == "cost": plan["cem"]["cost"] += .01
    if corruption == "trace": plan["gn_receipts"]["gn8"]["trace"][0]["cost_before"] = .7
    if corruption == "device": plan["gn_receipts"]["gn4"]["trace"][0]["qp_certificate"]["device"] = "cpu"
    if corruption == "gradient": plan["gn16_gradient"][0] = .5
    if corruption == "outcome": plan["physical_future_labels_read"] = True
    if corruption == "budget": plan["actual_predict_samples"] -= 1
    if corruption == "incumbent": plan["cem_receipt"]["singleton_incumbent_guard"] = False
    with pytest.raises(ValueError):
        check_task("shoulder_only", plan, np.zeros(192))
