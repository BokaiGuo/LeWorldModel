"""Logic fixtures only; live neural and physical evidence is host CUDA."""
import copy

import pytest
import torch

from experiments.spectral_reachability import reacher_d2p_goal_local_geometry as local
from experiments.spectral_reachability import bounded_response_qp as qp
from experiments.spectral_reachability import reacher_d2p_geometry_dispatch as common


def receipt():
    rows = [{"proposal_index": 0, "immediate_goal_cost": 1., "refined_goal_cost": .5},
            {"proposal_index": 1, "immediate_goal_cost": .2, "refined_goal_cost": .3},
            {"proposal_index": 2, "immediate_goal_cost": 1.2, "refined_goal_cost": .1}]
    return {"proposal_rows": rows, "non_greedy_margin": 1e-6, "dispatch": common.base.select_refined_bank(rows)}


def inputs():
    dtype = torch.float64
    # Controls act only along x; the candidate acts along y. Its residual is y,
    # whereas the greedy branch's residual is x. Common gating rejects it.
    responses = {0: torch.tensor([[1.], [0.]], dtype=dtype),
                 1: torch.tensor([[1.], [0.]], dtype=dtype),
                 2: torch.tensor([[0.], [1.]], dtype=dtype)}
    endpoints = {0: torch.tensor([0., 0.], dtype=dtype), 1: torch.tensor([0., 1.], dtype=dtype),
                 2: torch.tensor([1., 0.], dtype=dtype)}
    return responses, endpoints, torch.ones(2, dtype=dtype)


def test_common_condition_is_not_necessary_for_candidate_local_response():
    responses, endpoints, goal = inputs()
    metrics = local.response_swap_metrics(responses, endpoints, goal, greedy=1)
    assert local.select_goal_local(receipt(), metrics)["selected_program_index"] == 2
    shared = {i: qp.certified_response_gain(response, goal - endpoints[1]) for i, response in responses.items()}
    assert qp.qualify_certified_decision(receipt(), shared)["selected_program_index"] == 1
    assert metrics[2]["gain_lower_above_control_uppers"] > 0


def test_geometry_identity_cannot_make_witness_from_residual_change():
    responses, endpoints, goal = inputs()
    responses[2] = responses[0].clone()
    metrics = local.response_swap_metrics(responses, endpoints, goal, greedy=1)
    assert not local.select_goal_local(receipt(), metrics)["reconfiguration_witness_found"]


def test_goal_translation_leaves_decision_unchanged():
    responses, endpoints, goal = inputs()
    shift = torch.tensor([2., -3.], dtype=goal.dtype)
    first = local.response_swap_metrics(responses, endpoints, goal, greedy=1)
    second = local.response_swap_metrics(responses, {i: value + shift for i, value in endpoints.items()}, goal + shift, greedy=1)
    assert local.select_goal_local(receipt(), first) == local.select_goal_local(receipt(), second)
    assert first == second


def test_map_insertion_order_does_not_pair_roots_or_responses():
    responses, endpoints, goal = inputs()
    first = local.response_swap_metrics(responses, endpoints, goal, greedy=1)
    second = local.response_swap_metrics(dict(reversed(list(responses.items()))), endpoints, goal, greedy=1)
    assert first == second


@pytest.mark.parametrize("mutation", ["missing", "extra", "nan", "uncertified", "interval", "residual", "advantage", "labels"])
def test_bad_metrics_fail_closed(mutation):
    responses, endpoints, goal = inputs()
    metrics = copy.deepcopy(local.response_swap_metrics(responses, endpoints, goal, greedy=1))
    if mutation == "missing":
        del metrics[2]
    elif mutation == "extra":
        metrics[3] = metrics[2]
    elif mutation == "nan":
        metrics[2]["response_swap"]["own"]["bounded_quadratic_gain"] = float("nan")
    elif mutation == "uncertified":
        metrics[2]["response_swap"]["own"]["objective_certificate_passed"] = False
    elif mutation == "interval":
        metrics[2]["response_swap"]["own"]["bounded_quadratic_gain_upper_bound"] = -1.
    elif mutation == "residual":
        metrics[2]["response_swap"]["greedy_response"]["baseline_squared_residual"] += 1
    elif mutation == "advantage":
        metrics[2]["gain_lower_above_control_uppers"] += .1
    else:
        metrics[2]["physical_labels_used"] = True
    with pytest.raises(ValueError):
        local.select_goal_local(receipt(), metrics)


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "dtype", "nan", "coordinate"])
def test_bad_tensors_fail_closed(mutation):
    responses, endpoints, goal = inputs()
    if mutation == "missing":
        del responses[0]
    elif mutation == "extra":
        endpoints[3] = endpoints[2]
    elif mutation == "shape":
        endpoints[2] = endpoints[2][:1]
    elif mutation == "dtype":
        responses[2] = responses[2].float()
    elif mutation == "nan":
        goal[0] = float("nan")
    else:
        responses[2] = responses[2].expand(2, 2)
    with pytest.raises(ValueError):
        local.response_swap_metrics(responses, endpoints, goal, greedy=1)
