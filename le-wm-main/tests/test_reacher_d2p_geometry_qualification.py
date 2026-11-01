import copy

import numpy as np
import pytest

from experiments.spectral_reachability import reacher_d2p_geometry_dispatch as compiler
from experiments.spectral_reachability.audit_reacher_d2p_geometry_qualification import fixed_decisions


def report():
    rows = [{"proposal_index": 0, "immediate_goal_cost": 1., "refined_goal_cost": .5},
        {"proposal_index": 1, "immediate_goal_cost": .2, "refined_goal_cost": .3},
        {"proposal_index": 2, "immediate_goal_cost": 1.2, "refined_goal_cost": .1}]
    endpoint = compiler.base.select_refined_bank(rows)
    receipt = {"proposal_rows": rows, "non_greedy_margin": 1e-6, "dispatch": endpoint}
    geometry = {index: {"bounded_quadratic_gain": gain} for index, gain in enumerate([.5, .4, .2])}
    qualified = compiler.qualify_decision(receipt, geometry)
    receipt.update({"endpoint_only_dispatch": endpoint, "dispatch": qualified,
        "common_task_response": {"candidate_metrics": {str(key): value for key, value in geometry.items()}}})
    actions = np.zeros((15, 2)).tolist()
    row = {"seed": 1, "compilation": receipt, "methods": {name: {"action_witness_raw": actions,
        "first_action_witness_raw": actions[:5]} for name in ("dispatch", "same_bank_greedy", "zero")}}
    return {"schema": "reacher_d2p_full_bank_refined_dispatch.v1", "status": "completed", "gates": {"checked": True},
        "sealed_protocol": {"root_seeds": [1], "compiler_variant": "geometry"}, "rows": [row]}


def test_fixed_sealed_decisions_reproduced():
    assert fixed_decisions(report()) == {1: {"dispatch": 1, "endpoint_only": 2, "same_bank_greedy": 1}}


def test_modified_decision_rejected():
    value = report()
    value["rows"][0]["compilation"]["dispatch"]["selected_program_index"] = 2
    with pytest.raises(ValueError, match="reproduced"):
        fixed_decisions(value)


def test_identity_pairing_not_array_order():
    value = report()
    second = copy.deepcopy(value["rows"][0])
    second["seed"] = 2
    value["rows"].insert(0, second)
    value["sealed_protocol"]["root_seeds"] = [1, 2]
    assert set(fixed_decisions(value)) == {1, 2}


def test_wrong_variant_rejected():
    value = report()
    value["sealed_protocol"]["compiler_variant"] = "refined"
    with pytest.raises(ValueError, match="geometry-qualified"):
        fixed_decisions(value)
