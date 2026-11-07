import numpy as np
import pytest

from experiments.pc_tworoom_retention_counterfactual_pool_20261107_codex import candidate_pool, matched_decisions
from experiments.pc_tworoom_retention_counterfactual_pool_analysis_20261107_codex import classify


def decision(iteration, actions, dispatched, remaining=15):
    return {"iteration": iteration, "remaining": remaining, "actions": actions,
            "dispatched": dispatched, "candidate_seed": 1234 + iteration}


def test_candidate_pool_binds_each_prefix_to_common_tail():
    prefix, chains = candidate_pool({"remaining": 15, "candidate_seed": 1234}, None)
    assert prefix == 10
    assert chains.shape == (6, 15, 2)
    for i in range(3):
        np.testing.assert_array_equal(chains[2*i, :prefix], chains[2*i+1, :prefix])
        for j in range(2):
            np.testing.assert_array_equal(chains[2*i+j, prefix:], chains[j, prefix:])


def test_matching_stops_when_executed_actions_diverge():
    a = [decision(0, [1, 2], [1]), decision(1, [3], [3])]
    b = [decision(0, [1, 2], [1]), decision(1, [4], [4])]
    row = {"arms": {"lookahead": {"decisions": a}, "effect": {"decisions": b}}}
    result = list(matched_decisions(row))
    assert len(result) == 2
    assert result[1][2:] == ([1], [2])


def test_matching_rejects_mismatched_seed():
    a = decision(0, [1], [1])
    b = {**a, "candidate_seed": 999}
    with pytest.raises(ValueError):
        list(matched_decisions({"arms": {"lookahead": {"decisions": [a]}, "effect": {"decisions": [b]}}}))


def test_tail_success_excludes_prefix_and_initial_success():
    fail = {"success": False}
    success = {"success": True}
    entry = {"prefix": 2, "initial_success": False,
             "retained": {"lookahead": [0, 1], "effect": [0, 2]},
             "physical": [[fail, fail, success], [fail, success], [fail] * 3, [fail] * 3,
                          [fail, fail, success], [fail] * 3]}
    result = classify(entry)
    assert result["tail_after_no_prefix_success"] == {"pool": True, "lookahead": True, "effect": True}
    entry["initial_success"] = True
    assert not any(classify(entry)["tail_after_no_prefix_success"].values())
