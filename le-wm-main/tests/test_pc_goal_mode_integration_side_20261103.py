from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import pc_goal_mode_integration_side_20261103 as exp
from experiments.spectral_reachability.d2p_feedback_controller import compiler_plan


def test_task_rank_avoids_irrelevant_large_gain():
    args = (np.diag([8., 1.]), [0., 1.], np.eye(2), np.eye(2), .01, 1)
    task, gain = exp.geometry(*args, "task"), exp.geometry(*args, "gain")
    assert task["selected_mode_indices"] == [1]
    assert gain["selected_mode_indices"] == [0]
    assert task["linear_reach_value"] < gain["linear_reach_value"]
    np.testing.assert_allclose(task["raw_delta"], [0., 1 / 1.01])


def test_full_modes_equal_existing_weighted_ridge():
    g = np.array([[1., 2.], [-1., .5], [2., 0.]])
    args = (g, [3., -2., 1.], np.diag([2., .3, 1.]), np.diag([.4, 3.]), .1)
    original = exp.local.local_geometry(*args)
    ranked = exp.geometry(*args, 2, "task")
    np.testing.assert_allclose(ranked["raw_delta"], original["raw_delta"], atol=1e-12)
    np.testing.assert_allclose(ranked["linear_reach_value"], original["linear_reach_value"], atol=1e-12)


def test_zero_response_no_arbitrary_modes():
    result = exp.geometry(np.zeros((2, 3)), [1., 2.], np.eye(2), np.eye(3), .1, 2, "task")
    assert result["selected_mode_indices"] == []
    assert result["raw_spectral_directions"] == []
    np.testing.assert_array_equal(result["raw_delta"], np.zeros(3))
    assert result["linear_reach_value"] == 2.5


@pytest.mark.parametrize("rule", ["gain", "task"])
def test_atlas_and_feedback_adapter_use_complete_selected_witness(rule):
    current = deepcopy(exp.INITIAL)
    saved = deepcopy(current)
    compiler = exp.make_compiler([.2, .8], 0, rule)
    result = compiler.compile(current)
    assert current == saved
    assert result["svd_calls"] == len(result["mode_trace"]) > 0
    assert result["operation_counts"]["jacobian_returned"] == result["svd_calls"]
    for program in result["programs"]:
        actions = np.asarray(program["actions"])
        assert np.all(actions >= compiler.low) and np.all(actions <= compiler.high)
        direct = exp.unicycle.UnicycleBackend().rollout(current, actions)
        assert direct == program["output_context"]
    adapter = compiler_plan(exp.make_compiler([.2, .8], 0, rule), current)
    chosen = result["programs"][result["selected_index"]]
    np.testing.assert_array_equal(adapter.actions, chosen["actions"])
    assert adapter.prediction["selected_program"]["actions"] == chosen["actions"]


def test_joint_saved_result_audit():
    result = exp.audit_joint()
    assert result["replayed_selected_chains"] == 192
    assert result["max_distance_discrepancy"] <= 1e-12
    assert all(x["limited_witnesses"] == 0 for x in result["summary"].values())


def test_invalid_policy_and_dimensions_rejected():
    with pytest.raises(ValueError):
        exp.geometry(np.eye(2), [1., 1.], np.eye(2), np.eye(2), .1, 1, "outcome")
    with pytest.raises(ValueError):
        exp.geometry(np.eye(2), [1.], np.eye(2), np.eye(2), .1, 1, "task")


def test_saved_panel_binding_actions_and_accounting():
    report = json.loads(exp.OUTPUT.read_text())
    for path, digest in report["dependencies"].items():
        assert exp.sha(Path(path)) == digest
    assert len(report["rows"]) == 16
    for row in report["rows"]:
        assert set(row["arms"]) == {"gain", "task"}
        for arm in row["arms"].values():
            chosen = arm["programs"][arm["selected_index"]]
            distance, _ = exp.unicycle.evaluate(chosen["actions"], row["goal"])
            assert abs(distance - arm["terminal_distance"]) < 1e-12
            counts = arm["operation_counts"]
            assert counts["jacobian_returned"] == arm["svd_calls"]
            assert arm["backend_action_steps"]["jacobian_action_steps"] >= arm["svd_calls"]
            assert arm["backend_action_steps"]["rollout_action_steps"] > 0
    assert report["full_cost_certified"] is False
    assert report["P1_admission"] is False
