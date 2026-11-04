import numpy as np
import pytest

from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.pc_goal_mode_matched_feedback_20261103_codex import plan_at_context


def run_arm(arm):
    meter = Meter(132)
    target = np.array([.5, .2])

    def predict(chains):
        def raw(batch):
            position = np.asarray(batch).sum(axis=1) / 5
            return position, np.linalg.norm(position - target, axis=1)
        return meter.predict(np.asarray(chains), raw)

    result = plan_at_context(arm, 0, 0, lambda x: predict(x[None])[0][0],
                             lambda x: predict(x)[1], target, np.linalg.norm(target))
    return result, meter


@pytest.mark.parametrize("arm", ("goal", "greedy", "ridge", "random", "cem"))
def test_per_arm_budget_and_legal_selected_actions(arm):
    result, meter = run_arm(arm)
    assert meter.token_samples == 132
    assert np.asarray(result["selected"]["prefix"]).shape == (5, 2)
    assert np.asarray(result["selected"]["suffix"]).shape == (10, 2)
    if arm == "random":
        assert result["counts"].get("svd", 0) == 0
        assert result["counts"].get("linear_solve", 0) == 0


def test_greedy_uses_goal_bank_on_identical_inputs():
    goal, _ = run_arm("goal")
    greedy, _ = run_arm("greedy")
    assert goal["generator"] == greedy["generator"]
    assert goal["prefix_scores"] == greedy["prefix_scores"]
    assert greedy["selected"]["name"] == greedy["same_bank_greedy_name"]
