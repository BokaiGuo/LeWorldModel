import numpy as np

from experiments.spectral_reachability import pc_response_program_joint_refinement_20261103_codex as exp


def test_same_bank_and_caps_and_incumbent():
    goal = np.array([.3, .7])
    response = exp.plan(goal, 0, 64, "response")
    greedy = exp.plan(goal, 0, 64, "same_bank_greedy")
    assert [x["point"] for x in response["bank"]] == [x["point"] for x in greedy["bank"]]
    assert response["generation_calls"] == 45
    assert greedy["generation_calls"] == 54
    for method in exp.METHODS:
        result = exp.plan(goal, 0, 64, method)
        assert result["total_calls"] <= 64
        assert result["analytic_gradients"] == result["optimizer_objectives"]
        if result["pre_refinement_incumbent"]:
            assert result["terminal_distance"] <= result["pre_refinement_incumbent"]["terminal_distance"] + 1e-12
        replay, _ = exp.dynamics.evaluate(exp.actions(result["point"]), goal)
        np.testing.assert_allclose(result["terminal_distance"], replay, atol=1e-12)
        assert all(lo <= v <= hi for v, (lo, hi) in zip(result["point"], exp.BOUNDS, strict=True))


def test_intervention_uses_post_refinement_prefix_and_controls():
    result = exp.intervention(np.array([0., .9]), [0., np.pi / 2, .9], [.2, 0., .3])
    assert result["controls"]["selected"]["prefix"] == [0., np.pi / 2]
    assert result["controls"]["same_bank_greedy"]["prefix"] == [.2, 0.]
    assert result["controls"]["selected"]["active_change"] < result["controls"]["zero"]["active_change"]
