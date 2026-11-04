import numpy as np

from experiments.spectral_reachability.pc_spectral_candidate_ablation_20261103 import run_arm


def test_arms_have_same_candidate_count_and_shared_controls():
    arms = [run_arm(0, source) for source in ("spectral", "random", "coordinate_lookahead")]
    assert all(len(arm["candidates"]) == 6 for arm in arms)
    for i in (0, 1):
        assert arms[0]["candidates"][i] == arms[1]["candidates"][i] == arms[2]["candidates"][i]
    for arm in arms:
        selected = arm["candidates"][arm["selected_index"]]
        assert selected["terminal_cost"] == min(candidate["terminal_cost"] for candidate in arm["candidates"])
        assert arm["operation_counts"]["rollout_attempted"] == arm["operation_counts"]["rollout_returned"]
        assert np.isclose(selected["terminal_cost"], selected["terminal_position_error"] ** 2 / 3)
    assert arms[0]["operation_counts"]["jacobian_attempted"] == arms[1]["operation_counts"]["jacobian_attempted"] + 1


def test_random_arm_is_deterministic():
    first = run_arm(2, "random")
    second = run_arm(2, "random")
    assert first["candidates"] == second["candidates"]
    assert first["selected_index"] == second["selected_index"]
