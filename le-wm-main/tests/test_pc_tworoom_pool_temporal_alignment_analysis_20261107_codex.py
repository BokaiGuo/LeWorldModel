from experiments.pc_tworoom_pool_temporal_alignment_analysis_20261107_codex import pool_outcome


def test_temporal_argmin_and_sampled_success_are_separate():
    failure = {"success": False}
    success = {"success": True}
    traces = [[failure] * 10 for _ in range(6)]
    traces[0] = [failure, success] + [failure] * 8
    traces[1] = [failure] * 4 + [success] + [failure] * 5
    row = {"predicted_cost": [[0, 3], [2, 2]] + [[5, 5]] * 4,
           "actual_cost": [[3, 3], [2, 0]] + [[5, 5]] * 4,
           "physical": traces}
    out = pool_outcome(row)
    assert out["predicted"] and out["encoded"] and out["pool"]
    assert not out["predicted_sampled"]
    assert out["encoded_sampled"] and out["pool_sampled"]
    assert out["different_choice"]
    assert not out["encoded_only"] and not out["predicted_only"]
