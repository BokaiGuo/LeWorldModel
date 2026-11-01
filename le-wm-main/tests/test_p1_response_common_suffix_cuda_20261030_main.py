import pytest

from experiments.spectral_reachability import p1_response_common_suffix_cuda_20261030_main as plan


def test_candidate_sets_and_tie_breaking():
    arm = {"slot": 0, "arm": "a",
           "result": {"candidates": [{"stage_Q": i // 2} for i in range(20)]}}
    selection = {"slot": 0, "arm": "a", "selected_indices": [0, 7, 11, 19]}
    assert plan.candidate_sets(arm, selection) == {
        "diverse": [0, 7, 11, 19], "immediate": [0, 1, 2, 3]
    }
    selection["selected_indices"] = [0, 0, 11, 19]
    with pytest.raises(ValueError):
        plan.candidate_sets(arm, selection)


def test_saved_panel_sources_reconstruct():
    report, prepared, selection = plan.inputs()
    assert len(report["rows"]) == len(selection["arms"]) == 8
    assert len(prepared["requests"]) == 4
    for arm, chosen in zip(report["rows"], selection["arms"], strict=True):
        assert len(plan.candidate_sets(arm, chosen)["diverse"]) == 4
