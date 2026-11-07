from copy import deepcopy

from experiments.pc_tworoom_timed_clock_fixed_gpu_replay_20261107_codex import compare_decision


def pair():
    current = {"remaining_steps": 5, "actions": [[0.1, 0.2]] * 5,
               "choice": {"source": "parent", "index": 0, "cost": 1.0},
               "parent_cost": [1.0, 2.0], "probe_cost": [], "tail_cost": [],
               "receipts": [{"inputs": {"pixels": {"sha256": "a"}}}]}
    previous = deepcopy(current)
    previous["actions"] += [[0.3, 0.4]] * 20
    return current, previous


def test_comparison_recognizes_equivalent_scored_prefix():
    current, previous = pair()
    result = compare_decision(current, previous)
    assert result["chosen_action_equal"]
    assert result["chosen_source_index_equal"]
    assert result["scores_close"]
    assert result["input_receipts_equal"]
    assert result["score_max_abs_error"] == 0.0


def test_comparison_detects_different_actions_scores_and_input():
    current, previous = pair()
    previous["actions"][0] = [0.5, 0.6]
    previous["parent_cost"][0] = 2.0
    previous["choice"]["index"] = 1
    previous["receipts"][0]["inputs"]["pixels"]["sha256"] = "b"
    result = compare_decision(current, previous)
    assert not result["chosen_action_equal"]
    assert not result["chosen_source_index_equal"]
    assert not result["scores_close"]
    assert not result["input_receipts_equal"]
    assert result["score_max_abs_error"] == 1.0
