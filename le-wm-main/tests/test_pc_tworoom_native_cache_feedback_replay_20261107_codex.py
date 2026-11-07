from copy import deepcopy

import pytest

from experiments.pc_tworoom_native_cache_feedback_replay_20261107_codex import compare


def sample():
    step = {"action": [0.1, 0.2], "distance": 2.0, "terminated": False, "truncated": False}
    decision = {"iteration": 1, "remaining": 5, "winner": "a", "actions": [[0.1, 0.2]],
                "actions_sha256": "a", "old_tail_sha256": "b", "queries": 2, "packet_steps": 2,
                "forward_batches": 1, "propagation_rollouts": 1, "all_scores": [1.0, 2.0], "score": 1.0}
    arm = {"trace": [step], "decisions": [decision], "ever_success": False, "queries": 2,
           "packet_steps": 2, "initial_rgb_sha256": "rgb"}
    return {"track": "rgb", "model_sha256": "model",
            "rows": [{"root": 1, "arms": {"real_rgb": deepcopy(arm), "imagined": deepcopy(arm)}}]}


def test_matching_trace_and_tampering():
    reference = sample()
    changed = deepcopy(reference)
    assert all(not r["errors"] for r in compare(reference, changed))
    changed["rows"][0]["arms"]["imagined"]["decisions"][0]["all_scores"][0] += 0.01
    assert "numeric tolerance" in compare(reference, changed)[1]["errors"]
    changed["rows"][0]["arms"]["real_rgb"]["trace"][0]["action"][0] += 0.1
    assert "physical trace" in compare(reference, changed)[0]["errors"]


def test_root_and_track_fail_closed():
    reference = sample()
    changed = deepcopy(reference)
    changed["rows"].append(deepcopy(changed["rows"][0]))
    with pytest.raises(ValueError):
        compare(reference, changed)
    changed = deepcopy(reference)
    changed["track"] = "state_assisted"
    with pytest.raises(ValueError):
        compare(reference, changed)
