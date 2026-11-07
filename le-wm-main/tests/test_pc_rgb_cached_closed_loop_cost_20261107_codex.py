from copy import deepcopy

from experiments.pc_rgb_cached_closed_loop_cost_20261107_codex import compare


def example():
    return {"decisions": [{"choice": {"index": 1}, "queries": 16, "packet_steps": 80,
                           "remaining": 25, "actions": [[0.1, 0.2]]}],
            "trace": [{"distance": 15., "terminated": True, "truncated": False}]}


def test_equivalent_pair():
    assert compare(example(), deepcopy(example()))["same"]


def test_disagreement_is_recorded():
    other = deepcopy(example())
    other["decisions"][0]["actions"][0][0] = .3
    other["trace"][0]["terminated"] = False
    result = compare(example(), other)
    assert not result["same"]
    assert result["issues"] == ["action_values", "termination"]


def test_missing_trace_is_not_silent():
    other = deepcopy(example())
    other["trace"].clear()
    assert "trace_length" in compare(example(), other)["issues"]
