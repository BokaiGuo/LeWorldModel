import copy

import pytest

from experiments.pc_bound_rgb_closed_loop_replay_20261107_codex import compare


def fixture():
    return {"ever_success": True,
            "trace": [{"action": [0., 1.], "distance": 1., "terminated": True, "truncated": False}],
            "decisions": [{"iteration": 0, "remaining": 25, "queries": 16, "packet_steps": 80,
                           "forward_calls": 3, "retained": [0, 1],
                           "choice": {"source": "tail", "index": 1},
                           "actions": [[0., 1.]], "dispatched": [[0., 1.]]}]}


def test_same_trace_passes():
    a = fixture()
    assert compare("effect", a, copy.deepcopy(a))["max_distance_abs_diff"] == 0


@pytest.mark.parametrize("field", ["action", "clock", "terminal", "choice"])
def test_replay_rejects_changed_contract(field):
    a, b = fixture(), fixture()
    if field == "action":
        b["decisions"][0]["actions"][0][0] = .1
    elif field == "clock":
        b["decisions"][0]["packet_steps"] += 1
    elif field == "terminal":
        b["trace"][0]["terminated"] = False
    else:
        b["decisions"][0]["choice"]["index"] = 2
    with pytest.raises(ValueError):
        compare("effect", a, b)
