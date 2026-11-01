import copy

import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev303_cp5_tail_handoff_20261102_exclusive as handoff


def example():
    prefix = np.zeros((5, 2), dtype="<f4")
    tail = np.full((10, 2), 0.5, dtype="<f4")
    checkpoints = [{"step": i, "state": [i], "snapshot": {"state": [i], "v": 0},
                    "pixel_sha256": str(i)} for i in range(1, 6)]
    return {"prefix": prefix.tolist(), "tail": tail.tolist(),
            "tail_sha256": handoff.digest(tail.tobytes()), "prefix_checkpoints": checkpoints}


def test_sealed_sources_read_only():
    result = handoff.build()
    assert result["planner_report_sha256"] == handoff.REPORT_SHA
    assert [x["arm"] for x in result["roles"]] == list(handoff.ARMS)
    assert result["P1_admission"] is False
    assert result["physical_actions_authorized"] is False


def test_exact_history_gate():
    role = example()
    gate = handoff.ReplayGate(role)
    with pytest.raises(ValueError):
        gate.fixed_tail()
    for point in role["prefix_checkpoints"]:
        gate.accept_prefix([0.0, 0.0], point)
    tail = gate.fixed_tail()
    tail[0, 0] = 0
    assert gate.fixed_tail()[0, 0] == 0.5


@pytest.mark.parametrize("failure", ["action", "snapshot", "pixel", "terminated", "truncated"])
def test_mismatch_latches_failure(failure):
    role = example()
    gate = handoff.ReplayGate(role)
    point = copy.deepcopy(role["prefix_checkpoints"][0])
    action = [0.0, 0.0]
    flags = {}
    if failure == "action":
        action[0] = 0.5
    elif failure == "snapshot":
        point["snapshot"]["v"] = 1
    elif failure == "pixel":
        point["pixel_sha256"] = "bad"
    else:
        flags[failure] = True
    with pytest.raises(ValueError):
        gate.accept_prefix(action, point, **flags)
    with pytest.raises(ValueError):
        gate.fixed_tail()
    with pytest.raises(ValueError):
        gate.accept_prefix([0.0, 0.0], role["prefix_checkpoints"][0])


def test_bad_tail_and_action_rejected():
    role = example()
    role["tail"][0][0] = 0.25
    with pytest.raises(ValueError):
        handoff.ReplayGate(role)
    with pytest.raises(ValueError):
        handoff.actions([[float("nan"), 0]], (1, 2))
    with pytest.raises(ValueError):
        handoff.actions([[0.1, 0]], (1, 2))
