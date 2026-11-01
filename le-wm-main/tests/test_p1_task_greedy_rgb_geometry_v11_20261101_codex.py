from copy import deepcopy

from experiments.spectral_reachability import (
    p1_dev302_mixed_bank_cuda_v12_20261101_codex as planner,
)
from experiments.spectral_reachability import (
    p1_task_greedy_rgb_geometry_v11_20261101_codex as subject,
)


def test_geometry_greedy_selects_direct_without_label_privilege():
    protocol = planner.protocol()
    _audit, _report, current, goal = planner.runner._images()
    trace = {row["index"]: row for row in subject.score_candidates(protocol, current, goal)}
    assert trace[4]["opportunity_by_step"] == [False, False, False, True, True]
    assert trace[6]["opportunity_by_step"] == [False, False, False, False, False]
    assert trace[16]["opportunity_by_step"] == [False, False, False, False, True]
    assert subject.task_greedy_index(protocol, current, goal) == 4
    renamed = deepcopy(protocol)
    for row in renamed["prefix_bank"]["candidates"]:
        row["label"] = f"candidate_{row['index']}"
    assert subject.task_greedy_index(renamed, current, goal) == 4
    reversed_protocol = deepcopy(protocol)
    reversed_protocol["prefix_bank"]["candidates"].reverse()
    assert subject.task_greedy_index(reversed_protocol, current, goal) == 4


def test_geometry_greedy_fails_closed_on_action_or_outcome_tampering():
    protocol = planner.protocol()
    _audit, _report, current, goal = planner.runner._images()
    tampered = deepcopy(protocol)
    tampered["prefix_bank"]["candidates"][4]["actions"][0][0] += 0.01
    try:
        subject.task_greedy_index(tampered, current, goal)
    except ValueError as exc:
        assert "actions SHA" in str(exc)
    else:
        raise AssertionError("tampered action must fail closed")
    leaked = deepcopy(protocol)
    leaked["prefix_bank"]["candidates"][4]["native_success"] = True
    try:
        subject.task_greedy_index(leaked, current, goal)
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("outcome field must fail closed")
