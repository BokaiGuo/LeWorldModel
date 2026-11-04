from copy import deepcopy

import pytest

from experiments.spectral_reachability import p1_timing_second_feedback_inputs_20261102 as subject


def test_real_panel_only_exports_observations_and_executed_actions():
    result = subject.build()
    assert len(result["rows"]) == 8
    assert not result["planning_performed"] and not result["P1_admission"]
    for row in result["rows"]:
        assert len(row["executed_raw_action_packets"]) == 2
        assert all(len(packet) == 5 for packet in row["executed_raw_action_packets"])
        assert not ({"state", "reward", "goal_state"} & row.keys())


@pytest.mark.parametrize("corruption", ["missing_action", "censored", "missing_frame", "wrong_arm"])
def test_invalid_feedback_inputs_fail_closed(corruption):
    contract = subject.saved.physical.read(subject.saved.CONTRACT, subject.saved.CONTRACT_SHA)
    report = subject.saved.physical.read(subject.saved.RUN / "report.json", subject.saved.REPORT_SHA)
    role, row = deepcopy(contract["roles"][0]), deepcopy(report["rows"][0])
    if corruption == "missing_action":
        row["transitions"] = [t for t in row["transitions"]
                              if not (t["phase"] == "tail" and t["step"] == 3)]
    elif corruption == "censored":
        next(t for t in row["transitions"] if t["phase"] == "tail")["terminated"] = True
    elif corruption == "missing_frame":
        row["observations"] = [o for o in row["observations"]
                               if not (o["phase"] == "tail" and o["step"] == 5)]
    else:
        row["arm"] = "other"
    with pytest.raises(ValueError):
        subject.node_input(role, row, subject.saved.RUN)
