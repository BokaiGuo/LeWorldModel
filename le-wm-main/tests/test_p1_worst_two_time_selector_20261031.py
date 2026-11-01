import copy
import json

import pytest

from experiments.spectral_reachability import p1_worst_two_time_selector_20261031 as experiment


@pytest.fixture(scope="module")
def inputs():
    b = experiment.binding
    return [json.loads(path.read_bytes()) for path in (b.MID, b.real.INPUT)]


def test_all_candidates_and_no_actual_rgb_selection(inputs):
    mid, actual = inputs
    result = experiment.analyze(mid, actual)
    assert len(result["rows"]) == 24
    assert all(result["summary"][role]["records"] == 8
               for role in ("selected", "zero", "greedy"))
    for row in result["rows"]:
        expected = min(row["scores"], key=lambda c: (max(c["predicted5"], c["open10"]),
                                                      c["candidate"]))
        assert row["worst_two_time_candidate"] == expected["candidate"]
    assert result["P1_admission"] is False


def test_rejects_misalignment_and_nonfinite_score(inputs):
    mid, actual = copy.deepcopy(inputs)
    actual["rows"][0]["candidate"] = 99
    with pytest.raises(ValueError):
        experiment.analyze(mid, actual)
    mid, actual = copy.deepcopy(inputs)
    mid["rows"][0]["predicted_midpoint_goal_mse"] = float("nan")
    with pytest.raises(ValueError):
        experiment.analyze(mid, actual)
