import copy
import math

import pytest

from experiments.spectral_reachability.analyze_pc_fd_common_tail_effects_20261102 import analyze


def fixture_report():
    rows = []
    for seed in (20263160, 20263161):
        branches = []
        for arm in ("selected", "parameter_zero"):
            for tail in range(5):
                actions = [[0.0, 0.0] for _ in range(15)]
                actions[5:] = [[tail / 10, 0.0] for _ in range(10)]
                points = []
                for step in range(16):
                    q = 3.2 if step == 0 else (0.1 if step <= 5 else 0.1 + tail / 10)
                    point = {"qpos": [q, 0.0], "target_qpos": [0.0, 0.0],
                             "qpos_distance": abs((q + math.pi) % (2 * math.pi) - math.pi)}
                    if step:
                        point["reward"] = 0.0
                    points.append(point)
                branches.append({"arm": arm, "tail_index": tail, "status": "complete",
                                 "stop": None, "attempted_steps": 15, "returned_steps": 15,
                                 "raw_actions": actions, "trajectory": points})
        rows.append({"seed": seed, "status": "branches_recorded", "branches": branches})
    return {"status": "finished", "expected_branches": 20, "recorded_branches": 20,
            "complete_branches": 20, "rows": rows}


def test_wrapped_distance_and_pairing():
    result = analyze(fixture_report())
    assert result["roots"][0]["initial_distance"] == pytest.approx(2 * math.pi - 3.2)
    assert all(p["selected_minus_zero_relative_to_tail0"] == 0 for p in result["roots"][0]["pairs"])


@pytest.mark.parametrize("mutation", ["missing", "tail", "stop", "distance", "prefix"])
def test_invalid_input_fails_closed(mutation):
    report = copy.deepcopy(fixture_report())
    branch = report["rows"][0]["branches"][5]
    if mutation == "missing":
        report["recorded_branches"] = 19
    elif mutation == "tail":
        branch["raw_actions"][5][0] = 0.99
    elif mutation == "stop":
        branch["stop"] = {"step": 5}
    elif mutation == "distance":
        branch["trajectory"][5]["qpos_distance"] = 100
    else:
        report["rows"][0]["branches"][1]["trajectory"][5]["qpos_distance"] = 0.2
    with pytest.raises(ValueError):
        analyze(report)
