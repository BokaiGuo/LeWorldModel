import copy

import pytest

from experiments.spectral_reachability import p1_cp5_mechanism_planning_join_20261031 as joiner
from test_p1_crossover_paired_table_20261031 import fixture


def inputs():
    cross = fixture()
    anchors, entries, plans, results = [], [], [], []
    metrics = dict.fromkeys(joiner.METRICS, 1.0) | {"native_success": False}
    for i in range(24):
        pair, role = i // 3, joiner.paired.ROLES[i % 3]
        anchors.append({"index": i, "role": role, "logical_pair": pair, "root_slot": pair,
                        "condition": "on", "arm": "a", "prefix5_float32_hex": role,
                        "checkpoint": {"pixel_sha256": role}})
        entries.append({"index": i, "role": role, "logical_pair": pair, "root_slot": pair,
                        "condition": "on", "blind": {"prefix_raw_f32_hex": role,
                        "cp5_pixel_sha256": role}, "action_sha256": role,
                        "observations": [{}, {"physical": metrics}]})
        plans.append({"index": i, "role": role, "logical_pair": pair,
                      "selected_action_sha256": role, "initial_f32_sha256": str(pair),
                      "planning_wall_seconds": 1.0})
        results.append({"index": i, "role": role, "root_slot": pair, "condition": "on",
                        "arm": "a", "status": "probe_completed", "returned_suffix": 10})
    return cross, anchors, entries, plans, results


def test_join_produces_planning_metrics_for_every_crossover_branch():
    rows = joiner.join(*inputs())
    assert len(rows) == 96
    assert all(row["own_h10_object_position_error"] == 1.0 for row in rows)


@pytest.mark.parametrize("which", ["prefix", "rgb", "start", "action", "stop"])
def test_join_rejects_inconsistent_contract(which):
    cross, anchors, entries, plans, results = copy.deepcopy(inputs())
    if which == "prefix":
        entries[1]["blind"]["prefix_raw_f32_hex"] = "wrong"
    elif which == "rgb":
        entries[1]["blind"]["cp5_pixel_sha256"] = "wrong"
    elif which == "start":
        plans[1]["initial_f32_sha256"] = "wrong"
    elif which == "action":
        plans[1]["selected_action_sha256"] = "wrong"
    else:
        results[1]["returned_suffix"] = 9
    with pytest.raises(ValueError):
        joiner.join(cross, anchors, entries, plans, results)
