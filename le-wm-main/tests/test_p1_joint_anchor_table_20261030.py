import copy
import math

import pytest

from experiments.spectral_reachability import p1_joint_anchor_table_20261030 as table


def fixture():
    anchor = {"index": 0, "logical_pair": 0, "root_slot": 0, "condition": "off",
              "arm": "a", "role": "selected", "task_id": "task", "physical_root_sha256": "root",
              "prefix5_float32_hex": "00", "prefix5_sha256": "prefix",
              "checkpoint": {"pixel_sha256": "pixel", "state": [0.0] * 7},
              "root_state": [0.0] * 7, "goal": [0.0] * 7}
    item = {key: anchor[key] for key in ("index", "logical_pair", "root_slot", "condition", "arm", "role", "task_id", "prefix5_float32_hex")}
    item.update(cp5_pixel_sha256="pixel", raw10_sha256="suffix")
    row = {key: anchor[key] for key in ("index", "root_slot", "condition", "arm", "role")}
    row.update(completed_endpoint=[0.0] * 7, status="probe_completed", returned_suffix=10,
               last_accepted_state_metrics={"native_success": True, "object_position_error": 0.0})
    anchors, roster, rows = [], [], []
    for i in range(24):
        a, b, c = copy.deepcopy(anchor), copy.deepcopy(item), copy.deepcopy(row)
        a["index"] = b["index"] = c["index"] = i
        anchors.append(a)
        roster.append(b)
        rows.append(c)
    return ({"anchors": anchors}, {"roster": roster, "P1_admission": False},
            {"rows": rows, "P1_admission": False})


def test_identity_and_pending_panel():
    p, q, r = fixture()
    result = table.build(p, q, r)
    assert len(result["rows"]) == 24
    assert result["rows"][0]["common_suffix"] is None
    assert result["rows"][0]["prefix_native_success_transition"] == [True, True]
    q["roster"][0]["prefix5_float32_hex"] = "01"
    with pytest.raises(ValueError, match="prefix identity"):
        table.build(p, q, r)


def test_censoring_and_wrapped_angle():
    p, q, r = fixture()
    r["rows"][0].update(completed_endpoint=None, status="early_stop", returned_suffix=3)
    result = table.build(p, q, r)
    assert result["rows"][0]["independent_suffix_20step"]["endpoint"] is None
    r["rows"][0]["status"] = "probe_completed"
    with pytest.raises(ValueError, match="censoring"):
        table.build(p, q, r)
    state = [0., 0., 0., 0., 2 * math.pi + 0.1, 0., 0.]
    result = table.metrics(state, [0.] * 7)
    assert result["object_angle_error_wrapped"] == pytest.approx(0.1)
    assert result["native_angle_negative"]
