import json

import pytest

from experiments.spectral_reachability import codex_p1_diag32_cp5_saved_paired_audit_20261101_exclusive as audit


def test_matching_requires_cp5_prefix_and_full_tail():
    point = dict(state=[1], snapshot={"state": [1]}, pixel_sha256="p",
                 goal=[2], goal_rgb_sha256="g")
    row = {"observations": [point] * 6, "executed_raw": [[0, 0]] * 5}
    schedule = {"commands": [[0, 0]] * 5 + [[1, 1]] * 10}
    assert audit.matches(row, [[1, 1]] * 10, row, schedule)
    assert not audit.matches(row, [[1, 0]] * 10, row, schedule)
    different = dict(point, goal_rgb_sha256="different")
    assert not audit.matches(row, [[1, 1]] * 10,
                             dict(row, observations=[different] * 6), schedule)
    assert not audit.matches(row, [[1, 1]] * 10,
                             dict(row, observations=[point] * 5), schedule)


def test_write_new_cannot_overwrite(tmp_path):
    path = tmp_path / "result.json"
    audit.write_new(path, {"original": True})
    with pytest.raises(FileExistsError):
        audit.write_new(path, {"original": False})
    assert json.loads(path.read_text()) == {"original": True}


def test_censor_is_based_on_returned_count():
    row = {"observations": [{"state": [0, 0, 0, 0], "goal": [0, 0, 0, 0]}],
           "final_metrics": {"native_success": True, "wrapped_allxy_success": True,
                             "native_angle_term": 0, "wrapped_object_angle_error": 0},
           "control_returned": 14, "terminated": True, "truncated": False}
    assert audit.outcome(row)["unobserved_planned_steps"] == 1
    row["control_returned"] = 15
    assert audit.outcome(row)["unobserved_planned_steps"] == 0
