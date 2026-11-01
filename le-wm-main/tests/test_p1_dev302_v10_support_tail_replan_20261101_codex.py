import numpy as np

from experiments.spectral_reachability import (
    p1_dev302_v10_support_tail_replan_20261101_codex as subject,
)


def test_v10_binds_v9_cp5_and_equal_budget_support_replanning():
    value = subject.protocol()
    assert value["schema"] == subject.NAME + ".protocol.v1"
    assert [row["role"] for row in value["roles"]] == list(subject.ROLES)
    assert [row["candidate_index"] for row in value["roles"]] == [16, 0, 4, 7]
    assert all(np.asarray(row["actual_prefix"]).shape == (5, 2) for row in value["roles"])
    assert all(np.asarray(row["contact_seed_tail"]).shape == (2, 5, 2) for row in value["roles"])
    assert value["equal_cem_and_robust_budget"] is True
    assert value["post_cp5_physical_outcomes_used"] is False
    assert value["P1_admission"] is False


def test_v10_selected_cp5_support_direction_points_toward_object():
    value = subject.protocol()
    selected = next(row for row in value["roles"] if row["role"] == "selected")
    direction = np.asarray(
        selected["rgb_contact_geometry"]["contact_direction_raw_xy"], np.float64
    )
    assert direction.shape == (2,)
    assert direction[1] > 0.95
