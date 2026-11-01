import numpy as np

from experiments.spectral_reachability import (
    p1_dev302_mixed_bank_cuda_v3_20261101_codex as v3,
)
from experiments.spectral_reachability import (
    p1_dev302_mixed_bank_cuda_v4_20261101_codex as subject,
)


def _rows(value):
    return {row["label"]: row for row in value["prefix_bank"]["candidates"]}


def test_v4_only_changes_routed_rgb_candidates_and_tightens_guard():
    old = _rows(v3.protocol())
    new_protocol = subject.protocol()
    new = _rows(new_protocol)
    unchanged = {
        "zero",
        "spectral",
        "gn",
        "rgb_nearest_contact",
        "rgb_goal_support_direct",
        "random_control",
    }
    for label in unchanged:
        assert new[label]["actions_sha256"] == old[label]["actions_sha256"]
    for label in ("rgb_goal_support_top_route", "rgb_goal_support_bottom_route"):
        assert new[label]["actions_sha256"] != old[label]["actions_sha256"]
        assert new[label]["route_terminal_overtravel_pixels"] == 3.0
    guard = new_protocol["prefix_bank"]["contact_guard"]
    assert guard["contact_center_to_object_pixels"] == 5.5
    assert guard["route_terminal_overtravel_pixels"] == 3.0
    assert guard["simulator_state_used"] is False
    assert new_protocol["P1_admission"] is False


def test_v4_admits_goal_support_paths_and_rejects_old_false_candidates():
    rows = _rows(subject.protocol())
    assert rows["zero"]["contact_feasible_prior"] is False
    assert rows["spectral"]["contact_feasible_prior"] is False
    assert rows["gn"]["contact_feasible_prior"] is False
    assert rows["rgb_nearest_contact"]["contact_feasible_prior"] is False
    assert rows["random_control"]["contact_feasible_prior"] is False
    assert rows["rgb_goal_support_direct"]["contact_feasible_prior"] is True
    assert rows["rgb_goal_support_top_route"]["contact_feasible_prior"] is True
    assert rows["rgb_goal_support_bottom_route"]["contact_feasible_prior"] is True
    top = np.asarray(rows["rgb_goal_support_top_route"]["actions"], np.float32)
    assert top.shape == (5, 2)
    assert np.max(np.abs(top)) <= 1.0
