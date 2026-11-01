import numpy as np
import pytest

from experiments.spectral_reachability import (
    p1_rgb_edge_approach_diagnostic_20261102_exclusive as subject,
)


def images(object_x=40, goal_x=100, pusher_x=20, y=100):
    current = np.zeros((224, 224, 3), dtype=np.uint8)
    goal = np.zeros_like(current)
    current[y - 2:y + 3, pusher_x - 2:pusher_x + 3] = subject.rgb.BLUE
    current[y - 2:y + 3, object_x:object_x + 5] = subject.rgb.BLOCK
    goal[y - 2:y + 3, goal_x:goal_x + 5] = subject.rgb.BLOCK
    return current, goal


def candidate(label, actions):
    return {"label": label, "actions": np.asarray(actions).tolist()}


def test_offscreen_support_target_and_no_holdout_claim():
    current, goal = images(object_x=4, goal_x=70, pusher_x=20)
    value = subject.diagnose(current, goal, [candidate("zero", np.zeros((5, 2)))])
    assert value["object_bounds_xyxy"][0] == 4
    assert value["support_target_xy"][0] < 0
    assert value["support_target_center_in_frame"] is False
    assert value["support_target_disk_in_frame"] is False
    assert value["P1_admission"] is False
    assert value["independent_confirmation"] is False
    assert "not_original_holdout_confirmation" in value["classification"]


def test_interior_approach_uses_frozen_dynamics_and_thresholds():
    current, goal = images(object_x=39)
    action = np.zeros((5, 2), dtype=np.float32)
    # Construct an endpoint on the support side under the frozen dynamics.
    action[4, 0] = 1.0
    value = subject.diagnose(current, goal, [candidate("approach", action)])
    row = value["candidates"][0]
    assert row["predicted_edge_safe"] is True
    assert row["frozen_contact_distance_pass"] is True
    assert row["frozen_support_alignment_pass"] is True
    assert row["approach_feasible_approximation"] is True
    assert value["contact_distance_pixels"] == subject.dynamics.CONTACT_CENTER_TO_OBJECT_PIXELS
    assert value["minimum_support_alignment"] == subject.support.MIN_SUPPORT_ALIGNMENT


def test_early_contact_invalidates_noncontact_extrapolation():
    current, goal = images()
    action = np.zeros((5, 2), dtype=np.float32)
    action[0, 0] = 1.0
    row = subject.diagnose(current, goal, [candidate("early", action)])["candidates"][0]
    assert row["predicted_contact_before_final_step"] is True
    assert row["approach_feasible_approximation"] is False


@pytest.mark.parametrize("axis,side", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_image_edge_check_is_symmetric(axis, side):
    point = np.array([100.0, 100.0])
    point[axis] = 0.0 if side == 0 else 223.0
    assert not bool(subject._inside(point, 224, 224, subject.geometry.PUSHER_RADIUS_PIXELS))


def test_edge_crossing_rejects_otherwise_nearby_candidate():
    current, goal = images(object_x=10, goal_x=70, pusher_x=20)
    action = np.zeros((5, 2), dtype=np.float32)
    action[0, 0] = -1
    row = subject.diagnose(current, goal, [candidate("out", action)])["candidates"][0]
    assert row["predicted_edge_safe"] is False
    assert row["approach_feasible_approximation"] is False


def test_target_center_can_be_inside_but_disk_cross_edge():
    current, goal = images(object_x=10, goal_x=70, pusher_x=30)
    value = subject.diagnose(current, goal, [candidate("zero", np.zeros((5, 2)))])
    assert value["support_target_center_in_frame"] is True
    assert value["support_target_disk_in_frame"] is False


def test_clipped_or_missing_masks_fail_closed():
    current, goal = images(object_x=4)
    current[100, 0] = subject.rgb.BLOCK
    with pytest.raises(ValueError, match="clipped"):
        subject.diagnose(current, goal, [candidate("zero", np.zeros((5, 2)))])
    current[:] = 0
    with pytest.raises(ValueError, match="mask absent"):
        subject.diagnose(current, goal, [candidate("zero", np.zeros((5, 2)))])


@pytest.mark.parametrize("actions", [np.full((5, 2), np.nan), np.full((5, 2), 1.1),
                                     np.zeros((4, 2))])
def test_bad_actions_fail_closed(actions):
    current, goal = images()
    with pytest.raises(ValueError, match="bounded finite"):
        subject.diagnose(current, goal, [candidate("bad", actions)])


def test_no_hidden_fields_or_duplicate_labels():
    current, goal = images()
    row = candidate("x", np.zeros((5, 2)))
    with pytest.raises(ValueError, match="unique"):
        subject.diagnose(current, goal, [row, row])
    with pytest.raises(ValueError, match="candidate fields"):
        subject.diagnose(current, goal, [{**row, "outcome": 1}])
