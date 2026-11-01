"""Target-construction algebra, not a CPU neural experiment."""
import numpy as np
import pytest

from experiments.spectral_reachability.reacher_d2p_feasible_goal import feasible_goal_delta


LIMIT = np.deg2rad(160.)
RANGES = [[0., 0.], [-LIMIT, LIMIT]]


@pytest.mark.parametrize("wrist", [-2.6023704184604957, -2.5009439938453513])
def test_observed_failure_root_rejects_periodic_wrist_goal(wrist):
    effective, receipt = feasible_goal_delta([-.7, wrist], [.65, -.65], [False, True], RANGES)
    assert effective.tolist() == [.65, .65]
    assert receipt["historical_target_outside_limits"] == [1]
    assert receipt["flipped_joint_indices"] == [1]
    assert abs(receipt["feasible_target_qpos"][1]) <= LIMIT
    assert abs(effective[1]) == .65


def test_valid_root_retains_exact_original_delta_and_goal():
    root, delta = np.array([3., -.8]), np.array([.65, -.65])
    effective, receipt = feasible_goal_delta(root, delta, [False, True], RANGES)
    assert np.array_equal(effective, delta)
    assert receipt["feasible_target_qpos"] == receipt["historical_target_qpos"]
    assert receipt["flipped_joint_indices"] == []
    assert receipt["planning_uses_privileged_state"] is False


def test_positive_wrist_limit_also_reflects_full_offset():
    effective, receipt = feasible_goal_delta([0., 2.6], [.65, .65], [False, True], RANGES)
    assert effective.tolist() == [.65, -.65]
    assert receipt["feasible_target_qpos"][1] == 2.6 - .65


def test_wrapped_valid_endpoint_does_not_authorize_crossing_a_joint_stop():
    # Zero-4 would wrap back INTO the legal interval, but crossing a bounded
    # stop is not a short periodic displacement; neither full offset fits.
    with pytest.raises(ValueError):
        feasible_goal_delta([0., 0.], [.65, -4.], [False, True], RANGES)


@pytest.mark.parametrize("root,delta", [([0., -3.], [.65, -.65]), ([0., 0.], [.65, 7.]), ([0., float("nan")], [.65, -.65])])
def test_invalid_or_impossible_target_fails_closed(root, delta):
    with pytest.raises(ValueError):
        feasible_goal_delta(root, delta, [False, True], RANGES)


@pytest.mark.parametrize("wrist",[2.82945144,-2.82945144,3.,-3.])
def test_v2_soft_source_preserved_and_complete_offset_target_still_legal(wrist):
    root = np.array([-.7,wrist])
    before = root.copy()
    effective,receipt = feasible_goal_delta(root,[.65,-.65],[False,True],RANGES,allow_soft_source=True)
    np.testing.assert_array_equal(root,before)
    assert abs(effective[1]) == .65
    assert -LIMIT <= receipt["feasible_target_qpos"][1] <= LIMIT
    assert receipt["source_limit_violation_radians"][1] == abs(wrist)-LIMIT
    assert receipt["policy"].endswith("soft_source_v2")
    with pytest.raises(ValueError): feasible_goal_delta(root,[.65,-.65],[False,True],RANGES)


@pytest.mark.parametrize("wrist",[-2.6,-1.,0.,1.,2.6])
def test_in_range_v2_does_not_change_v1_actions_or_targets(wrist):
    args = ([.4,wrist],[.65,-.65],[False,True],RANGES)
    old,old_receipt = feasible_goal_delta(*args)
    new,new_receipt = feasible_goal_delta(*args,allow_soft_source=True)
    np.testing.assert_array_equal(old,new)
    for key in ("effective_delta","feasible_target_qpos","historical_target_qpos","flipped_joint_indices"):
        assert old_receipt[key] == new_receipt[key]
    assert new_receipt["source_limit_violation_radians"] == [0.,0.]


def test_v2_still_refuses_target_when_neither_complete_offset_fits():
    with pytest.raises(ValueError):
        feasible_goal_delta([0.,4.],[.65,.65],[False,True],RANGES,allow_soft_source=True)
