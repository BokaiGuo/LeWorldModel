import pytest

from experiments.run_live_cube_rgb_pick_place import (
    GRIPPER_CLOSE_ACTION,
    GRIPPER_OPEN_ACTION,
    _fit_effect,
    _inverse_effect_action,
    _validate_effect_probe_protocol,
)


def test_cube_gripper_increment_semantics_match_ogbench():
    # OGBench's Cube oracle uses +1 to close/grasp and -1 to open/release;
    # the state field name ``gripper_opening`` is not the physical direction.
    assert GRIPPER_OPEN_ACTION == -1.0
    assert GRIPPER_CLOSE_ACTION == 1.0


def test_effect_transport_requires_post_lift_probes():
    with pytest.raises(ValueError, match="lift-before-probes"):
        _validate_effect_probe_protocol("effect", False)


def test_chart_transport_can_use_non_effect_probe_protocol():
    _validate_effect_probe_protocol("chart", False)


def test_inverse_effect_action_uses_the_declared_chart():
    action = _inverse_effect_action(
        [1.0, 0.0],
        [[2.0, 0.0], [0.0, 1.0]],
        0.5,
    )
    assert action is not None
    assert float(action[0]) == 0.5
    assert float(action[1]) == 0.0


def test_effect_fit_rejects_full_rank_zero_effect_history():
    actions = [
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
    ]
    mapping, rank = _fit_effect(actions, [[0.0, 0.0]] * 4)
    assert mapping is None
    assert rank == 2


def test_effect_fit_rejects_inconsistent_contact_modes():
    actions = [
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
    ]
    deltas = [[1.0, 0.0], [0.8, 0.0], [0.0, 0.0], [0.0, 1.0]]
    mapping, _rank = _fit_effect(actions, deltas)
    assert mapping is None
