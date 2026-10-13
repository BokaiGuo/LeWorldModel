import numpy as np

from experiments.run_pusht_contact_predicted_latent_adapter import (
    block_motion,
    intervention_actions,
    jsonable,
)


def test_block_motion_ignores_agent_only_changes():
    before = np.asarray([10.0, 20.0, 30.0, 40.0, 0.0, 1.0, 0.0])
    after = before.copy()
    after[:2] += 5.0
    assert not block_motion(before, after)


def test_block_motion_detects_position_and_wrapped_angle_changes():
    before = np.asarray([10.0, 20.0, 30.0, 40.0, np.pi - 0.01, 0.0, 0.0])
    moved = before.copy()
    moved[2] += 0.01
    rotated = before.copy()
    rotated[4] = -np.pi + 0.01
    assert block_motion(before, moved)
    assert block_motion(before, rotated)


def test_contact_interventions_are_paired_and_bounded():
    case = {"state": np.asarray([10.0, 20.0, 50.0, 20.0, 0.0, 0.0, 0.0])}
    actions = intervention_actions(case, 0.85)
    assert len(actions) == 3
    assert np.allclose(actions[0], -actions[1])
    assert np.allclose(actions[2], 0.0)
    assert all(np.max(np.abs(action)) <= 0.85 + 1e-8 for action in actions)


def test_jsonable_converts_numpy_scalars_and_arrays():
    converted = jsonable({"array": np.asarray([1.0, 2.0]), "scalar": np.float64(3.0)})
    assert converted == {"array": [1.0, 2.0], "scalar": 3.0}
