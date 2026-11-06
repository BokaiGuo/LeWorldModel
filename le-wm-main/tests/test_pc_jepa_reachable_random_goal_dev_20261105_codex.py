import numpy as np

from experiments.pc_jepa_reachable_random_goal_dev_20261105_codex import bank


def test_reference_is_distinct_and_actions_obey_declared_contract():
    actions = bank()
    assert actions.shape == (65, 15, 2) and actions.dtype == np.float32
    assert np.max(np.abs(actions)) <= 1
    assert len({action.tobytes() for action in actions}) == 65
    assert all(np.array_equal(actions[:, i:i + 5], np.repeat(actions[:, i:i + 1], 5, axis=1))
               for i in (0, 5, 10))
    np.testing.assert_array_equal(actions, bank())
