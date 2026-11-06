import numpy as np

from experiments.audit_pc_pusht_val_action_replay_dev_20261105_codex import errors


def test_error_uses_short_periodic_angle_and_separate_fields():
    reference = np.asarray([0, 0, 2, 2, .01, 1, 2])
    actual = np.asarray([3, 4, 2, 3, 2*np.pi-.01, 4, 6])
    result = errors(actual, reference)
    assert result["agent_position"] == 5
    assert result["object_position"] == 1
    assert np.isclose(result["angle"], .02)
    assert result["agent_velocity"] == 5
