import numpy as np
import pytest
from experiments.spectral_reachability.run_pusht_observed_prefix_prediction import observed_context


def test_incoming_history_shifts_exactly_once():
    frames = [np.full((224, 224, 3), i, np.uint8) for i in range(4)]
    actions = np.array([[0., .1], [.2, .3], [.4, .5]], np.float32)
    command = np.array([.6, .7], np.float32)
    result, carry = observed_context(frames[:3], actions, frames[3], command)
    assert [int(x[0, 0, 0]) for x in result] == [1, 2, 3]
    np.testing.assert_array_equal(carry, np.vstack((actions[1:], command)))
    result[-1][:] = 0
    assert frames[3][0, 0, 0] == 3


@pytest.mark.parametrize('command', [[2., 0.], [float('nan'), 0.], ['x', 'y'], [True, False]])
def test_invalid_executed_commands_rejected(command):
    frames = [np.zeros((224, 224, 3), np.uint8) for _ in range(3)]
    with pytest.raises(ValueError): observed_context(frames, np.zeros((3, 2)), frames[0], command)
