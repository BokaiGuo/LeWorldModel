import numpy as np
import pytest
from experiments.spectral_reachability.run_pusht_d2p_sealed_program_execution import checked_commands


def test_exact_commands_preserved():
    p = np.array([[0., .2], [-.3, 2.]], dtype=np.float32)
    c = np.tanh(p)
    assert np.array_equal(checked_commands({'executed_program_params': p, 'program_live_commands': c}), c)


@pytest.mark.parametrize('commands', [[[0., 0.]], [[0., 0.], [0., .1]], [[0., 0.], [0., float('nan')]], [[0., 0.], [0., 2.]]])
def test_bad_commands_rejected(commands):
    with pytest.raises(ValueError):
        checked_commands({'executed_program_params': np.zeros((2, 2)), 'program_live_commands': commands})
