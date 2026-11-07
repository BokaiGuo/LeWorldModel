import numpy as np
import pytest

from experiments.pc_pointmaze_packet_teacher_forcing_20261107_codex import STEPS, check_trace


def test_trace_clock_and_finiteness():
    trace = np.zeros((3, 26, 5))
    trace[:, :, 4] = np.arange(26) * .05
    assert check_trace(trace).shape == (3, 26, 5)
    assert STEPS.tolist() == [0, 5, 10, 15, 20, 25]
    trace[0, 5, 4] += .05
    with pytest.raises(AssertionError):
        check_trace(trace)
    trace[0, 5, 4] = np.nan
    with pytest.raises(ValueError):
        check_trace(trace)
