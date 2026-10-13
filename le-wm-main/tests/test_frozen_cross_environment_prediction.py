import numpy as np
from experiments.audit_frozen_cross_environment_prediction import CONTEXT, _action_stats, _windows


def test_action_calibration_uses_only_real_transition_rows_not_terminal_padding():
    first = np.asarray([[1.0, 3.0], [5.0, 7.0], [0.0, 0.0]], dtype=np.float32)
    second = np.asarray([[2.0, 4.0], [6.0, 8.0], [0.0, 0.0]], dtype=np.float32)
    mean, std = _action_stats([first, second])
    assert np.allclose(mean, [3.5, 5.5])
    assert np.all(std > 0.0)


def test_windows_never_request_a_target_outside_context_plus_horizon():
    assert _windows(CONTEXT + 4 - 1, 4) == []
    assert _windows(CONTEXT + 4, 4) == [0]
    assert _windows(CONTEXT + 7, 4) == [0, 3]
