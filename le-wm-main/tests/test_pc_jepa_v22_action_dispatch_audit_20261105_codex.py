import numpy as np

from experiments.audit_pc_jepa_v22_action_dispatch_20261105_codex import expected_history


def test_history_uses_float32_pixel_roundtrip_and_time_order():
    actions = np.arange(20, dtype=np.float64).reshape(10, 2) / 37 - .2
    mean = np.asarray([-.0087, .0068], dtype=np.float32)
    std = np.asarray([.2019, .2002], dtype=np.float32)
    expected = (((100 * actions).astype(np.float32) / np.float32(100)) - mean) / std
    np.testing.assert_array_equal(expected_history(actions, mean, std), expected.reshape(2, 10))
    assert not np.array_equal(expected_history(actions[::-1], mean, std), expected.reshape(2, 10))
