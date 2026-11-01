import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_cp10_rgb_refresh_control_20261101_exclusive as m


def test_same_cp5_decision_and_cp10_policy_difference():
    assert m.choose([1.0, 0.5]) == "optimized"
    assert m.choose([1.0, 1.0]) == "hold"
    assert m.choose_last("open_loop_shadow", "optimized", [0.1, 0.2]) == "optimized"
    assert m.choose_last("rgb_refresh", "optimized", [0.1, 0.2]) == "hold"
    with pytest.raises(ValueError):
        m.choose([float("nan"), 0.2])


def test_cp10_history_uses_actual_packets_and_normalization():
    original = {"latent_history": [[0.0]*192, [1.0]*192, [2.0]*192],
                "action_history": [[9.0]*10]*2}
    prefix = np.full((5, 2), [0.1, 0.2], dtype=np.float32)
    first = np.full((5, 2), [0.3, 0.4], dtype=np.float32)
    actual = m.cp10_context(original, prefix, first, [3.0]*192, [0.1, 0.2], [0.2, 0.1])
    assert np.allclose(actual["latent_history"], [[1.0]*192, [2.0]*192, [3.0]*192])
    assert np.allclose(actual["action_history"][0], [0, 0]*5)
    assert np.allclose(actual["action_history"][1], [1, 2]*5)
    assert original["latent_history"][0] == [0.0]*192
    with pytest.raises(ValueError):
        m.cp10_context(original, prefix, first, [3.0]*192, [0, 0], [1, 0])


def test_actions_fail_closed():
    with pytest.raises(ValueError):
        m.actions(np.full((5, 2), 1.01), (5, 2))
    with pytest.raises(ValueError):
        m.actions(np.zeros((4, 2)), (5, 2))
