import numpy as np
import pytest

from experiments.spectral_reachability.audit_mixedmode_capability_native_MAIN_20261028_v1 import native


def test_native_reward_velocity_is_not_success_threshold():
    value = native([0, 0, 0, 0, 0, 100, 0], [0]*7)
    assert value['reward'] == -100 and value['native_success'] is True


def test_native_includes_agent_xy_and_preserves_angle_formula():
    assert native([30, 0, 0, 0, 0, 0, 0], [0]*7)['native_success'] is False
    assert native([0, 0, 0, 0, np.pi, 0, 0], [0]*7)['native_success'] is False
    assert native([0, 0, 0, 0, 8*np.pi, 0, 0], [0]*7)['native_success'] is True


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        native([np.nan]*7, [0]*7)
