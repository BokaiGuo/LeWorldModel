from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from experiments.decision_benchmark import extend_pointmaze_state_action_pool as extension
from experiments.decision_benchmark.extend_pointmaze_state_action_pool import rollout_trajectory


def test_extension_requires_enough_new_roots_for_the_isolated_protocol(tmp_path) -> None:
    from experiments.decision_benchmark.extend_pointmaze_state_action_pool import build_extension
    with pytest.raises(ValueError, match="at least 208"):
        build_extension(tmp_path, tmp_path / "out", episode_count=207)


def test_rollout_rejects_invalid_action_shape_before_touching_mujoco() -> None:
    with pytest.raises(ValueError, match="actions must be finite"):
        rollout_trajectory(np.zeros(4), np.zeros((100, 3)))


def test_rollout_uses_the_archive_high_level_transition_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh extension must not silently revert to the old one-tick clock."""

    class FakeData:
        def __init__(self, _model: object) -> None:
            self.qpos = np.zeros(2, dtype=np.float64)
            self.qvel = np.zeros(2, dtype=np.float64)
            self.ctrl = np.zeros(2, dtype=np.float64)

    calls = {"forward": 0, "step": 0, "controls": []}

    def forward(_model: object, _data: FakeData) -> None:
        calls["forward"] += 1

    def step(_model: object, data: FakeData) -> None:
        calls["step"] += 1
        calls["controls"].append(data.ctrl.copy())
        data.qvel += data.ctrl
        data.qpos += data.qvel

    fake_mujoco = types.SimpleNamespace(MjData=FakeData, mj_forward=forward, mj_step=step)
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    monkeypatch.setattr(extension, "build_model", lambda: object())

    states = extension.rollout_trajectory(
        np.asarray([0.0, 0.0, 9.0, -9.0]),
        np.asarray([[2.0, -2.0], [0.5, 0.25], [0.0, 0.0]], dtype=np.float64),
    )

    assert calls["step"] == 2 * extension.FRAME_SKIP
    assert calls["forward"] == 3  # reset plus pre-action velocity clipping
    assert np.allclose(calls["controls"][: extension.FRAME_SKIP], [1.0, -1.0])
    assert np.allclose(states[1], [40.0, -40.0, 10.0, -10.0])
    assert states.dtype == np.float64
