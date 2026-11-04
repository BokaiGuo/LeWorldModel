"""Unit tests for state transfer and fail-closed behavior."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import pc_pointmaze_shadow_renderer_worker_v2_20261103_codex as worker


class Sim:
    def __init__(self):
        self.data = SimpleNamespace(**{field: np.arange(i + 1, dtype=float)
                                      for i, field in enumerate(worker.EXTRA)})
        self.state = SimpleNamespace(time=1.25, qpos=np.array([1., 2.]), qvel=np.array([3., 4.]),
                                     act=None, udd_state={})

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = SimpleNamespace(time=state.time, qpos=state.qpos.copy(),
                                     qvel=state.qvel.copy(), act=None, udd_state=dict(state.udd_state))


def test_transfer_copies_all_fields_without_aliasing():
    source, shadow = Sim(), Sim()
    source.data.ctrl[:] = 9
    worker.transfer(source, shadow)
    assert worker.same(worker.snapshot(source), worker.snapshot(shadow))
    shadow.data.ctrl[:] = 0
    shadow.state.qpos[:] = 0
    assert np.all(source.data.ctrl == 9)
    assert np.array_equal(source.state.qpos, [1, 2])


def test_nonempty_udd_fails_closed():
    source, shadow = Sim(), Sim()
    source.state.udd_state = {"hidden": 1}
    with pytest.raises(ValueError, match="udd_state"):
        worker.transfer(source, shadow)


def test_none_field_is_supported_but_none_array_mismatch_fails():
    source, shadow = Sim(), Sim()
    source.data.mocap_pos = shadow.data.mocap_pos = None
    worker.transfer(source, shadow)
    shadow.data.mocap_pos = np.array([])
    with pytest.raises(ValueError, match="shape mismatch"):
        worker.transfer(source, shadow)
