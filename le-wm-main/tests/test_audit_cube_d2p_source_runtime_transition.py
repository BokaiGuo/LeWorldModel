import numpy as np
import pytest
import h5py
from unittest.mock import patch

from experiments.spectral_reachability.audit_cube_d2p_source_runtime_transition import (
    _source_predecessor_transition,
    _restore_source_root,
    _state_error,
    _source_successor_trace,
    _source_successor,
)


def test_cube_transition_state_error_requires_identical_shapes_and_reports_maximum() -> None:
    assert _state_error(np.asarray([1.0, 3.0]), np.asarray([1.0, 2.5])) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="incompatible"):
        _state_error(np.zeros(2), np.zeros(3))


def test_cube_transition_reads_packet_after_observed_root(tmp_path) -> None:
    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("pixels", data=np.arange(8 * 2 * 2 * 3, dtype=np.uint8).reshape(8, 2, 2, 3))
        handle.create_dataset("qpos", data=np.arange(8 * 2, dtype=np.float64).reshape(8, 2))
        handle.create_dataset("qvel", data=np.arange(8 * 3, dtype=np.float64).reshape(8, 3))
        handle.create_dataset("action", data=np.arange(8 * 5, dtype=np.float32).reshape(8, 5))
        handle.create_dataset("time", data=np.arange(8, dtype=np.float64)[:, None])
    with h5py.File(path, "a") as handle:
        handle.create_dataset("control", data=np.arange(8 * 7, dtype=np.float64).reshape(8, 7))
    rgb, qpos, qvel, packet, controls = _source_successor(path, 1)
    np.testing.assert_array_equal(rgb, np.arange(8 * 2 * 2 * 3, dtype=np.uint8).reshape(8, 2, 2, 3)[6])
    np.testing.assert_array_equal(qpos, np.arange(8 * 2, dtype=np.float64).reshape(8, 2)[6])
    np.testing.assert_array_equal(qvel, np.arange(8 * 3, dtype=np.float64).reshape(8, 3)[6])
    np.testing.assert_array_equal(packet, np.arange(8 * 5, dtype=np.float32).reshape(8, 5)[1:6])
    np.testing.assert_array_equal(controls, np.arange(8 * 7, dtype=np.float64).reshape(8, 7)[2:7])
    qpos_trace, qvel_trace = _source_successor_trace(path, 1)
    np.testing.assert_array_equal(qpos_trace, np.arange(8 * 2, dtype=np.float64).reshape(8, 2)[2:7])
    np.testing.assert_array_equal(qvel_trace, np.arange(8 * 3, dtype=np.float64).reshape(8, 3)[2:7])
    previous = _source_predecessor_transition(path, 1)
    np.testing.assert_array_equal(previous[0], np.arange(8 * 2, dtype=np.float64).reshape(8, 2)[0])
    np.testing.assert_array_equal(previous[3], np.arange(8 * 5, dtype=np.float32).reshape(8, 5)[1])
    np.testing.assert_array_equal(previous[4], np.arange(8 * 7, dtype=np.float64).reshape(8, 7)[1])


def test_cube_restore_explicitly_sets_source_qpos_qvel_after_reset() -> None:
    class FakeData:
        def __init__(self) -> None:
            self.ctrl = np.zeros(2, dtype=np.float64)
            self.qacc_warmstart = np.ones(3, dtype=np.float64)
            self.time = 0.0

    class FakeRaw:
        def __init__(self) -> None:
            self.data = FakeData()
            self.model = object()
            self._num_cubes = 1
            self.state = None
            self.pre_step_called = False

        def set_target_pos(self, *_args) -> None:
            pass

        def set_state(self, qpos, qvel) -> None:
            self.state = (np.asarray(qpos).copy(), np.asarray(qvel).copy())

        def pre_step(self) -> None:
            self.pre_step_called = True

        def post_step(self) -> None:
            pass

    class FakeEnv:
        def __init__(self) -> None:
            self.unwrapped = FakeRaw()
            self.reset_calls = []

        def reset(self, **kwargs) -> None:
            self.reset_calls.append(kwargs)

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

    context = {
        "evaluator_only_source_state": {
            "qpos": [1.0, 2.0], "qvel": [3.0, 4.0],
            "prev_qpos": [5.0, 6.0], "prev_qvel": [7.0, 8.0],
            "control": [9.0, 10.0], "time": [11.0],
            "target_block": 0, "target_pos": [0.1, 0.2, 0.3],
            "target_yaw": [0.4], "target_task": "cube",
        }
    }
    env = FakeEnv()
    with patch("experiments.spectral_reachability.audit_cube_d2p_source_runtime_transition.mujoco.mj_forward"):
        _restore_source_root(env, context)
    np.testing.assert_array_equal(env.unwrapped.state[0], [1.0, 2.0])
    np.testing.assert_array_equal(env.unwrapped.state[1], [3.0, 4.0])
    assert env.reset_calls == [{"seed": 0}]
    assert not env.unwrapped.pre_step_called
    np.testing.assert_array_equal(env.unwrapped.data.qacc_warmstart, 0.0)
