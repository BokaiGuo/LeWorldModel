import numpy as np
from types import SimpleNamespace

from experiments.spectral_reachability.reacher_d2p_context import (
    WARMUP_POLICY,
    advance_executed_context,
    collect_executed_context,
    replay_executed_warmup,
    require_new_output,
    restore_execution_snapshot,
    warmup_action_blocks,
)
import pytest


class _ActionSpace:
    low = np.asarray([-1.0, -1.0], dtype=np.float32)
    high = np.asarray([1.0, 1.0], dtype=np.float32)
    shape = (2,)


class _Env:
    action_space = _ActionSpace()

    def reset(self, *, seed):
        self.seed = int(seed)
        self.step_count = 0
        self.state = np.asarray([float(seed), 0.0], dtype=np.float32)
        return None, {"goal": self._image(99)}

    def step(self, action):
        self.state += np.asarray(action, dtype=np.float32)
        self.step_count += 1
        return None, 0.0, False, False, {}

    def render(self):
        return self._image(self.step_count)

    def _image(self, suffix):
        return np.full((2, 2, 3), int(suffix), dtype=np.uint8)


def test_executed_context_pairs_each_history_frame_with_an_executed_action_block():
    env = _Env()
    context = collect_executed_context(env, seed=7, history_len=3, action_block=5)
    assert len(context["frames"]) == len(context["history_action_blocks"]) == 3
    assert context["warmup_raw_steps"].shape == (15, 2)
    assert np.array_equal(context["frames"][0], np.zeros((2, 2, 3), dtype=np.uint8))
    assert np.array_equal(context["frames"][1], np.full((2, 2, 3), 5, dtype=np.uint8))
    assert np.array_equal(context["root_frame"], np.full((2, 2, 3), 15, dtype=np.uint8))
    assert context["root_rgb_sha256"] != context["history_rgb_sha256"][-1]
    assert context["warmup_policy"] == WARMUP_POLICY


def test_warmup_packets_are_deterministic_full_range_and_not_constant_action_manifolds():
    kwargs = {
        "seed": 31, "history_len": 3, "action_block": 5,
        "action_low": np.asarray([-1.0, -1.0], dtype=np.float32),
        "action_high": np.asarray([1.0, 1.0], dtype=np.float32),
    }
    first = warmup_action_blocks(**kwargs)
    second = warmup_action_blocks(**kwargs)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    packets = np.asarray(first, dtype=np.float32).reshape(3, 5, 2)
    assert np.all(packets >= -1.0) and np.all(packets <= 1.0)
    assert all(np.unique(packet, axis=0).shape[0] > 1 for packet in packets)


def test_evidence_output_must_be_new(tmp_path):
    fresh = tmp_path / "fresh"
    require_new_output(fresh, receipt="test receipt")
    fresh.mkdir()
    with pytest.raises(FileExistsError, match="immutable test receipt"):
        require_new_output(fresh, receipt="test receipt")


def test_replaying_recorded_warmup_reconstructs_the_exact_planning_root():
    context = collect_executed_context(_Env(), seed=13, history_len=3, action_block=5)
    root, goal = replay_executed_warmup(_Env(), seed=13, warmup_raw_steps=context["warmup_raw_steps"])
    assert np.array_equal(root, context["root_frame"])
    assert np.array_equal(goal, context["goal"])


class _SnapshotEnv(_Env):
    def __init__(self):
        self.env = SimpleNamespace(
            physics=SimpleNamespace(data=SimpleNamespace(
                qpos=np.zeros(2, dtype=np.float64), qvel=np.zeros(2, dtype=np.float64),
            ), forward=lambda: None),
            task=SimpleNamespace(target_qpos=np.zeros(2, dtype=np.float64)),
        )

    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed):
        self.seed = int(seed)
        self.step_count = 0
        self.env.physics.data.qpos[:] = [float(seed), 0.0]
        self.env.physics.data.qvel[:] = 0.0
        self.env.task.target_qpos = self.env.physics.data.qpos + 0.5
        return None, {"goal": self._image(99)}

    def step(self, action):
        self.env.physics.data.qpos += np.asarray(action, dtype=np.float64)
        self.step_count += 1
        return None, 0.0, False, False, {}

    def render(self):
        return self._image(int(round(self.env.physics.data.qpos[0])) % 255)


def test_execution_snapshot_restores_observed_root_without_replaying_warmup():
    env = _SnapshotEnv()
    context = collect_executed_context(env, seed=13, history_len=3, action_block=5)
    root, goal = restore_execution_snapshot(env, seed=13, snapshot=context["execution_snapshot"])
    assert np.array_equal(root, context["root_frame"])
    assert np.array_equal(goal, context["goal"])
    assert np.allclose(env.env.physics.data.qpos, context["execution_snapshot"]["qpos"])


def test_advanced_context_binds_the_observed_successor_and_shifted_actions():
    env = _SnapshotEnv()
    context = collect_executed_context(env, seed=19, history_len=3, action_block=5)
    raw = np.tile(np.asarray([[0.2, -0.1]], dtype=np.float32), (5, 1))
    advanced = advance_executed_context(env, context=context, raw_action_block=raw, action_block=5)
    assert advanced["context_mode"] == "executed_warmup_plus_observed_action"
    assert np.array_equal(advanced["frames"][-1], context["root_frame"])
    assert np.array_equal(advanced["history_action_blocks"][-1], raw.reshape(-1))
    assert advanced["root_rgb_sha256"] != context["root_rgb_sha256"]
    restored, goal = restore_execution_snapshot(env, seed=19, snapshot=advanced["execution_snapshot"])
    assert np.array_equal(restored, advanced["root_frame"])
    assert np.array_equal(goal, advanced["goal"])
