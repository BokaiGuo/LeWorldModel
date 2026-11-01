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


def test_execution_snapshot_prefers_full_physics_state_when_available():
    class _Physics:
        def __init__(self):
            self.data = SimpleNamespace(qpos=np.zeros(2, dtype=np.float64), qvel=np.zeros(2, dtype=np.float64))
            self.hidden = np.asarray([0.0], dtype=np.float64)

        def get_state(self):
            return np.concatenate([self.data.qpos, self.data.qvel, self.hidden])

        def set_state(self, state):
            state = np.asarray(state, dtype=np.float64)
            self.data.qpos[:] = state[:2]
            self.data.qvel[:] = state[2:4]
            self.hidden[:] = state[4:]

        def forward(self):
            return None

    env = _SnapshotEnv()
    env.env.physics = _Physics()
    context = collect_executed_context(env, seed=23, history_len=3, action_block=5)
    assert "physics_state" in context["execution_snapshot"]
    env.env.physics.hidden[:] = 99.0
    original_render = env.render

    def render_with_goal_side_effect():
        # Rendering the temporary qpos goal emulates a hidden simulator field
        # that can be disturbed by an off-root render.  The planning root must
        # still be completely restored before any physical prefix is executed.
        if np.allclose(env.env.physics.data.qpos, env.env.task.target_qpos):
            env.env.physics.hidden[:] = 7.0
        return original_render()

    env.render = render_with_goal_side_effect
    restore_execution_snapshot(env, seed=23, snapshot=context["execution_snapshot"])
    assert np.allclose(env.env.physics.hidden, 0.0)


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


@pytest.mark.parametrize("bound_goal", [False, True])
@pytest.mark.parametrize("forward_mutates_warmstart", [False, True])
def test_integration_snapshot_restores_time_control_and_warmstart(bound_goal, forward_mutates_warmstart):
    class _Physics:
        def __init__(self):
            self.model = SimpleNamespace(ptr=object())
            self.data = SimpleNamespace(ptr=object(), qpos=np.zeros(2), qvel=np.zeros(2))
            self.integration_only = np.array([12.0, 0.25, -0.7])
            self.model.to_bytes = lambda: b"synthetic-compiled-scene"

        def get_state(self, sig=None):
            basic = np.concatenate((self.data.qpos, self.data.qvel))
            return basic if sig is None else np.concatenate((basic, self.integration_only))

        def set_state(self, state, sig=None):
            self.data.qpos[:] = state[:2]
            self.data.qvel[:] = state[2:4]
            if sig is not None:
                self.integration_only[:] = state[4:]

        def forward(self):
            if forward_mutates_warmstart:
                self.integration_only[-1] = np.nextafter(self.integration_only[-1], np.inf)

    env = _SnapshotEnv()
    if bound_goal:
        env._goal = np.full((2, 2, 3), 99, np.uint8)
    env.env.physics = _Physics()
    context = collect_executed_context(env, seed=23, history_len=3, action_block=5)
    snapshot = context["execution_snapshot"]
    assert "physics_integration_state" in snapshot
    env.env.physics.integration_only[:] = 99
    restore_execution_snapshot(env, seed=23, snapshot=snapshot)
    np.testing.assert_array_equal(env.env.physics.integration_only, [12.0, 0.25, -0.7])
    incomplete = dict(snapshot)
    del incomplete["physics_state"]
    with pytest.raises(ValueError, match="complete physics-state restore"):
        restore_execution_snapshot(env, seed=23, snapshot=incomplete)
    incomplete = dict(snapshot)
    del incomplete["integration_state_spec"]
    with pytest.raises(ValueError, match="incomplete integration"):
        restore_execution_snapshot(env, seed=23, snapshot=incomplete)


def test_snapshot_replays_the_bound_goal_without_off_root_render():
    env = _SnapshotEnv()
    env._goal = np.full((2, 2, 3), 99, dtype=np.uint8)
    context = collect_executed_context(env, seed=23, history_len=3, action_block=5)
    snapshot = context["execution_snapshot"]
    assert "rendered_goal_rgb" in snapshot
    original_render = env.render
    def reject_off_root_goal_render():
        if np.array_equal(env.env.physics.data.qpos, env.env.task.target_qpos):
            raise AssertionError("saved goal must not be re-rendered off-root")
        return original_render()
    env.render = reject_off_root_goal_render
    root, goal = restore_execution_snapshot(env, seed=23, snapshot=snapshot)
    np.testing.assert_array_equal(root, context["root_frame"])
    np.testing.assert_array_equal(goal, context["goal"])
    corrupt = dict(snapshot)
    corrupt["rendered_goal_rgb"] = {**snapshot["rendered_goal_rgb"], "sha256": "tampered"}
    with pytest.raises(ValueError, match="RGB hash mismatch"):
        restore_execution_snapshot(env, seed=23, snapshot=corrupt)


def test_verified_root_path_checks_exact_state_without_a_second_root_render(monkeypatch):
    from experiments.spectral_reachability import reacher_d2p_context as runtime
    env = _SnapshotEnv()
    context = collect_executed_context(env, seed=23, history_len=3, action_block=5)
    def capture(unused):
        return {"physics_integration_state": env.env.physics.data.qpos.tolist(), "integration_state_spec": 42,
                "model_binary_sha256": "synthetic-scene", "target_qpos": env.env.task.target_qpos.tolist(),
                "rendered_goal_rgb": {"sha256": "synthetic-bound-goal"}}
    context["execution_snapshot"] = capture(env)
    monkeypatch.setattr(runtime, "capture_execution_snapshot", capture)
    original_render = env.render
    before_steps = env.step_count
    def render():
        assert env.step_count > before_steps, "a second pre-action render is forbidden on this path"
        return original_render()
    env.render = render
    advanced = advance_executed_context(env, context=context, raw_action_block=np.full((5, 2), 0.1),
                                       action_block=5, verified_root_frame=context["root_frame"])
    assert advanced["root_identity_validation"] == "strict_restored_rgb_then_fresh_exact_integration_and_scene"
    # The identical saved RGB must not authorize an actually stale state.
    with pytest.raises(RuntimeError, match="stale integration/scene"):
        advance_executed_context(env, context=context, raw_action_block=np.zeros((5, 2)),
                                 action_block=5, verified_root_frame=context["root_frame"])


def test_verified_rgb_path_rejects_legacy_incomplete_snapshots():
    env = _SnapshotEnv()
    context = collect_executed_context(env, seed=23, history_len=3, action_block=5)
    with pytest.raises(ValueError, match="complete integration/scene"):
        advance_executed_context(env, context=context, raw_action_block=np.zeros((5, 2)),
                                 action_block=5, verified_root_frame=context["root_frame"])


def _complete_sealed_context():
    from experiments.spectral_reachability.reacher_d2p_context import pack_rgb_observation
    env = _SnapshotEnv()
    physics = env.env.physics
    physics.model = SimpleNamespace(ptr=object(), to_bytes=lambda: b"complete-scene")
    physics.data.ptr = object()
    physics.named = SimpleNamespace(model=SimpleNamespace(geom_pos={"target": np.array([0.1, 0.2, 0.01])}))
    physics.get_state = lambda sig=None: np.concatenate((physics.data.qpos, physics.data.qvel))
    def set_state(state, sig=None):
        physics.data.qpos[:] = state[:2]
        physics.data.qvel[:] = state[2:]
    physics.set_state = set_state
    env._goal = np.full((2, 2, 3), 99, np.uint8)
    context = collect_executed_context(env, seed=23, history_len=3, action_block=5)
    context["execution_snapshot"]["observed_root_rgb"] = pack_rgb_observation(context["root_frame"])
    return env, context


def test_sealed_observation_replay_uses_exact_state_and_original_image_not_rerender():
    env, context = _complete_sealed_context()
    env.render = lambda: (_ for _ in ()).throw(AssertionError("sealed root must not be rerendered"))
    root, goal = restore_execution_snapshot(env, seed=23, snapshot=context["execution_snapshot"],
                                            root_identity="sealed_observation_exact_state")
    np.testing.assert_array_equal(root, context["root_frame"])
    np.testing.assert_array_equal(goal, context["goal"])


@pytest.mark.parametrize("field", ["model_binary_sha256", "physics_integration_state", "target_marker_pos"])
def test_sealed_root_rejects_changed_causal_or_scene_proof(field):
    env, context = _complete_sealed_context()
    snapshot = dict(context["execution_snapshot"])
    # Changing the expected proof is rejected even with identical saved RGB.
    if field == "model_binary_sha256":
        snapshot[field] = "different-scene"
    else:
        snapshot[field] = list(snapshot[field])
        snapshot[field][0] += 0.5
        # Do not allow mutation of the integration vector to alter the stored
        # qpos while still passing the full-state consistency check.
    with pytest.raises(RuntimeError):
        if field == "target_marker_pos":
            # Marker is restored from the snapshot; the complete compiled
            # model hash must also bind that geometric edit in a real model.
            env.env.physics.model.to_bytes = lambda: b"changed-marker-scene"
        restore_execution_snapshot(env, seed=23, snapshot=snapshot, root_identity="sealed_observation_exact_state")


def test_sealed_root_rejects_corrupt_observation_or_legacy_snapshot():
    env, context = _complete_sealed_context()
    snapshot = dict(context["execution_snapshot"])
    snapshot["observed_root_rgb"] = {**snapshot["observed_root_rgb"], "sha256": "wrong"}
    with pytest.raises(ValueError, match="RGB hash mismatch"):
        restore_execution_snapshot(env, seed=23, snapshot=snapshot, root_identity="sealed_observation_exact_state")
    del snapshot["observed_root_rgb"]
    with pytest.raises(ValueError, match="missing"):
        restore_execution_snapshot(env, seed=23, snapshot=snapshot, root_identity="sealed_observation_exact_state")
