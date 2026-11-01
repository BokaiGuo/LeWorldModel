"""CPU mock checks, not physics evidence."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p1_devroot_witness_producer_20261101_hypatia as p

NOISY_IMAGE = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)


class Space:
    def step(self, dt):
        pass


class Body:
    def __init__(self):
        self.agent = SimpleNamespace(position=(0., 0.), velocity=(0., 0.), angle=0.,
                                     angular_velocity=0., force=(0., 0.), torque=0., is_sleeping=False)
        self.block = SimpleNamespace(position=(2., 0.), velocity=(0., 0.), angle=0.,
                                     angular_velocity=0., force=(0., 0.), torque=0., is_sleeping=False,
                                     shapes=[])
        self.goal_pose = np.zeros(3)
        self.render_action, self.render_size = False, 224
        self._goal = np.zeros((224, 224, 3), dtype=np.uint8)

    def _get_obs(self):
        return np.array([*self.agent.position, *self.block.position, self.block.angle, 0., 0.])

    def _set_state(self, value):
        self.agent.position = tuple(value[:2])
        self.block.position = tuple(value[2:4])
        self.block.angle = value[4]
        self.space.step(.01)

    def _set_goal_state(self, value):
        self.goal_state = value.copy()

    def render(self):
        return NOISY_IMAGE.copy()

    def eval_state(self, goal, state):
        dist = float(np.linalg.norm(goal - state))
        return (dist < 1e-10, dist)


class Env:
    def __init__(self):
        self.unwrapped = Body()

    def reset(self, seed, options=None):
        b = self.unwrapped
        b.space = Space()
        goal = np.array([100., 100., 100., 100., 0., 0., 0.]) if options is None else options["goal_state"]
        state = np.array([0., 0., 2., 0., 0., 0., 0.]) if options is None else options["state"]
        b._set_state(goal)
        b._set_goal_state(goal)
        b._goal = b.render()
        b._set_state(state)
        return {"state": b._get_obs()}, {"goal_state": goal}

    def step(self, action):
        b = self.unwrapped
        b.latest_action = action.copy()
        for _ in range(10):
            b.space.step(.01)
        b.agent.position = tuple(np.array(b.agent.position) + action.astype(float))
        state = b._get_obs()
        success, distance = b.eval_state(b.goal_state, state)
        return {"state": state}, -distance, success, False, {}

    def close(self):
        pass


def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(p.moving, "validate_env", lambda body: None)
    monkeypatch.setattr(p.moving, "guard", lambda rgb, state: None)
    monkeypatch.setattr(p.v4.source, "sample_state", lambda *a: np.array([0., 0., 2., 0., 0., 0., 0.]))
    packet = {"metadata": {"selected_attempt": {"random_reset_seed": 1, "attempt_index": 0},
                           "source": {"proposal_index": 0}},
              "state": np.array([0., 0., 2., 0., 0., 0., 0.]),
              "goal": np.array([100., 100., 100., 100., 0., 0., 0.])}
    return p.accounting.Ledger(tmp_path / "run", min_free_bytes=0), packet, {"slots": [{}, {}]}


def test_full_two_slots_three_phases_budget_and_replay(monkeypatch, tmp_path):
    ledger, packet, plan = setup(monkeypatch, tmp_path)
    for slot in (0, 1):
        c = p.phase(slot, "construction", plan, packet, ledger, Env)
        assert len(c["future"]) == 10
        for name in ("replay0", "replay1"):
            assert p.phase(slot, name, plan, packet, ledger, Env, c)["steps"] == 10
    expected = {"reset": 12, "set_state": 24, "goal_setter": 16, "goal_bind": 4,
                "warmup_step": 60, "witness_step": 60, "physics_step": 1224,
                "explicit_render": 126, "goal_render": 12}
    assert {k: v["completed"] for k, v in ledger.counts.items()} == expected
    assert ledger.bytes_written > 18 * 1024 * 1024
    assert ledger.bytes_written < ledger.max_bytes - ledger.reserve_bytes


def test_state_binding_rejects_before_warmup(monkeypatch, tmp_path):
    ledger, packet, plan = setup(monkeypatch, tmp_path)
    packet["state"][0] = 1.
    with pytest.raises(ValueError, match="source state/goal binding"):
        p.phase(0, "construction", plan, packet, ledger, Env)
    assert ledger.counts["warmup_step"]["attempted"] == 0


def test_early_stop_keeps_partial_and_does_not_construct_goal(monkeypatch, tmp_path):
    ledger, packet, plan = setup(monkeypatch, tmp_path)
    class Stopped(Env):
        def step(self, action):
            obs, reward, _, _, info = super().step(action)
            return obs, reward, True, False, info
    with pytest.raises(p.SlotFailure, match="warmup stopped"):
        p.phase(0, "construction", plan, packet, ledger, Stopped)
    assert ledger.counts["warmup_step"]["completed"] == 1
    assert ledger.counts["witness_step"]["attempted"] == 0
    assert (ledger.directory / "s0_construction_01.png").exists()


def test_replay_snapshot_mismatch_is_rejected(monkeypatch, tmp_path):
    ledger, packet, plan = setup(monkeypatch, tmp_path)
    c = p.phase(0, "construction", plan, packet, ledger, Env)
    c["warm"][5]["snapshot"]["state_float64_bits"] = "tampered"
    with pytest.raises(ValueError, match="warmup snapshot mismatch"):
        p.phase(0, "replay0", plan, packet, ledger, Env, c)
    assert ledger.counts["goal_bind"]["attempted"] == 0


def test_preregister_hash_drift_refuses_prepare(monkeypatch):
    monkeypatch.setattr(p.v4, "sha", lambda _: "0" * 64)
    with pytest.raises(ValueError, match="preregistration or v4 prepare drift"):
        p.prepared()
