import numpy as np
import pytest
from experiments.spectral_reachability import codex_p3_six_physical_adapter_20261030_v1 as a


class Env:
    def __init__(self, corrupt=False):
        self.unwrapped = self
        self.corrupt = corrupt
    def step(self, command):
        self.latest_action = command.copy()
        if self.corrupt:
            self.latest_action[0] = -1
        return None, 0, False, False, {}


class Store:
    def __init__(self):
        self.events = []
    def emit(self, kind, **fields):
        self.events.append((kind, fields))


def test_adapter_uses_original_branch_and_restores_step(monkeypatch):
    def branch(env, ref, src, plan, row, ops, store, index):
        for _ in range(10):
            env.step(np.zeros(2, np.float32))
        for action in np.asarray(plan["commands"], np.float32):
            env.step(action)
        row["status"] = "horizon_completed"
    monkeypatch.setattr(a.moving, "branch", branch)
    env, store = Env(), Store()
    ops = type("Ops", (), {"where": {"branch": 0}})()
    actions = np.full((10, 2), 0.25, np.float32).astype(float).tolist()
    out = a.run_branch(env, None, None, {"commands": actions}, {}, ops, store, 0)
    assert out["returned_steps"] == 10 and "step" not in env.__dict__
    assert len([e for e in store.events if e[0] == "step_ack_observed"]) == 20


def test_ack_mismatch_records_actual_then_fails(monkeypatch):
    def branch(env, *args):
        env.step(np.zeros(2, np.float32))
    monkeypatch.setattr(a.moving, "branch", branch)
    env, store = Env(corrupt=True), Store()
    ops = type("Ops", (), {"where": {}})()
    with pytest.raises(ValueError, match="ACK bytes"):
        a.run_branch(env, None, None, {"commands": np.zeros((10, 2)).tolist()}, {}, ops, store, 0)
    assert store.events[0][0] == "step_ack_observed"
    assert "step" not in env.__dict__
