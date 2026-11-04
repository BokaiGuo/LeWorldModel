from contextlib import nullcontext
from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import p2_conditional_physical_executor_20261102_exclusive as exe


class MemoryStore:
    def __init__(self):
        self.events = []

    def check(self, *args, **kwargs):
        return None

    def emit(self, kind, **kwargs):
        self.events.append({"kind": kind, **deepcopy(kwargs)})


class DummyEnv:
    def __init__(self):
        self.unwrapped = self
        self.render_action = False
        self.render_size = 224
        self.render_mode = "rgb_array"
        self.closed = False

    def close(self):
        self.closed = True


def real_plan():
    old = json.loads(exe.handoff.old.PREP.read_bytes())
    return exe._derived(old, exe._checked_recipe())


def test_real_recipe_new_witness_and_fixed_prefix():
    p = real_plan()
    assert [r["policy"] for r in p["plans"]] == list(exe.handoff.ARMS)
    assert p["plans"][0]["commands"][:15] == p["plans"][1]["commands"][:15]
    assert p["plans"][0]["new_model_plan"]["suffix_sha256"] != p["plans"][1]["new_model_plan"]["suffix_sha256"]
    original = json.loads(exe.handoff.old.PREP.read_bytes())
    with pytest.raises(ValueError):
        exe.physical.validate_execution(p)
    assert original["plans"] != p["plans"]
    p["plans"][0]["selection_digest"] = "bad"
    with pytest.raises(ValueError, match="conditional new"):
        exe._derived(original, {"plans": p["plans"]})


def test_normal_pair_uses_distinct_envs(monkeypatch):
    p = real_plan()
    store = MemoryStore()
    envs = []
    executed = []
    def factory():
        env = DummyEnv()
        envs.append(env)
        return env
    def episode(env, root, source_plan, plan, row, meter, store, fixed):
        executed.append((id(env), plan["raw_float32_hex"], fixed["prefix_reference"]))
        row.update(status="horizon_completed", own_returned=25)
    monkeypatch.setattr(exe.physical, "validate_env", lambda _: None)
    monkeypatch.setattr(exe.physical, "native_hooks", lambda *args: nullcontext())
    monkeypatch.setattr(exe.physical, "episode", episode)
    rows, meter, errors = exe.execute_pair(p, {"roots": [None], "source_plan": None}, store, factory)
    assert not errors and len(executed) == 2
    assert executed[0][0] != executed[1][0]
    assert executed[0][2] == executed[1][2] == p["prefix_reference"]
    assert [x[1] for x in executed] == [x["raw_float32_hex"] for x in p["plans"]]
    assert all(e.closed for e in envs)
    assert [r["status"] for r in rows] == ["horizon_completed"] * 2
    assert meter.counts["constructor"]["completed"] == 2


def test_early_stop_kept_and_engineering_failure_stops_pair(monkeypatch):
    p = real_plan()
    monkeypatch.setattr(exe.physical, "validate_env", lambda _: None)
    monkeypatch.setattr(exe.physical, "native_hooks", lambda *args: nullcontext())
    def stop(*args):
        raise exe.physical.native.allmode.NativeStop("fixture before suffix")
    monkeypatch.setattr(exe.physical, "episode", stop)
    rows, _, errors = exe.execute_pair(p, {"roots": [None], "source_plan": None}, MemoryStore(), DummyEnv)
    assert not errors and all(r["status"] == "native_stopped" and r["own_returned"] == 0 for r in rows)
    calls = []
    def broken(*args):
        calls.append(1)
        raise RuntimeError("injected failure")
    monkeypatch.setattr(exe.physical, "episode", broken)
    rows, _, errors = exe.execute_pair(p, {"roots": [None], "source_plan": None}, MemoryStore(), DummyEnv)
    assert len(calls) == 1 and len(errors) == 1
    assert rows[0]["status"] == "engineering_failed" and rows[1]["status"] == "not_started"


def test_auth_and_historical_view_fail_before_dispatch(monkeypatch):
    monkeypatch.setattr(exe, "verify", lambda *_: None)
    monkeypatch.setattr(exe.handoff.host, "file_sha", lambda *_: "not-history")
    with pytest.raises(ValueError, match="authorization"):
        exe.run_child("unused", "wrong")
    with pytest.raises(ValueError, match="historical view"):
        exe.run_child("unused", exe.AUTH)
