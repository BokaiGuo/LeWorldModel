import pytest
from contextlib import contextmanager
from experiments.spectral_reachability import codex_p3_six_physical_host_20261030_v1 as h
from experiments.spectral_reachability import codex_p3_six_physical_batch_saved_20261030_v1 as consumer


def test_existing_and_wrong_location_rejected(tmp_path):
    with pytest.raises(ValueError, match="exclusive"):
        h.exclusive(tmp_path / (h.NAME + "_RUN_x"), "_RUN_")


def test_incomplete_saved_terminal_refused():
    with pytest.raises((ValueError, KeyError)):
        consumer.validate({}, {"schema": h.NAME + ".report", "status": "partial_or_failed"}, [], None)


def test_contract_budget_and_no_gate_change():
    c = h.spec.contract()
    assert c["caps"]["environment_construct"] == 6
    assert c["caps"]["public_step"] == 120
    assert c["scientific_pass"] is False


def test_roster_failure_continues_and_cleanup(monkeypatch):
    class Store:
        def __init__(self):
            self.events, self.path, self.previous = [], None, "tail"
        def emit(self, kind, **fields):
            self.events.append({"kind": kind, **fields})
        def put(self, *args, **kwargs):
            return None
        def check(self, *args):
            return None
    class FakeEnv:
        def __init__(self):
            self.unwrapped = self
            self.space = self.render_buffer = self.latest_action = None
            self.render_action, self.render_size, self.render_mode = False, 224, "rgb_array"
        def close(self):
            pass
    @contextmanager
    def hooks(*args):
        yield
    def branch(env, ref, source, plan, row, ops, store, i):
        if i == 2:
            raise RuntimeError("fixture failure")
        row["status"] = "horizon_completed"
    monkeypatch.setattr(h.kernel.moving, "validate_env", lambda body: None)
    monkeypatch.setattr(h.kernel, "native_hooks", hooks)
    monkeypatch.setattr(h.adapter, "run_branch", branch)
    monkeypatch.setattr(h, "sha", lambda path: "fixture")
    store = Store()
    store.path = type("PathFixture", (), {"__truediv__": lambda self, x: x})()
    plans = [dict(root_slot=root, arm=arm, selection_digest=str(i))
             for i, (root, arm) in enumerate(h.spec.ORDER)]
    p = {"plans": plans, "contract": {**h.spec.contract(), "wall_seconds": 600}, "bindings": {}}
    result = h.execute(p, {"references": [None, None], "plan": {}}, store, FakeEnv)
    assert result["status"] == "partial_or_failed"
    assert len(result["rows"]) == 6
    assert result["rows"][2]["status"] == "failed"
    assert result["counts"]["environment_construct"] == {"attempted": 6, "completed": 6}
    assert result["counts"]["environment_close"] == {"attempted": 6, "completed": 6}
