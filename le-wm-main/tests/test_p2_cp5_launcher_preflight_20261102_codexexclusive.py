import json

import pytest

from experiments.spectral_reachability import p2_cp5_launcher_preflight_20261102_codexexclusive as m


class FakePhysical:
    @staticmethod
    def verify(path, digest):
        assert (path, digest) == ("prepared", "a" * 64)
        return {"source_slots": [0], "source_roots": ["root"],
                "warm_traces": ["trace"], "plans": [{"root_slot": 0, "task_id": "task"}]}, {
                    "roots": [{"slot": 0, "warm_trace": "trace"}], "source_plan": {}}

    class native:
        class allmode:
            @staticmethod
            def root_metadata(data):
                return ["root"]

    @staticmethod
    def validate_env(body):
        if not body.valid:
            raise ValueError("invalid native body")


class Body:
    valid = True
    render_action = False
    render_size = 224
    render_mode = "rgb_array"


def test_source_and_env_preflight():
    p, data, plan = m.verified_source(FakePhysical, "prepared", "a" * 64, 0, "task")
    assert plan == p["plans"][0] and data["roots"][0]["slot"] == 0
    m.validate_environment(FakePhysical, Body())
    with pytest.raises(ValueError, match="task"):
        m.verified_source(FakePhysical, "prepared", "a" * 64, 0, "wrong")
    bad = Body()
    bad.render_size = 128
    with pytest.raises(ValueError, match="renderer"):
        m.validate_environment(FakePhysical, bad)


def test_aggregate_attempts_persist_and_cap_prevents_dispatch(tmp_path):
    ledger = m.Ledger(tmp_path / "ledger.jsonl", {"public_step": 2})
    assert ledger.call("public_step", lambda: 3, session="a") == 3
    with pytest.raises(RuntimeError):
        ledger.call("public_step", lambda: (_ for _ in ()).throw(RuntimeError("failure")), session="b")
    called = []
    with pytest.raises(ValueError, match="cap"):
        ledger.call("public_step", lambda: called.append(True), session="c")
    assert called == []
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [x["kind"] for x in rows] == ["contract", "attempt", "returned", "attempt", "failed", "cap_rejected"]
    assert ledger.attempted["public_step"] == 2 and ledger.returned["public_step"] == 1


def test_missing_feedback_model_consumer_rejects_before_any_environment():
    invoked = []
    with pytest.raises(ValueError, match="consumer"):
        m.require_feedback_winner(None, "a" * 64, "arm", lambda: invoked.append(True))
    assert not invoked


def test_session_identity_or_trace_tampering_rejected():
    class Altered(FakePhysical):
        @staticmethod
        def verify(path, digest):
            p, data = FakePhysical.verify(path, digest)
            p["warm_traces"] = ["tampered"]
            return p, data
    with pytest.raises(ValueError, match="source"):
        m.verified_source(Altered, "prepared", "a" * 64, 0, "task")


def test_real_meter_wrapper_limits_across_fake_sessions(tmp_path):
    class FakeMeter:
        def __init__(self):
            self.where = {}

        def call(self, kind, fn, **kwargs):
            return fn()

    ledger = m.Ledger(tmp_path / "nested.jsonl", {"public_step": 2, "physics_step": 2})
    first = m.AggregateMeter(FakeMeter(), ledger, "one")
    second = m.AggregateMeter(FakeMeter(), ledger, "two")
    first.call("public_step", lambda: first.call("physics_step", lambda: 1))
    second.call("public_step", lambda: second.call("physics_step", lambda: 1))
    with pytest.raises(ValueError, match="cap"):
        second.call("public_step", lambda: pytest.fail("dispatched over cap"))
    assert ledger.attempted == {"public_step": 2, "physics_step": 2}


def test_ledger_chain_rejects_tampering(tmp_path):
    path = tmp_path / "chain.jsonl"
    ledger = m.Ledger(path, {"public_step": 2})
    ledger.call("public_step", lambda: None, session="one")
    ledger.close()
    assert m.check_chain(path)["count"] == 3
    data = path.read_bytes().replace(b'"session":"one"', b'"session":"two"', 1)
    path.write_bytes(data)
    with pytest.raises(ValueError, match="chain"):
        m.check_chain(path)
