import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p1_physical_callback_guard_20261101_leibniz as p


class Store:
    def __init__(self):
        self.events = []

    def check(self):
        pass

    def emit(self, kind, **kwargs):
        self.events.append((kind, kwargs))


def guards(tmp_path, caps=None, total_caps=None):
    paths = [Path(x.__file__).resolve() for x in (p, p.total, p.legacy, p.moving)]
    c = {"schema": p.total.NAME + ".contract.v1", "run_id": "physical-test", "arm": "cem",
         "caps": dict.fromkeys(p.total.AXES, 100), "identity": dict.fromkeys(p.total.IDENTITY, "a" * 64),
         "source_sha256": {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
         "shared_initialization": "full_standalone_charge_each_arm", "hook_modules": p.total.HOOKS,
         "scope": "per_root_per_arm_start_before_load"}
    if total_caps:
        c["caps"].update(total_caps)
    guard = p.total.Guard(tmp_path / "total.jsonl", c, p.total.digest(c))
    limits = dict.fromkeys(p.RESOURCES, 20)
    if caps:
        limits.update(caps)
    contract = {"schema": p.NAME + ".contract.v1", "total_contract_sha256": p.total.digest(c), "caps": limits}
    physical = p.PhysicalGuard(tmp_path / "physical.jsonl", guard, limits, p.total.digest(contract))
    return guard, physical


def events(receipt):
    path = Path(receipt["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    previous = None
    for index, row in enumerate(rows):
        sha = row.pop("sha256")
        assert row["previous"] == previous and row["index"] == index
        assert p.total.digest(row) == sha
        previous = sha
    return rows


class FakeEnvironment:
    def __init__(self):
        self._setup()

    def _setup(self):
        self.space = SimpleNamespace(step=lambda: "physics")

    def reset(self):
        self._set_state(1)
        self._set_goal_state(2)
        self.render()

    def _set_state(self, value):
        return value

    def _set_goal_state(self, value):
        return value

    def render(self):
        return "rgb"

    def step(self, action):
        return self.space.step()

    def close(self):
        return None


def test_real_meter_and_metered_class_nested_callback_accounting(tmp_path):
    guard, physical = guards(tmp_path)
    store = Store()
    meter = p.moving.Meter(store)
    original_call = meter.call
    cls = p.legacy.metered_class(FakeEnvironment, meter)
    with p.physical_meter_callbacks(physical, meter):
        env = meter.call("constructor", cls)
        env.reset()
        meter.region = "goal_bind"
        meter.call("goal_bind", lambda: env._set_goal_state(3))
        meter.region = "warmup"
        assert env.step(0) == "physics"
        assert env.render() == "rgb"
        meter.call("close", env.close)
    assert meter.call == original_call and "close" not in meter.caps
    expected = dict.fromkeys(p.RESOURCES, 1)
    expected["goal_setter"] = 2
    assert physical.charged == expected
    assert guard.charged["reset_calls"] == 1
    assert guard.charged["render_calls"] == 2
    assert guard.charged["physical_dispatch_calls"] == 1
    assert guard.charged["physics_steps"] == 1
    receipt = physical.close()
    assert p.verify_journal(receipt["path"], receipt["sha256"], p.total.digest(physical.contract))["charged"] == expected
    rows = events(receipt)
    assert len([x for x in rows if x["kind"] == "attempt"]) == 11
    assert len([x for x in rows if x["kind"] == "returned"]) == 11
    assert receipt["status"] == guard.close()["status"] == "closed_unverified"


@pytest.mark.parametrize("resource", p.RESOURCES)
def test_callback_failure_charged_and_only_close_allowed(tmp_path, resource):
    guard, physical = guards(tmp_path)
    meter = p.moving.Meter(Store())
    with p.physical_meter_callbacks(physical, meter):
        with pytest.raises(RuntimeError, match="failure"):
            meter.call(resource, lambda: (_ for _ in ()).throw(RuntimeError("failure")))
        assert physical.charged[resource] == 1
        if resource != "close":
            with pytest.raises(ValueError, match="fail-stopped"):
                meter.call("reset", lambda: None)
            assert meter.call("close", lambda: "cleanup") == "cleanup"
    receipt = physical.close()
    assert p.verify_journal(receipt["path"], receipt["sha256"], p.total.digest(physical.contract))["status"] == "failed_or_blocked"
    rows = events(receipt)
    assert any(x["kind"] == "failure" for x in rows)
    assert receipt["status"] == guard.close()["status"] == "failed_or_blocked"


def test_physical_and_total_caps_block_before_callback(tmp_path):
    guard, physical = guards(tmp_path, caps={"set_state": 0}, total_caps={"reset_calls": 0})
    meter = p.moving.Meter(Store())
    calls = []
    with p.physical_meter_callbacks(physical, meter):
        with pytest.raises(ValueError, match="physical cap"):
            meter.call("set_state", lambda: calls.append("state"))
        assert meter.call("close", lambda: None) is None
    assert calls == []
    assert physical.close()["status"] == guard.close()["status"] == "failed_or_blocked"


def test_total_cap_blocks_actual_callback(tmp_path):
    guard, physical = guards(tmp_path, total_caps={"reset_calls": 0})
    meter = p.moving.Meter(Store())
    calls = []
    with p.physical_meter_callbacks(physical, meter):
        with pytest.raises(ValueError, match="budget exceeded"):
            meter.call("reset", lambda: calls.append("reset"))
    assert not calls and physical.charged["reset"] == 1
    assert physical.close()["status"] == guard.close()["status"] == "failed_or_blocked"


def test_swallowed_nested_failure_does_not_return_success(tmp_path):
    guard, physical = guards(tmp_path)
    meter = p.moving.Meter(Store())
    def outer():
        try:
            meter.call("set_state", lambda: (_ for _ in ()).throw(RuntimeError("inner")))
        except RuntimeError:
            return "ignored"
    with p.physical_meter_callbacks(physical, meter):
        with pytest.raises(ValueError, match="guard violation"):
            meter.call("constructor", outer)
    receipt = physical.close()
    assert p.verify_journal(receipt["path"], receipt["sha256"], p.total.digest(physical.contract))["status"] == "failed_or_blocked"
    assert guard.close()["status"] == "failed_or_blocked"


@pytest.mark.parametrize("method,args", [("begin", ("reset",)), ("end", (0,)), ("call", ("other", lambda: None))])
def test_direct_or_unknown_meter_bypass_rejected(tmp_path, method, args):
    guard, physical = guards(tmp_path)
    meter = p.moving.Meter(Store())
    with p.physical_meter_callbacks(physical, meter):
        with pytest.raises(ValueError):
            getattr(meter, method)(*args)
    assert physical.close()["status"] == guard.close()["status"] == "failed_or_blocked"


def test_readiness_unchanged():
    from experiments.spectral_reachability import p1_visual_live_host_wiring_20261101_leibniz as host
    with pytest.raises(ValueError, match="fail-closed"):
        host.run_cem()


def test_precaptured_begin_is_detected_at_scope_exit(tmp_path):
    guard, physical = guards(tmp_path)
    meter = p.moving.Meter(Store())
    old_begin = meter.begin
    with pytest.raises(ValueError, match="accounting mismatch"):
        with p.physical_meter_callbacks(physical, meter):
            old_begin("reset")
    assert physical.charged["reset"] == 0 and meter.counts["reset"]["attempted"] == 1
    # The independent host gate remains closed despite this isolated scope.
    from experiments.spectral_reachability import p1_visual_live_host_wiring_20261101_leibniz as host
    with pytest.raises(ValueError, match="fail-closed"):
        host.run_structured()
    assert physical.close()["status"] == guard.close()["status"] == "failed_or_blocked"
