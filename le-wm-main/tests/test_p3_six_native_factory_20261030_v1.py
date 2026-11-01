"""CPU-only injected factory tests; never constructs the real environment."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p3_six_native_factory_20261030_v1 as m
from experiments.spectral_reachability import p3_six_lifecycle_20261030_v1 as lifecycle


class Store:
    def __init__(self):
        self.frames, self.index, self.output = 0, 0, Path("/fixture")
        self.events, self.writes = [], {}
        self.fail_ready = False

    def check(self, category, size):
        assert category == "physical_png" and size == 192 * m.PNG_CAP

    def event(self, origin, kind, resource, token, detail):
        if self.fail_ready and resource == "six_session_ready":
            raise OSError("journal failure")
        self.events.append((origin, kind, resource, detail))
        self.index += 1

    def write(self, name, data, category):
        assert category == "physical_png"
        if name in self.writes:
            raise ValueError("duplicate file")
        self.writes[name] = data


class Raw:
    def __init__(self, close_error=False):
        self.calls = 0
        self.close_error = close_error

    def close(self):
        self.calls += 1
        if self.close_error:
            raise RuntimeError("close error")


def make(*, fail=None, close_error=False, store=None):
    store = store or Store()
    roots = [dict(root_sha256=c * 64) for c in ("a", "b")]
    prepared = dict(tasks=[dict(binding=dict(root_sha256=r["root_sha256"])) for r in roots])
    source = dict(roots=roots, source_plan={})
    created = []
    def constructor(meter):
        if fail == "constructor":
            raise RuntimeError("constructor error")
        raw = Raw(close_error)
        created.append(raw)
        return raw
    class Traced:
        def __init__(self, raw, store, episode):
            if fail == "wrapper":
                raise RuntimeError("wrapper error")
            self.raw = raw
        def close(self):
            return self.raw.close()
    def setup(env, root, plan, task, row, meter, capture):
        if fail == "setup":
            raise RuntimeError("setup error")
        if fail == "initial_stop":
            raise m.native.NativeInitialStop("stop")
        return "goal", "goal_rgb", "prior"
    class Session:
        def __init__(self, env, goal, rgb, meter, capture, row, model, torch, device):
            if fail == "session":
                raise RuntimeError("session error")
            self.episode_id = row["episode_id"]
            self.capture = capture
            self.goal = goal
    meter = SimpleNamespace(Meter=lambda store, episode: SimpleNamespace(episode=episode),
                            native_factory=constructor)
    module = SimpleNamespace(setup=setup, Session=Session)
    factory = m.Factory(prepared, source, store, None, None, None,
                        meter_module=meter, native_module=module,
                        reset_module=SimpleNamespace(TracedEnv=Traced))
    return factory, created, store, prepared, source


def test_six_order_and_session_cleanup():
    factory, created, store, _, _ = make()
    with pytest.raises(ValueError):
        factory(3, 1, "fixed_order")
    for slot, root, policy in m.control.ROSTER:
        session, prior, close = factory(slot, root, policy)
        assert session.episode_id == f"matched_slot_{slot:02}" and prior == "prior"
        if slot < 5:
            with pytest.raises(ValueError):
                factory(*m.control.ROSTER[slot + 1])
        close()
        with pytest.raises(ValueError):
            close()
    assert len(created) == 6 and all(raw.calls == 1 for raw in created)
    assert len(store.events) == 6 and not factory.preflight["storage_admitted"]
    with pytest.raises(ValueError):
        factory(0, 0, "fixed_order")


@pytest.mark.parametrize("failure", ["wrapper", "setup", "session", "initial_stop", "journal"])
def test_setup_failure_closes_original_once_and_never_retries(failure):
    factory, created, store, _, _ = make(fail=failure)
    store.fail_ready = failure == "journal"
    with pytest.raises((RuntimeError, OSError, m.native.NativeInitialStop)):
        factory(0, 0, "fixed_order")
    assert len(created) == 1 and created[0].calls == 1
    with pytest.raises(ValueError):
        factory(0, 0, "fixed_order")


def test_constructor_failure_consumes_attempt_without_close():
    factory, created, _, _, _ = make(fail="constructor")
    with pytest.raises(RuntimeError):
        factory(0, 0, "fixed_order")
    assert not created
    with pytest.raises(ValueError):
        factory(0, 0, "fixed_order")
    with pytest.raises(ValueError):
        factory(1, 0, "signal_order")


def test_cleanup_failure_poisons_factory_and_preserves_cause():
    factory, created, _, _, _ = make(fail="setup", close_error=True)
    with pytest.raises(RuntimeError, match="unsuccessful cleanup") as error:
        factory(0, 0, "fixed_order")
    assert str(error.value.__cause__) == "setup error" and created[0].calls == 1
    with pytest.raises(ValueError):
        factory(1, 0, "signal_order")
    factory, created, _, _, _ = make(close_error=True)
    _, _, close = factory(0, 0, "fixed_order")
    with pytest.raises(RuntimeError, match="close error"):
        close()
    with pytest.raises(ValueError):
        factory(1, 0, "signal_order")
    assert created[0].calls == 1


def test_preflight_rejects_bad_roots_and_storage_before_constructor():
    _, _, store, prepared, source = make()
    source["roots"][1]["root_sha256"] = "z" * 64
    with pytest.raises(ValueError):
        m.preflight(prepared, source, store)
    store.frames = 1
    with pytest.raises(ValueError):
        m.preflight(prepared, source, store)
    class Insufficient(Store):
        def check(self, category, size):
            raise ValueError("no PNG capacity")
    with pytest.raises(ValueError, match="no PNG capacity"):
        make(store=Insufficient())


def test_per_slot_and_global_frame_cap_and_invalid_image():
    store = Store()
    capture = m.CaptureStore(store)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(32):
        capture.frame(f"frame{i}.png", image)
    with pytest.raises(ValueError, match="frame limits"):
        capture.frame("frame32.png", image)
    capture = m.CaptureStore(store)
    with pytest.raises(ValueError, match="RGB224"):
        capture.frame("bad.png", np.zeros((2, 2), dtype=np.uint8))
    store.frames = 191
    capture.frame("frame191.png", image)
    with pytest.raises(ValueError, match="frame limits"):
        capture.frame("frame192.png", image)
    assert store.frames == 192


def test_contract_wraps_prior_seal(monkeypatch):
    from experiments.spectral_reachability import p3_six_host_prepare_20261030_v1 as prior
    monkeypatch.setattr(prior, "build", lambda: {"test_fixture": True})
    value = m.source_contract()
    assert value["frame_cap"] == 192 and not value["host_runner_ready"]
    m.verify_source_contract(value)
    value["frame_cap"] = 128
    with pytest.raises(ValueError):
        m.verify_source_contract(value)


def test_factory_wires_fixed_six_lifecycle_without_native_execution(monkeypatch):
    factory, created, store, prepared, _ = make()
    def callback(root, policy, raw, session, backend, sink, permit_ram=False):
        assert raw == "prior" and session.goal == "goal"
        return {"policy": policy, "status": "fixture_complete"}
    monkeypatch.setattr(lifecycle.control, "execute_slot", callback)
    result = lifecycle.execute(prepared["tasks"], store, factory, None, permit_ram=True)
    assert [x["status"] for x in result["rows"]] == ["fixture_complete"] * 6
    assert all(x["cleanup"] == "returned" for x in result["rows"])
    assert len(created) == 6 and all(x.calls == 1 for x in created)
    assert not result["physical_certified"] and not result["scientific_pass"]
