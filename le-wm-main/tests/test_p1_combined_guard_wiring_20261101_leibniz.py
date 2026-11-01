import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p1_combined_guard_wiring_20261101_leibniz as c

p, g, visual = c.p, c.g, c.visual


class Module:
    def __init__(self):
        self.hooks = []
    def register_forward_pre_hook(self, fn):
        self.hooks.append(fn)
        return SimpleNamespace(remove=lambda: self.hooks.remove(fn))
    def fire(self, sink, name, index):
        for fn in self.hooks:
            fn(self, ())
        entry = {"index": index, "resource": name, "completed": False, "image": 0}
        sink.event({"kind": "module_attempt", "entry": entry})
        sink.event({"kind": "module_return", "entry": {**entry, "completed": True}})


class Store:
    def __init__(self, path):
        self.path, self.events = path, []
    def check(self):
        pass
    def namespace(self, name):
        path = self.path / name
        path.mkdir()
        return path
    def emit(self, *args, **kwargs):
        self.events.append((args, kwargs))
    def put(self, *args, **kwargs):
        pass


def make(tmp_path):
    modules = (c, c.constructor, p, g, p.legacy, p.moving, visual, visual.encoding)
    paths = [Path(m.__file__).resolve() for m in modules]
    contract = {"schema": g.NAME + ".contract.v1", "run_id": "combined-test", "arm": "cem",
                "caps": dict.fromkeys(g.AXES, 20), "identity": dict.fromkeys(g.IDENTITY, "a" * 64),
                "source_sha256": {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
                "shared_initialization": "full_standalone_charge_each_arm", "hook_modules": g.HOOKS,
                "scope": "per_root_per_arm_start_before_load"}
    total = g.Guard(tmp_path / "total.jsonl", contract, g.digest(contract))
    model = SimpleNamespace(**{name: Module() for names in g.HOOKS.values() for name in names})
    total.install_model_sentinels(model)
    caps = dict.fromkeys(p.RESOURCES, 20)
    pc = {"schema": p.NAME + ".contract.v1", "total_contract_sha256": g.digest(contract), "caps": caps}
    physical = p.PhysicalGuard(tmp_path / "physical.jsonl", total, caps, g.digest(pc))
    store = Store(tmp_path)
    meter = p.moving.Meter(store)
    meter.episode = "test"
    return c.Combined(physical, meter, store), model, contract


def operation(model, combined):
    sink = visual.visual.Sink(combined.store, combined.meter, "sample", "encode")
    model.encoder.fire(sink, "encoder", 0)
    model.projector.fire(sink, "projector", 1)
    sink.complete()
    return "latent"


def finish(combined, contract):
    ph = combined.physical.close()
    total = combined.guard.close()
    return (p.verify_journal(ph["path"], ph["sha256"], g.digest(combined.physical.contract)),
            g.verify_journal(total["journal"], total["sha256"], g.digest(contract)))


def test_combined_actual_sink_and_physical_meter(tmp_path):
    combined, model, contract = make(tmp_path)
    with combined.scope():
        assert combined.meter.call("reset", lambda: "reset") == "reset"
        assert combined._visual(lambda: operation(model, combined), combined.meter, combined.store) == "latent"
        assert combined.meter.call("public_step", lambda: 1) == 1
    ph, total = finish(combined, contract)
    assert ph["charged"]["reset"] == ph["charged"]["public_step"] == 1
    assert total["charged"]["source_encode_calls"] == total["charged"]["reset_calls"] == 1
    assert total["charged"]["physical_dispatch_calls"] == 1
    assert total["status"] == "closed_unverified"


def test_real_pusht_cpu_plus_actual_sink_fake_neural(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import numpy as np
    from stable_worldmodel.envs.pusht.env import PushT
    native = p.legacy.kernel.moving.ENV
    assert hashlib.sha256(native.read_bytes()).hexdigest() == p.legacy.kernel.moving.ENV_SHA
    combined, model, contract = make(tmp_path)
    with combined.scope() as factory:
        env = combined.meter.call("constructor", lambda: factory(PushT)(
            render_mode="rgb_array", resolution=224, with_target=False, render_action=False))
        try:
            obs, _ = env.reset(seed=0)
            assert env.render().shape == (224, 224, 3)
            assert combined._visual(lambda: operation(model, combined), combined.meter, combined.store) == "latent"
            combined.meter.region = "warmup"
            _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
            assert not terminated and not truncated
        finally:
            combined.meter.call("close", env.close)
    ph, total = finish(combined, contract)
    assert ph["charged"]["constructor"] == ph["charged"]["close"] == 1
    assert ph["charged"]["physics_step"] == 12
    assert total["charged"]["source_encode_calls"] == 1
    assert total["status"] == "closed_unverified"


def test_combined_structured_real_capture_and_episode(tmp_path, monkeypatch):
    combined, model, contract = make(tmp_path)
    def setup(env, ref, plan, row, meter, store):
        meter.call("reset", lambda: None)
        return True
    monkeypatch.setattr(visual.moving, "setup", setup)
    def boundary(handler, runtime, task, layer, row, obs, body, meter, store, step):
        operation(model, combined)
        raise RuntimeError("stop after guarded boundary")
    monkeypatch.setattr(visual.moving, "encode_boundary", boundary)
    with combined.scope():
        with c.structured_wiring(combined) as configured:
            with pytest.raises(RuntimeError, match="stop after guarded"):
                configured.rgb.episode(SimpleNamespace(unwrapped="body"), {}, {"plan": {}, "visual_tasks": [0]},
                    "layer", {"root_context": {}}, {"last_observation": 0}, [{"boundary": 0}],
                    "handler", {}, combined.meter, combined.store, None)
    ph, total = finish(combined, contract)
    assert ph["charged"]["reset"] == 1 and total["charged"]["source_encode_calls"] == 1
    assert total["status"] == "failed_or_blocked"


def test_cem_passes_combined_proxy_to_execute_one(tmp_path, monkeypatch):
    combined, _, contract = make(tmp_path)
    runner = SimpleNamespace(rgb=SimpleNamespace(moving=visual.moving))
    calls = []
    def execute(*args):
        calls.append(args[6])
        return "reached real-host seam"
    monkeypatch.setattr(c.host.cem, "execute_one", execute)
    with combined.scope():
        assert c.cem_execute(combined, {}, {}, combined.store, None, {}, None, runner, None, 0) == "reached real-host seam"
        assert isinstance(calls[0].rgb.moving, c.MovingProxy)
        assert calls[0].rgb.moving.combined is combined and runner.rgb.moving is visual.moving
    assert finish(combined, contract)[1]["status"] == "closed_unverified"


def test_cem_real_execute_one_combined_callback(tmp_path, monkeypatch):
    combined, model, contract = make(tmp_path)
    cem = c.host.cem
    identity = {"root_sha256": "r", "task_sha256": "t", "model_sha256": "m"}
    monkeypatch.setattr(cem.source, "inspect", lambda root: {
        "rebound_goal_binding": "binding", "identity": identity})
    monkeypatch.setattr(cem, "fixed_identity_check", lambda *args: None)
    def setup(env, ref, plan, row, meter, store):
        meter.call("reset", lambda: None)
        row.update(initial_exact=True, last_observation="obs")
        return True
    monkeypatch.setattr(visual.moving, "setup", setup)
    def boundary(handler, runtime, task, layer, row, obs, body, meter, store, step):
        operation(model, combined)
        raise RuntimeError("stop after combined CEM boundary")
    monkeypatch.setattr(visual.moving, "encode_boundary", boundary)
    class Controller:
        def __init__(self, *, leaf, observe, **kwargs):
            self.leaf, self.observe = leaf, observe
            self.journal = SimpleNamespace(append=lambda payload: None)
        def step(self):
            self.observe(0, SimpleNamespace(encode=lambda *args, **kwargs: self.leaf.encode({}, phase="setup")))
        def close(self):
            pass
    monkeypatch.setattr(cem.cem, "Controller", Controller)
    store = combined.store
    store.previous, store.finalizing = None, False
    body = SimpleNamespace(with_target=False, render_action=False, render_size=224, relative=True,
                           action_scale=100, control_hz=10, dt=.01, _live_constructor_complete=True)
    def factory(meter):
        body._live_meter = meter
        return SimpleNamespace(unwrapped=body, close=lambda: None)
    def retain(row, exc, phase):
        row["first_error"] = str(exc)
    runner = SimpleNamespace(
        rgb=SimpleNamespace(moving=visual.moving, node_roster=lambda row: [{"boundary": 0}]),
        p3=SimpleNamespace(roster=lambda: [{"control_returned": 0, "first_error": None}],
                           Meter=lambda *args: combined.meter),
        old=SimpleNamespace(kernel=SimpleNamespace(retain=retain)),
        protocol=lambda: {}, sha=lambda path: "journal-hash")
    base = {"binding": {"root_sha256": "r", "task_sha256": "t"}, "root_context": {}}
    prepared = {"source": {"tasks": [base], "layers": ["layer"]}}
    data = {"references": [{"root_sha256": "r"}], "plan": {}, "visual_tasks": ["task"]}
    with combined.scope():
        result = c.cem_execute(combined, prepared, data, store, "handler",
                               {"model_sha256": "m", "check_limits": lambda: None},
                               factory, runner, lambda p, d: "binding", 0)
    assert result["errors"][0]["message"] == "stop after combined CEM boundary"
    ph, total = finish(combined, contract)
    assert ph["charged"]["constructor"] == ph["charged"]["reset"] == ph["charged"]["close"] == 1
    assert total["charged"]["source_encode_calls"] == 1 and total["status"] == "failed_or_blocked"


@pytest.mark.parametrize("mode", ["physical_during_visual", "direct_visual_begin", "wrong_store"])
def test_bypasses_rejected(tmp_path, mode):
    combined, _, contract = make(tmp_path)
    def invalid():
        if mode == "physical_during_visual":
            return combined.meter.call("reset", lambda: pytest.fail("callback ran"))
        return combined.meter.begin("encoder")
    with combined.scope():
        with pytest.raises(ValueError):
            combined._visual(invalid, combined.meter, object() if mode == "wrong_store" else combined.store)
    assert finish(combined, contract)[1]["status"] == "failed_or_blocked"


def test_visual_inside_physical_callback_rejected(tmp_path):
    combined, model, contract = make(tmp_path)
    with combined.scope():
        with pytest.raises(ValueError, match="scope mismatch"):
            combined.meter.call("reset", lambda: combined._visual(
                lambda: operation(model, combined), combined.meter, combined.store))
    ph, total = finish(combined, contract)
    assert ph["charged"]["reset"] == 1
    assert total["charged"]["source_encode_calls"] == 0
    assert total["status"] == "failed_or_blocked"


def test_full_host_still_closed():
    with pytest.raises(ValueError, match="fail-closed"):
        c.run_full_host()
