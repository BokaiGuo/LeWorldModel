from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p1_visual_live_host_wiring_20261101_leibniz as wiring


class BoundaryReached(Exception):
    pass


def test_structured_real_configure_and_episode_reach_guard(monkeypatch):
    host = wiring.structured.host
    before = (host.NAME, host.build, host.protocol, host.execute, host.rgb.roster, host.rgb.episode)
    guard = object()
    calls = []
    monkeypatch.setattr(wiring.visual.rgb.moving, "setup", lambda *a: True)

    def boundary(*args):
        calls.append(args)
        raise BoundaryReached()

    monkeypatch.setattr(wiring.visual.visual_guard, "encode_boundary", boundary)
    with wiring._structured_wiring(guard) as configured:
        guarded = wiring._captured(configured.rgb.episode, "old_episode")
        assert guarded.__code__ is before[-1].__code__
        row = {"last_observation": "obs"}
        with pytest.raises(BoundaryReached):
            configured.rgb.episode(
                SimpleNamespace(unwrapped="body"), {}, {"plan": {}, "visual_tasks": ["task"]},
                "layer", {"root_context": {}}, row, [{"boundary": 0}],
                "handler", {}, SimpleNamespace(), "store", None)
        assert calls[0][0] is guard
        assert calls[0][1] == "handler"
    assert (host.NAME, host.build, host.protocol, host.execute, host.rgb.roster, host.rgb.episode) == before


def test_structured_rejects_wrong_capture_and_restores(monkeypatch):
    host = wiring.structured.host
    before = host.rgb.episode
    monkeypatch.setattr(wiring, "_captured", lambda *a: before)
    with pytest.raises(ValueError, match="capture bypass"):
        with wiring._structured_wiring(object()):
            pytest.fail("bypassed capture admitted")
    assert host.rgb.episode is before


def test_structured_rejects_preconfigured_and_concurrent(monkeypatch):
    host = wiring.structured.host
    with monkeypatch.context() as p:
        p.setattr(host.rgb, "episode", lambda *a: None)
        with pytest.raises(ValueError, match="unexpected structured"):
            with wiring._structured_wiring(object()):
                pass
    with wiring._structured_wiring(object()):
        with pytest.raises(ValueError, match="concurrent"):
            with wiring._structured_wiring(object()):
                pass


def test_structured_detects_scope_mutation_and_restores():
    host = wiring.structured.host
    before = (host.NAME, host.build, host.protocol, host.execute, host.rgb.roster, host.rgb.episode)
    with pytest.raises(ValueError, match="episode changed"):
        with wiring._structured_wiring(object()):
            host.rgb.episode = lambda *a: None
    assert (host.NAME, host.build, host.protocol, host.execute, host.rgb.roster, host.rgb.episode) == before


def test_cem_real_execute_one_reaches_proxy_boundary(monkeypatch, tmp_path):
    cem = wiring.cem
    guard = object()
    calls = []
    identity = {"root_sha256": "r", "task_sha256": "t", "model_sha256": "m"}
    monkeypatch.setattr(cem.source, "inspect", lambda root: {
        "rebound_goal_binding": "binding", "identity": identity})
    monkeypatch.setattr(cem, "fixed_identity_check", lambda *a: None)
    monkeypatch.setattr(wiring.visual.rgb.moving, "setup", lambda env, ref, plan, row, meter, store:
                        row.update(initial_exact=True, last_observation="obs") or True)

    def boundary(*args):
        calls.append(args)
        raise BoundaryReached("boundary")

    monkeypatch.setattr(wiring.visual.visual_guard, "encode_boundary", boundary)

    class Meter:
        def __init__(self, *args):
            self.counts, self.pending, self.secondary = {}, [], []
        def call(self, name, fn):
            return fn()

    class Controller:
        def __init__(self, *, leaf, observe, **kw):
            self.leaf, self.observe = leaf, observe
            self.journal = SimpleNamespace(append=lambda payload: None)
        def step(self):
            self.observe(0, SimpleNamespace(encode=lambda *a, **kw: self.leaf.encode({}, phase="setup")))
        def close(self):
            pass

    def retain(row, exc, phase):
        row["first_error"] = str(exc)

    monkeypatch.setattr(cem.cem, "Controller", Controller)
    monkeypatch.setattr(cem, "SEED", 0)
    monkeypatch.setattr(cem.spec, "require", lambda condition, msg: None if condition else
                        (_ for _ in ()).throw(ValueError(msg)))
    store = SimpleNamespace(path=tmp_path, put=lambda *a, **kw: None, emit=lambda *a, **kw: None,
                            previous=None, check=lambda *a: None, finalizing=False)
    body = SimpleNamespace(with_target=False, render_action=False, render_size=224, relative=True,
                           action_scale=100, control_hz=10, dt=.01, _live_constructor_complete=True)
    def factory(meter):
        body._live_meter = meter
        return SimpleNamespace(unwrapped=body, close=lambda: None)
    runner = SimpleNamespace(
        rgb=SimpleNamespace(moving=wiring.visual.rgb.moving, node_roster=lambda row: [{"boundary": 0}]),
        p3=SimpleNamespace(roster=lambda: [{"control_returned": 0, "first_error": None}], Meter=Meter),
        old=SimpleNamespace(kernel=SimpleNamespace(retain=retain)),
        protocol=lambda: {}, sha=lambda path: "journal-hash")
    base = {"binding": {"root_sha256": "r", "task_sha256": "t"}, "root_context": {}}
    prepared = {"source": {"tasks": [base], "layers": ["layer"]}}
    data = {"references": [{"root_sha256": "r"}], "plan": {}, "visual_tasks": ["task"]}
    report = wiring._cem_execute_guarded(
        guard, prepared, data, store, "handler", {"model_sha256": "m", "check_limits": lambda: None},
        factory, runner, lambda p, d: "binding", 0)
    assert report["errors"][0]["type"] == "BoundaryReached", report["errors"]
    assert calls[0][0] is guard and calls[0][1] == "handler"
    assert runner.rgb.moving is wiring.visual.rgb.moving


def test_cem_rejects_unwrapped_and_both_public_entrypoints_closed(monkeypatch):
    runner = SimpleNamespace(rgb=SimpleNamespace(moving=object()))
    with pytest.raises(ValueError, match="unexpected CEM moving source"):
        wiring._cem_execute_guarded(object(), None, None, None, None, None, None, runner, None, 0)
    runner.rgb.moving = wiring.visual.rgb.moving
    monkeypatch.setattr(wiring.visual, "cem_runner", lambda guard, r: r)
    with pytest.raises(ValueError, match="CEM proxy mismatch"):
        wiring._cem_execute_guarded(object(), None, None, None, None, None, None, runner, None, 0)
    for run in (wiring.run_structured, wiring.run_cem):
        with pytest.raises(ValueError, match="fail-closed"):
            run()
