import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import codex_p1_full_cem_host_20261101_exclusive as m


def test_fixed_node_costs_are_not_equal_cost_claim():
    assert [m.node_cost(b)["predict_calls"] for b in (0, 5, 10, 15)] == [195, 130, 65, 65]
    assert sum(m.node_cost(b)["action_encode_calls"] for b in (0, 5, 10, 15)) == 455
    with pytest.raises(ValueError):
        m.node_cost(20)


def test_real_task_builds_fixed_budget_contract():
    from experiments.spectral_reachability import codex_p1_full_cem_source_20261101_exclusive as source
    prepared = json.loads((source.ROOT / "results/p1_first_feedback_carry_20261101_main_root0_PREP_MAIN02.json").read_bytes())
    task = prepared["source"]["tasks"][0]
    contract = m.node_contract(task, "a" * 64, "b" * 64, 5)
    assert contract["levels"][0]["caps"] == m.node_cost(5)
    assert contract["identity"]["optimizer"] == m.common.spectral.file_sha(m.cem.__file__)
    assert contract["phase_policy_sha256"] == m.spec.digest(
        {"kind": "full_cem", "seed": m.SEED, "boundary": 5, "horizon": 2,
         "iterations": 4, "samples": 16, "elites": 4})


@pytest.mark.parametrize("root", [0, 1])
def test_actual_saved_source_rebound_goal_and_contract(root):
    from experiments.spectral_reachability import codex_p1_full_cem_source_20261101_exclusive as source
    value = source.inspect(root)
    base = json.loads((source.ROOT / "results" /
        f"p1_first_feedback_carry_20261101_main_root{root}_PREP_MAIN02.json").read_bytes())
    task = base["source"]["tasks"][root]
    target = json.loads((source.ROOT / "results/e3_cp10_target_cuda_20261031_RUN01.json").read_bytes())
    task["goal"] = target["rows"][root]["latent"]
    m.fixed_identity_check(task, value["identity"])
    task["goal"][0] += 1.
    with pytest.raises(ValueError, match="goal"):
        m.fixed_identity_check(task, value["identity"])


def test_live_leaf_checks_guard_before_call():
    seen = []
    runtime = {"check_limits": lambda: seen.append("check")}
    leaf = m.LiveLeaf(runtime, lambda: "rgb")
    leaf.node = type("Node", (), {
        "predict": lambda self, a, b, phase: seen.append("predict") or a,
        "action_encode": lambda self, a, phase: seen.append("action") or a,
    })()
    assert leaf.encode({}, phase="setup") == "rgb"
    x = torch.ones(1)
    leaf.predict(x, x, phase="verification")
    leaf.action_encode(x, phase="verification")
    assert seen == ["check", "check", "predict", "check", "action"]
    runtime["check_limits"] = lambda: (_ for _ in ()).throw(ValueError("limit"))
    with pytest.raises(ValueError, match="limit"):
        leaf.predict(x, x, phase="verification")
    assert seen[-1] == "action"


def fixture_runtime(tmp_path, monkeypatch, *, fail_step=None, stop_step=None, mutate_planner=False):
    """CPU fake exercises orchestration only, not real model or simulator."""
    monkeypatch.setattr(m.cem, "ITERATIONS", 1)
    monkeypatch.setattr(m.cem, "SAMPLES", 2)
    monkeypatch.setattr(m.cem, "ELITES", 1)
    goal = torch.ones(1, 2)
    normalizer = torch.tensor([[0., 0.], [1., 1.]])
    matrix = torch.eye(2, dtype=torch.float64)
    identity = {"goal_sha256": m.cem.tensor_sha(goal),
                "normalizer_sha256": m.cem.tensor_sha(normalizer),
                "metric_sha256": m.cem.tensor_sha(matrix),
                "root_sha256": "a" * 64, "task_sha256": "b" * 64,
                "model_sha256": "c" * 64}
    rebound = {"root_sha256": identity["root_sha256"]}
    monkeypatch.setattr(m.source, "inspect", lambda root: {"identity": identity, "rebound_goal_binding": rebound})
    monkeypatch.setattr(m, "node_contract", lambda task, model, hardware, boundary:
                        {"boundary": boundary, "caps": m.node_cost(boundary)})

    class Driver:
        def __init__(self, path, contract, digest, arm, level, run_id):
            self.contract, self.counts, self.closed = contract, dict.fromkeys(m.spec.AXES, 0), False

        def finish(self):
            self.closed = True
            return {"journal": "fixture", "sha256": "f" * 64, "counts": self.counts.copy()}

    class Leaf:
        def __init__(self, model, driver, contract, sha):
            self.driver = driver

        def action_encode(self, actions, *, phase):
            self.driver.counts["action_encode_calls"] += 1
            return actions

        def predict(self, states, actions, *, phase):
            self.driver.counts["predict_calls"] += 1
            self.driver.counts["predict_samples"] += 1
            if mutate_planner:
                body.state += 1
            return states.clone()

    monkeypatch.setattr(m.durable, "Driver", Driver)
    monkeypatch.setattr(m.leafmod, "LeafAdapter", Leaf)
    monkeypatch.setattr(m.consumer, "consume", lambda journal, sha, contract_sha, arm, level, run:
                        {"status": "closed_unverified", "charged_upper_bounds": current_driver_counts()})
    drivers = []
    old_init = Driver.__init__

    def remember(self, *args):
        old_init(self, *args)
        drivers.append(self)
    monkeypatch.setattr(Driver, "__init__", remember)
    def current_driver_counts():
        return drivers[-1].counts.copy()

    class Store:
        def __init__(self):
            self.path, self.events, self.finalizing, self.previous = tmp_path, [], False, None
            (tmp_path / "journal.jsonl").write_bytes(b"fixture")

        def put(self, name, value):
            self.events.append(("put", name, deepcopy(value)))

        def emit(self, kind, **value):
            self.events.append((kind, deepcopy(value)))

        def check(self, size=4096):
            assert size > 0

        def namespace(self, name):
            path = self.path / name
            path.mkdir()
            return path

    class Meter:
        def __init__(self, store, check):
            self.counts, self.pending, self.secondary = {}, [], []
        def call(self, name, fn):
            return fn()

    class Body:
        _live_constructor_complete, with_target, render_action = True, False, False
        render_size, relative, action_scale, control_hz, dt = 224, True, 100, 10, .01
        state = 0
    body = Body()
    class Env:
        unwrapped = body
        def close(self):
            pass

    def factory(meter):
        body._live_meter = meter
        return Env()

    def setup(env, ref, plan, row, meter, store):
        row.update(initial_exact=True, last_observation={"pixel_sha256": "d" * 64})
        return True

    def encode(handler, runtime, task, layer, row, obs, body, meter, store, boundary):
        return np.zeros(2, np.float32), {"pixel_sha256": "d" * 64, "encoding_sha256": "e" * 64}

    def packet(env, commands, *, start, row, ref, meter, store):
        for i, command in enumerate(commands):
            store.emit("authorized_dispatch", step=start+i+1)
            if fail_step == start+i+1:
                raise RuntimeError("fixture physical failure")
            row["control_returned"] += 1
            row["executed_raw"].append(command)
            store.emit("control_return", step=start+i+1)
            if stop_step == start+i+1:
                row["status"] = "native_stopped_control"
                return None
        return {"pixel_sha256": "d" * 64}

    moving = SimpleNamespace(setup=setup, encode_boundary=encode, packet=packet,
        normalized=lambda base, raw: np.zeros((2, 10), np.float32).tolist(),
        roll=lambda context, z, raw, base: context)
    rgb = SimpleNamespace(moving=moving,
        node_roster=lambda row: [{"boundary": b, "status": "not_started"} for b in m.cem.BOUNDARIES])
    row = {"first_error": None, "control_returned": 0, "executed_raw": [],
           "completed_packets": 0, "status": "not_started"}
    p3 = SimpleNamespace(roster=lambda: [row], Meter=Meter,
                         local_task=lambda base, context, ident: {**base, "root_context": context})
    def retain(row, exc, stage):
        row["first_error"] = str(exc)
    old = SimpleNamespace(kernel=SimpleNamespace(retain=retain), state_seal=lambda body: body.state)
    runner = SimpleNamespace(rgb=rgb, p3=p3, old=old, protocol=lambda: {},
                             sha=lambda path: "9" * 64)
    task = {"goal": goal[0].tolist(), "action_mean": [0., 0.], "action_std": [1., 1.],
            "projection": matrix.tolist(),
            "binding": {"root_sha256": identity["root_sha256"], "task_sha256": identity["task_sha256"]},
            "root_context": {"latent_history": np.zeros((3, 2)).tolist(),
                             "action_history": np.zeros((2, 10)).tolist()}}
    prepared = {"source": {"tasks": [task], "layers": [{}]}}
    data = {"references": [{"root_sha256": identity["root_sha256"],
                            "raw_history": np.zeros((10, 2), np.float32)}],
            "plan": {}, "visual_tasks": {0: {}}}
    runtime = {"model_sha256": identity["model_sha256"], "hardware_sha256": "f" * 64,
               "model": object(), "device": torch.device("cpu"), "check_limits": lambda: None}
    return (prepared, data, Store(), runtime, factory, runner, lambda p, d: rebound)


def test_full_orchestration_with_cpu_fixtures(tmp_path, monkeypatch):
    prepared, data, store, runtime, factory, runner, rebind = fixture_runtime(tmp_path, monkeypatch)
    result = m.execute_one(prepared, data, store, None, runtime, factory, runner, rebind, 0)
    assert result["status"] == "recorded_known_root_not_admitted"
    assert result["rows"][0]["control_returned"] == 20
    assert len([event for event in store.events if event[0] == "authorized_dispatch"]) == 20
    assert len([event for event in store.events if event[0] == "control_return"]) == 20
    assert len([event for event in store.events if event[0] == "boundary_sealed"]) == 4


@pytest.mark.parametrize("fail_step,mutate", [(2, False), (None, True)])
def test_physical_failure_or_planner_mutation_stops_without_retry(tmp_path, monkeypatch, fail_step, mutate):
    prepared, data, store, runtime, factory, runner, rebind = fixture_runtime(
        tmp_path, monkeypatch, fail_step=fail_step, mutate_planner=mutate)
    result = m.execute_one(prepared, data, store, None, runtime, factory, runner, rebind, 0)
    assert result["status"] == "partial_or_failed"
    events = [event[0] for event in store.events]
    assert events.count("authorized_dispatch") == (2 if fail_step else 0)
    assert events.count("control_return") == (1 if fail_step else 0)


def test_native_early_stop_is_recorded_without_padding(tmp_path, monkeypatch):
    prepared, data, store, runtime, factory, runner, rebind = fixture_runtime(
        tmp_path, monkeypatch, stop_step=7)
    result = m.execute_one(prepared, data, store, None, runtime, factory, runner, rebind, 0)
    assert result["status"] == "recorded_known_root_not_admitted"
    assert result["rows"][0]["status"] == "native_stopped_control"
    assert result["rows"][0]["control_returned"] == 7
    assert len([e for e in store.events if e[0] == "control_return"]) == 7
    assert len([e for e in store.events if e[0] == "boundary_sealed"]) == 2
