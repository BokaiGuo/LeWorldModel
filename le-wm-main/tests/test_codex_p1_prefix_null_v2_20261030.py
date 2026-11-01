import copy
from contextlib import nullcontext
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_prefix_null_runner_20261030_v2 as runner
from experiments.spectral_reachability import codex_p1_prefix_null_saved_consumer_20261030_v2 as consumer


@pytest.fixture(scope="module")
def source():
    return runner.material()


def test_real_freeze_roster_and_contract(source):
    p, _ = source
    runner.check_roster(p)
    assert len(p["plans"]) == 16
    assert p["contract"] == runner.v1.base.contract.contract()
    assert p["P1_admission"] is p["independent_confirmation"] is p["equal_budget_certified"] is False


@pytest.mark.parametrize("kind", ["zero", "suffix", "task", "order", "budget", "group", "admission"])
def test_roster_tampering_rejected(source, kind):
    p = copy.deepcopy(source[0])
    if kind == "zero":
        p["plans"][1]["raw15_float32_hex"] = "0000803f" + p["plans"][1]["raw15_float32_hex"][8:]
    elif kind == "suffix":
        p["plans"][1]["raw15_float32_hex"] = p["plans"][1]["raw15_float32_hex"][:-8] + "0000803f"
    elif kind == "task":
        p["plans"][1]["task_id"] = "0" * 64
    elif kind == "order":
        p["plans"][0], p["plans"][2] = p["plans"][2], p["plans"][0]
    elif kind == "budget":
        p["contract"]["caps"]["physics_step"] += 1
    elif kind == "group":
        p["plans"][0]["duplicate_action_task_group"] = 999
    else:
        p["P1_admission"] = True
    with pytest.raises(ValueError):
        runner.check_roster(p)


def test_prepare_exclusive_and_drift(tmp_path, monkeypatch, source):
    root = tmp_path
    (root / "results").mkdir()
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "material", lambda: source)
    path = root / "results" / (runner.NAME + "_prepare_fixture.json")
    receipt = runner.prepare(path)
    assert runner.verify(path, receipt["sha256"])[0] == source[0]
    with pytest.raises(FileExistsError):
        runner.prepare(path)
    bad = copy.deepcopy(source[0])
    bad["plans"][0]["role"] = "wrong"
    monkeypatch.setattr(runner, "material", lambda: (bad, source[1]))
    with pytest.raises(ValueError, match="drift"):
        runner.verify(path, receipt["sha256"])


def test_real_adapter_partial_failure_reaches_v2_consumer(tmp_path, monkeypatch, source):
    """Real execute/Meter/journal/consumer wiring; injected adapter fails before physics."""
    root = tmp_path
    (root / "results").mkdir()
    prepared = copy.deepcopy(source[0])
    prepared["bindings"] = {}
    data = source[1]
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "verify", lambda *args: (prepared, data))
    monkeypatch.setattr(runner.v1.base.native, "constructor_contract", lambda: None)
    monkeypatch.setattr(runner.v1.base.native, "validate_env", lambda *_: None)
    monkeypatch.setattr(runner.v1.base.native, "native_hooks", lambda *_: nullcontext())
    def fail(env, root_data, source_plan, plan, row, meter, store):
        meter.call("public_step", lambda: (_ for _ in ()).throw(ValueError("fixture dispatch failure")),
                   phase="plan", step=1)
    monkeypatch.setattr(runner.v1.base.episode, "episode", fail)
    def make(*args, **kwargs):
        return SimpleNamespace(unwrapped=SimpleNamespace(render_action=False, render_size=224,
                                                          render_mode="rgb_array"), close=lambda: None)
    monkeypatch.setitem(sys.modules, "gymnasium", SimpleNamespace(make=make))
    monkeypatch.setitem(sys.modules, "stable_worldmodel", SimpleNamespace())
    output = root / "results" / (runner.NAME + "_run_fixture")
    report = runner.execute("/fixture-prepare", "a" * 64, output)
    assert report["status"] == "partial_or_failed"
    assert report["counts"]["public_step"] == {"attempted": 1, "completed": 0}
    assert report["rows"][0]["status"] == "engineering_failed"
    assert all(r["status"] == "not_started" for r in report["rows"][1:])
    out = consumer.consume(output, runner.v1.base.sha(output / "report.json"))
    assert out["status"] == "blocked_incomplete" and len(out["rows"]) == 16
    assert out["counts"]["public_step"] == {"attempted": 1, "completed": 0}
    with pytest.raises(ValueError):
        consumer.consume(output, "0" * 64)


def test_sequence_early_stop_and_out_of_bounds():
    sequence = runner.v1.base.flow.sequence
    calls = []
    def dispatch(step, action):
        calls.append(step)
        return {"terminated": step == 3, "truncated": False}
    result = sequence(np.zeros((15, 2), np.float32), dispatch, lambda step, _: step, lambda *_: True)
    assert calls == [1, 2, 3] and result["checkpoints"][5] is None
    calls.clear()
    def capture(step, _):
        if step == 4:
            raise ValueError("out of canvas")
        return step
    with pytest.raises(ValueError, match="canvas"):
        sequence(np.zeros((15, 2), np.float32),
                 lambda step, a: (calls.append(step) or {"terminated": False, "truncated": False}),
                 capture, lambda *_: True)
    assert calls == [1, 2, 3, 4]


@pytest.mark.parametrize("mode", ["complete", "early", "out_of_bounds"])
def test_all_16_actual_episode_to_strict_saved_consumer(tmp_path, monkeypatch, source, mode):
    """Full wiring with deterministic fake native I/O, no real physics or model."""
    root = tmp_path
    (root / "results").mkdir()
    base = runner.v1.base
    native = base.native
    prepared = copy.deepcopy(source[0])
    prepared["bindings"] = {}
    state = [0.0] * 7
    goal = [100.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0]
    state_hash = "b" * 64
    rgb = np.zeros((224, 224, 3), np.uint8)
    pixel_sha = native.moving.image_sha(rgb)
    warm = native.moving.warmup_commands()
    snapshot = {"state": state, "bodies": {
        "agent": {"position": state[:2], "velocity": state[5:7]},
        "block": {"position": state[2:4], "velocity": [0.0, 0.0], "angular_velocity": 0.0}}}
    packets = {c: {"raw_action_history": warm, "rgb_history": np.array([rgb, rgb, rgb]),
                   "goal_rgb": rgb} for c in ("off", "on")}
    roots = [{"packets": packets, "warm_trace": [], "slot": {}, "goal": np.array(goal),
              "root_sha256": state_hash} for _ in range(2)]
    data = {"roots": roots, "source_plan": {}}
    for plan in prepared["plans"]:
        plan["physical_root_sha256"] = state_hash
    for task in prepared["task_rows"]:
        task.update(native_root_sha256=state_hash, native_root_state=state, native_goal=goal,
                    goal_pixel_sha256=pixel_sha)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "verify", lambda *args: (prepared, data))
    monkeypatch.setattr(native, "constructor_contract", lambda: None)
    monkeypatch.setattr(native, "validate_env", lambda *_: None)
    monkeypatch.setattr(native, "native_hooks", lambda *_: nullcontext())
    monkeypatch.setattr(native.source, "root_hash", lambda *_: state_hash)
    monkeypatch.setattr(native.moving, "guard", lambda *_: None)
    monkeypatch.setattr(native, "reference_for", lambda *_: None)
    monkeypatch.setattr(native, "invariant", lambda *_: {"goal_pose": [0], "snapshot": snapshot})
    metric = native.base.previous.metric(goal, state)
    monkeypatch.setattr(native.paired, "check_return", lambda *_: metric["native_success"])
    def place(env, source_plan, slot, meter, store):
        def inner():
            for _ in range(2):
                meter.call("set_state", lambda: None)
                meter.call("physics_step", lambda: None)
            meter.call("goal_setter", lambda: None)
            meter.call("goal_render", lambda: None)
        for _ in range(2):
            meter.call("reset", inner)
        return {"state": np.array(state)}, np.array(goal)
    def bind(body, actual_goal, goal_rgb, meter, store):
        meter.call("goal_bind", lambda: meter.call("goal_setter", lambda: None))
    monkeypatch.setattr(native, "place", place)
    monkeypatch.setattr(native, "bind_goal", bind)
    def dispatch(env, command, phase, step, row, meter, store, actual_goal):
        def work():
            for _ in range(10):
                meter.call("physics_step", lambda: None)
        meter.call("public_step", work, phase=phase, step=step, raw_bits=command.tobytes().hex())
        truncated = mode == "early" and row["index"] == 0 and phase == "plan" and step == 6
        store.emit("returned_transition", **meter.where, phase=phase, step=step,
                   requested_bits=command.tobytes().hex(), actual_bits=command.tobytes().hex(),
                   actual_dtype=command.dtype.str, state=state, snapshot=snapshot,
                   reward=-metric["native_distance"], terminated=False, truncated=truncated, goal_state=goal)
        return ({"state": np.array(state)}, {"goal": rgb}, False, truncated,
                -metric["native_distance"], command.copy())
    monkeypatch.setattr(native, "dispatch", dispatch)
    def observe(env, obs, phase, step, row, meter, store, trace):
        chosen = None
        for overlay in range(3):
            meter.call("explicit_render", lambda: None)
            name = f'b{row["index"]}_{phase}_{step:02}_{overlay}.png'
            digest = store.frame(name, rgb)
            rec = {"phase": phase, "step": step, "condition": "on" if overlay == 1 else "off",
                   "overlay_index": overlay, "png": name, "file_sha256": digest,
                   "pixel_sha256": pixel_sha, "state": state, "snapshot": snapshot}
            invariant = {"goal_pose": [0], "snapshot": snapshot}
            store.emit("fresh_observation", **meter.where, receipt=rec, pre=invariant, post=invariant)
            row["PNG"] += 1
            if mode == "out_of_bounds" and row["index"] == 0 and phase == "plan" and step == 4:
                raise ValueError("fixture canvas guard")
            if phase == "warmup" or step <= 5:
                row["prefix51"]["frames"].append({"receipt": rec, "pre": invariant, "post": invariant})
            if overlay == (1 if row["condition"] == "on" else 0):
                chosen = rec
        row["trajectory"].append(copy.deepcopy(chosen))
        if phase == "plan" and step in (0, 5, 10):
            row["checkpoints"].append({k: copy.deepcopy(chosen[k]) for k in
                                      ("step", "png", "file_sha256", "pixel_sha256", "state", "snapshot")})
            row["missing_checkpoints"].remove(step)
        return rgb
    monkeypatch.setattr(native, "observe", observe)
    class FastStore(runner.Store):
        def __init__(self, path):
            self.path = path
            path.mkdir()
            self.previous, self.index = None, 0
        def check(self, *args, **kwargs):
            return None
        def write(self, name, value, append=False, final=False):
            path = self.path / name
            with path.open("ab" if append else "xb") as f:
                f.write(value)
            return None if append else native.sha(path)
    monkeypatch.setattr(runner, "Store", FastStore)
    def make(*args, **kwargs):
        body = SimpleNamespace(render_action=False, render_size=224, render_mode="rgb_array",
                               eval_state=lambda *_: (False,))
        return SimpleNamespace(unwrapped=body, close=lambda: None)
    monkeypatch.setitem(sys.modules, "gymnasium", SimpleNamespace(make=make))
    monkeypatch.setitem(sys.modules, "stable_worldmodel", SimpleNamespace())
    output = root / "results" / (runner.NAME + "_run_complete_fixture")
    report = runner.execute("/fixture-prepare", "a" * 64, output)
    if mode == "out_of_bounds":
        assert report["status"] == "partial_or_failed"
        assert report["rows"][0]["status"] == "engineering_failed"
        assert all(row["status"] == "not_started" for row in report["rows"][1:])
        assert report["rows"][0]["missing_checkpoints"] == [5, 10, 15]
        out = consumer.consume(output, native.sha(output / "report.json"))
        assert out["status"] == "blocked_incomplete" and len(out["rows"]) == 16
        assert out["rows"][0]["missing_checkpoints"] == [5, 10, 15]
        return
    assert report["status"] == "physical_recorded_not_independently_consumed", report["errors"]
    steps = 391 if mode == "early" else 400
    assert report["counts"]["public_step"] == {"attempted": steps, "completed": steps}
    assert report["counts"]["physics_step"] == {"attempted": 64 + 10 * steps, "completed": 64 + 10 * steps}
    out = consumer.consume(output, native.sha(output / "report.json"))
    assert out["status"] == "saved_checks_passed_exploratory_only"
    assert len(out["rows"]) == len(out["checkpoints"]) == 16
    assert [x["step"] for x in out["checkpoints"][0]["observed"]] == (
        [0, 5] if mode == "early" else [0, 5, 10, 15])
    assert out["checkpoints"][0]["missing"] == ([10, 15] if mode == "early" else [])
    assert all([x["step"] for x in row["observed"]] == [0, 5, 10, 15] for row in out["checkpoints"][1:])
    assert out["P1_admission"] is out["independent_confirmation"] is False
    if mode == "complete":
        frame = output / report["rows"][0]["checkpoints"][0]["png"]
        frame.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="frame file SHA"):
            consumer.consume(output, native.sha(output / "report.json"))
