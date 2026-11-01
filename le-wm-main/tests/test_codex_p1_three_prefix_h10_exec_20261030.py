"""CPU-only fixtures: simulated native I/O, never a physical or GPU run."""
import copy
import hashlib
import json
import io
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from experiments.spectral_reachability import codex_p1_three_prefix_h10_runner_20261030 as run
from experiments.spectral_reachability import codex_p1_three_prefix_h10_saved_20261030 as saved


def test_existing_real_censor_evidence_extends_to_ten_steps():
    source = run.old.prep.DIRECTORY.parent / "p1_matched_suffix_scale_20261030_run_MAIN02/anchor_05"
    old = run.native.read(run.old.ROOT / "results/p1_matched_suffix_scale_20261030_prepare_MAIN01.json",
                          cap=32*1024**2)
    report = run.native.read(source / "report.json", cap=64*1024**2)
    events = saved.saved.events(source / "journal.jsonl", report["journal_chain_tail"])
    anchor, probe, row = old["anchors"][5], copy.deepcopy(old["probe_sets"][5][16]), report["rows"][16]
    raw = bytes.fromhex(probe["raw5_float32_hex"]) + bytes(40)
    probe["raw10_float32_hex"], probe["raw10_sha256"] = raw.hex(), hashlib.sha256(raw).hexdigest()
    with Image.open(source / row["last_observation"]["png"]) as image:
        pixels = np.asarray(image)
    evidence = run.proof.incident(events, anchor, probe, row, pixels)
    assert evidence["step"] == 5 and evidence["endpoint_accepted"] is False
    # Relabel the saved incident as a synthetic step-8 case, retaining its
    # original ordering/state/action evidence. This is not a physical replay.
    events8, row8 = copy.deepcopy(events), copy.deepcopy(row)
    old_action = raw[32:40]
    raw8 = raw[:40] + bytes(16) + old_action + bytes(16)
    probe8 = dict(probe, raw10_float32_hex=raw8.hex(),
                  raw10_sha256=hashlib.sha256(raw8).hexdigest())
    row8["last_observation"]["step"] = 8
    row8["own_returned"] = 13
    row8["responses"] = [None]*7
    for event in events8:
        if event.get("branch", event.get("receipt", {}).get("branch")) != 16:
            continue
        if event["kind"] in ("operation_dispatch_intent", "operation_returned"):
            receipt = event["receipt"]
            if receipt.get("phase") == "probe" and receipt.get("step") == 5 and receipt.get("resource") == "public_step":
                receipt["step"] = 8
        elif event["kind"] == "returned_transition" and event.get("phase") == "probe" and event.get("step") == 5:
            event["step"] = 8
        elif event["kind"] == "fresh_observation" and event["receipt"]["phase"] == "probe" and event["receipt"]["step"] == 5:
            event["receipt"]["step"] = 8
    assert run.proof.incident(events8, anchor, probe8, row8, pixels)["step"] == 8
    probe["raw10_sha256"] = "0"*64
    with pytest.raises(ValueError):
        run.proof.incident(events, anchor, probe, row, pixels)


@pytest.mark.parametrize("stop", [None, 7])
def test_real_episode_dispatches_ten_or_native_stop(monkeypatch, stop):
    base = run.base
    state, goal = np.zeros(7), np.ones(7)
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    root = dict(packets={"off": dict(rgb_history=np.array([rgb]*3),
                                      raw_action_history=base.moving.warmup_commands(), goal_rgb=rgb)},
                slot={}, warm=[None]*11, goal=goal, root_sha256="x")
    env = SimpleNamespace(unwrapped=SimpleNamespace(eval_state=lambda *_: (False,)))
    monkeypatch.setattr(base, "place", lambda *_: ({"state": state}, goal))
    monkeypatch.setattr(base.source, "root_hash", lambda *_: "x")
    monkeypatch.setattr(base, "bind_goal", lambda *_: None)
    monkeypatch.setattr(base, "invariant", lambda *_: {"snapshot": {}})
    monkeypatch.setattr(base.paired, "check_return", lambda *_: None)
    monkeypatch.setattr(run.prior, "compare_trace", lambda *_: None)
    dispatched = []
    def dispatch(env, command, phase, step, row, meter, store, goal):
        dispatched.append((phase, step, command.tobytes()))
        return {"state": state}, {"goal": rgb}, False, phase == "probe" and step == stop, 0., command
    monkeypatch.setattr(base, "dispatch", dispatch)
    monkeypatch.setattr(base, "capture", lambda *_: rgb)
    raw = np.arange(20, dtype="<f4").reshape(10, 2) / 100
    probe = dict(raw10_float32_hex=raw.tobytes().hex())
    row = dict(responses=[], status="not_started")
    store = SimpleNamespace(emit=lambda *a, **k: None)
    run.episode(env, root, {}, dict(condition="off", commands=np.zeros((10, 2)).tolist()),
                {}, probe, row, SimpleNamespace(), store)
    steps = stop or 10
    actions = [x for x in dispatched if x[0] == "probe"]
    assert len(actions) == len(row["responses"]) == steps
    assert [x[2] for x in actions] == [x.tobytes() for x in raw[:steps]]
    assert row["status"] == ("native_stopped" if stop else "probe_completed")


def test_shard_failure_keeps_42_and_operation_failure(tmp_path, monkeypatch):
    anchor = dict(index=0, root_slot=0, condition="off", arm="a", role="selected",
                  task_id="task", prefix5_float32_hex=bytes(40).hex())
    probes = [dict(index=i, name=str(i)) for i in range(42)]
    monkeypatch.setattr(run.base, "validate_env", lambda *_: None)
    monkeypatch.setattr(run.base, "native_hooks", lambda *_: nullcontext())
    def episode(env, root, source, plan, anchor, probe, row, meter, store):
        meter.call("public_step", lambda: (_ for _ in ()).throw(ValueError("failed")),
                   phase="probe", step=1)
    monkeypatch.setattr(run, "episode", episode)
    class Env:
        def __init__(self):
            self.unwrapped = SimpleNamespace(render_action=False, render_size=224, render_mode="rgb_array")
        def close(self):
            return None
    report = run.execute_shard(anchor, probes, {"roots": [None], "source_plan": None},
                               tmp_path / "shard", Env, {})
    assert report["status"] == "partial_or_failed"
    assert len(report["rows"]) == 42 and report["rows"][0]["status"] == "engineering_failed"
    assert all(r["status"] == "not_started" and not r["responses"] for r in report["rows"][1:])
    assert report["counts"]["public_step"] == dict(attempted=1, completed=0)
    assert any(r["resource"] == "public_step" and r["status"] == "failed" for r in report["receipts"])
    with pytest.raises(ValueError, match="complete shard"):
        saved.check_shard(tmp_path / "shard", anchor, probes, run.runner.physical.sha(tmp_path / "shard/report.json"))


def test_fixed_24_by_42_orchestration_and_incomplete_consumer(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(run, "ROOT", tmp_path)
    p = dict(original=dict(anchors=[dict(index=i, logical_pair=i//3) for i in range(24)],
                           probe_banks=[[dict(index=j) for j in range(42)] for _ in range(8)]), bindings={})
    monkeypatch.setattr(run, "verify_prepare", lambda *_: p)
    monkeypatch.setattr(run.runner.physical, "verify", lambda *_: (None, {}))
    monkeypatch.setattr(run.native, "constructor_contract", lambda: None)
    seen = []
    def shard(anchor, bank, data, path, factory, bindings):
        seen.append((anchor["index"], len(bank)))
        path.mkdir()
        status = "partial_or_failed" if anchor["index"] == 2 else "recorded_not_consumed"
        run._write_new(path / "report.json", dict(status=status))
        return dict(status=status)
    monkeypatch.setattr(run, "execute_shard", shard)
    output = results / (run.NAME + "_run_fixture")
    fixture = results / "fixture.json"
    fixture.write_bytes(b"fixture")
    digest = run.runner.physical.sha(fixture)
    result = run.run(str(fixture), digest, output)
    report = json.loads((output / "run_report.json").read_bytes())
    assert seen == [(0, 42), (1, 42), (2, 42)]
    assert report["fixed_branches"] == 1008 and result["status"] == "partial_or_failed"
    with pytest.raises(ValueError, match="complete run"):
        saved.consume(output, result["sha256"], str(fixture), digest)


def test_source_code_and_prepare_freeze():
    p = run.build()
    assert (p["fixed_anchors"], p["fixed_branches"]) == (24, 1008)
    assert p["caps"]["public_step"] == 42*25
    assert p["caps"]["physics_step"] == 2*84 + 10*1050
    assert p["caps"]["explicit_render"] == 42*27*3
    assert all(run.runner.physical.sha(path) == digest for path, digest in p["bindings"].items())
    assert not p["P1_admission"] and not p["independent_confirmation"]


def test_h10_full_shard_real_episode_and_saved_consumer(tmp_path, monkeypatch):
    """Real episode, ledger and saved checks with simulated native I/O."""
    base = run.base
    state = [100., 100., 200., 200., 0., 0., 0.]
    goal = [200., 200., 300., 300., 0., 0., 0.]
    image = np.zeros((224, 224, 3), np.uint8)
    image[0, 0] = [65, 105, 225]
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    png = buffer.getvalue()
    file_sha, pixel_sha = hashlib.sha256(png).hexdigest(), base.moving.image_sha(image)
    snapshot = dict(state=state, bodies={"agent": {"position": state[:2]},
                                          "block": {"position": state[2:4]}})
    invariant = dict(snapshot=snapshot, goal_state=goal)
    def receipt(phase, step, overlay):
        return dict(phase=phase, step=step, condition="on" if overlay == 1 else "off",
                    overlay_index=overlay, png="reference.png", file_sha256=file_sha,
                    pixel_sha256=pixel_sha, state=state, snapshot=snapshot)
    trace = [dict(receipt=receipt(phase, step, overlay), pre=invariant, post=invariant)
             for phase, n in (("warmup", 11), ("plan", 6))
             for step in range(n) for overlay in range(3)]
    checkpoint = {k: receipt("plan", 5, 0)[k] for k in
                  ("step", "png", "file_sha256", "pixel_sha256", "state", "snapshot")}
    anchor = dict(index=0, root_slot=0, condition="off", arm="a", role="selected", task_id="task",
                  prefix5_float32_hex=bytes(40).hex(), trace=trace, checkpoint=checkpoint, goal=goal)
    bank = run.design.bank(bytes(80))
    warm = base.moving.warmup_commands()
    root = dict(packets={"off": dict(rgb_history=np.array([image]*3), raw_action_history=warm,
                                     goal_rgb=image)}, slot={}, warm=[None]*11,
                goal=np.asarray(goal), root_sha256="fixture")
    monkeypatch.setattr(base, "validate_env", lambda *_: None)
    monkeypatch.setattr(base, "native_hooks", lambda *_: nullcontext())
    monkeypatch.setattr(base.source, "root_hash", lambda *_: "fixture")
    monkeypatch.setattr(base, "invariant", lambda *_: invariant)
    monkeypatch.setattr(base.paired, "check_return", lambda *_: None)
    def place(env, source_plan, slot, meter, store):
        def reset():
            for _ in range(2):
                meter.call("set_state", lambda: None)
                meter.call("physics_step", lambda: None)
            meter.call("goal_setter", lambda: None)
            meter.call("goal_render", lambda: None)
        for _ in range(2):
            meter.call("reset", reset)
        return {"state": np.asarray(state)}, np.asarray(goal)
    def bind(body, g, rgb, meter, store):
        meter.call("goal_bind", lambda: meter.call("goal_setter", lambda: None))
        store.emit("goal_attached", **meter.where, goal=goal)
    def dispatch(env, command, phase, step, row, meter, store, goal_value):
        def step_fn():
            for _ in range(10):
                meter.call("physics_step", lambda: None)
        meter.call("public_step", step_fn, phase=phase, step=step, raw_bits=command.tobytes().hex())
        row["warmup_returned" if phase == "warmup" else "own_returned"] += 1
        stop = row["index"] == 0 and phase == "probe" and step == 7
        store.emit("returned_transition", **meter.where, phase=phase, step=step,
                   requested_bits=command.tobytes().hex(), actual_bits=command.tobytes().hex(),
                   actual_dtype="<f4", state=state, snapshot=snapshot, goal_state=goal,
                   reward=0., terminated=False, truncated=stop)
        return {"state": np.asarray(state)}, {"goal": image}, False, stop, 0., command
    def capture(env, obs, phase, step, row, meter, store, reference=None):
        for overlay in range(3):
            meter.call("explicit_render", lambda: None)
            rec = receipt(phase, step, overlay)
            rec["png"] = f'b{row["index"]}_{phase}_{step:02}_{overlay}.png'
            assert store.write(rec["png"], png) == file_sha
            row["PNG"] += 1
            row["last_observation"] = copy.deepcopy(rec)
            store.emit("fresh_observation", **meter.where, receipt=rec, pre=invariant, post=invariant)
            if overlay == 0:
                chosen = rec
        row["trajectory"].append(copy.deepcopy(chosen))
        if phase == "plan" and step in (0, 5, 10):
            row["checkpoints"].append({k: chosen[k] for k in
                                      ("step", "png", "file_sha256", "pixel_sha256", "state", "snapshot")})
            row["missing_checkpoints"].remove(step)
        return image
    monkeypatch.setattr(base, "place", place)
    monkeypatch.setattr(base, "bind_goal", bind)
    monkeypatch.setattr(base, "dispatch", dispatch)
    monkeypatch.setattr(base, "capture", capture)
    class FastStore(run.Store):
        def __init__(self, path):
            self.path = path
            path.mkdir()
            self.previous, self.index = None, 0
        def check(self, *a, **k):
            return None
        def write(self, name, value, append=False, final=False):
            path = self.path / name
            with path.open("ab" if append else "xb") as f:
                f.write(value)
            return None if append else hashlib.sha256(value).hexdigest()
    monkeypatch.setattr(run, "Store", FastStore)
    class Env:
        def __init__(self):
            self.unwrapped = SimpleNamespace(render_action=False, render_size=224, render_mode="rgb_array",
                                              eval_state=lambda *_: (False,))
        def close(self):
            return None
    directory = tmp_path / "shard"
    report = run.execute_shard(anchor, bank, dict(roots=[root], source_plan={}), directory, Env, {})
    assert report["status"] == "recorded_not_consumed", report["errors"]
    checked = saved.check_shard(directory, anchor, bank, run.runner.physical.sha(directory / "report.json"))
    assert checked["statuses"][0] == "native_stopped" and checked["endpoints"][0] is None
    assert checked["statuses"][1:] == ["probe_completed"]*41
    assert len(report["rows"][0]["responses"]) == 7
    assert len(report["rows"][1]["responses"]) == 10
    frame = directory / report["rows"][1]["trajectory"][-1]["png"]
    frame.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="PNG bytes"):
        saved.check_shard(directory, anchor, bank, run.runner.physical.sha(directory / "report.json"))
