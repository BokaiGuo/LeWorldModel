"""CPU-only injected producer; fixture results are not native execution."""
from copy import deepcopy
import io

import numpy as np
import pytest
from PIL import Image

import agent_p3_fullcommit_live_20261028_v1_tests as base_fixture
import test_p3_feedback_reverse_control_20261030_v1 as feedback_fixture
import test_p3_six_native_factory_20261030_v1 as factory_fixture
import test_p3_matched_order_feedback_joint_v2_codex_20261029 as joint_fixture
from experiments.spectral_reachability import p3_six_bound_producer_20261030_v1 as m
from experiments.spectral_reachability import p3_six_owned_constructor_20261030_v1 as owner
from experiments.spectral_reachability import p3_matched_order_feedback_encoder_v2_codex_20261029 as encoder


def setup(tmp_path, monkeypatch, *, drift=False):
    limits = {k: 1024 * 1024 for k in m.transport.base.LIMITS}
    limits["physical_png"] = 192 * m.native_factory.PNG_CAP
    store = m.BoundStore(tmp_path / "owned", limits=limits, cap=sum(limits.values()) + 1024 * 1024,
                         reserve=1024 * 1024, floor=1024 * 1024, wall=60)
    factory, created, _, prepared, data = factory_fixture.make(fail="setup", store=store)
    contract = dict(roots=[dict(task="fixture0"), dict(task="fixture1")], expected_model_hash="a" * 64,
                    precision={"fixture": True}, scientific_pass=False)
    calls = []
    def source():
        calls.append(True)
        value = deepcopy(contract)
        if drift and len(calls) > 1:
            value["expected_model_hash"] = "b" * 64
        return value, prepared, data
    monkeypatch.setattr(m.strict, "source_material", source)
    return store, factory, created, contract, m.strict.core.digest(contract), m.own_sha()


def events(store):
    path = store.output / "events.jsonl"
    return m.transport.inspect(path, m.terminal.file_sha(path), store.cap)["events"]


def test_source_events_wrap_real_lifecycle_and_strict_binding(tmp_path, monkeypatch):
    store, factory, created, contract, sha, own = setup(tmp_path, monkeypatch)
    result = m.execute(store, factory, None, source_sha256=sha, producer_sha256=own, permit_ram=True)
    assert result["status"] == "lifecycle_returned_not_certified"
    assert all(r["status"] == "engineering_failed" for r in result["rows"])
    assert len(created) == 6 and all(x.calls == 1 for x in created)
    es = events(store)
    assert [es[i]["resource"] for i in range(3)] == [
        "six_strict_source_begin", "six_denominator", "six_producer_binding"]
    assert es[-1]["resource"] == "six_strict_source_end"
    report = m.strict.native_saved._pinned(store.output / m.terminal.REPORT,
                                           result["terminal"]["report_sha256"])
    m.strict.check_binding(es, contract, sha)
    m.audit_binding(es, report, contract, sha, own)
    receipt = m.strict.native_saved.consume(store.output, result["terminal"]["report_sha256"],
                                            result["terminal"]["completion_sha256"])
    assert [row["status"] for row in receipt["rows"]] == ["engineering_failed"] * 6
    with pytest.raises(ValueError, match="six complete"):
        m.strict.consume(store.output, result["terminal"]["report_sha256"],
                         result["terminal"]["completion_sha256"], sha)
    changed = deepcopy(es)
    changed[2]["detail"]["profile"]["cap"] += 1
    with pytest.raises(ValueError, match="profile binding"):
        m.audit_binding(changed, report, contract, sha, own)
    changed = deepcopy(es)
    changed[-1]["detail"]["precision"] = {}
    with pytest.raises(ValueError, match="binding content"):
        m.audit_binding(changed, report, contract, sha, own)
    with pytest.raises(ValueError, match="producer source"):
        m.audit_binding(es, report, contract, sha, "0" * 64)


def test_source_drift_preserves_failed_roster_and_no_end(tmp_path, monkeypatch):
    store, factory, created, contract, sha, own = setup(tmp_path, monkeypatch, drift=True)
    with pytest.raises(ValueError, match="final source drift") as exc:
        m.execute(store, factory, None, source_sha256=sha, producer_sha256=own, permit_ram=True)
    assert any("partial terminal hashes" in x for x in exc.value.__notes__)
    assert store.sealed and len(created) == 6 and all(x.calls == 1 for x in created)
    es = events(store)
    assert es[-1]["resource"] == "six_producer_failure"
    assert not any(e["resource"] == "six_strict_source_end" for e in es)
    assert len([e for e in es if e["resource"] == "six_slot_return"]) == 6
    with pytest.raises(ValueError, match="source binding boundaries"):
        m.strict.check_binding(es, contract, sha)
    assert (store.output / m.terminal.REPORT).is_file()


def test_wrong_preflight_pin_never_starts_lifecycle(tmp_path, monkeypatch):
    store, factory, created, contract, sha, own = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="producer source pin"):
        m.execute(store, factory, None, source_sha256=sha, producer_sha256="0" * 64, permit_ram=True)
    assert not created
    assert [e["resource"] for e in events(store)] == ["six_producer_failure"]
    assert store.sealed and (store.output / m.terminal.REPORT).is_file()


def test_end_event_write_failure_retains_failure_and_no_end(tmp_path, monkeypatch):
    store, factory, created, contract, sha, own = setup(tmp_path, monkeypatch)
    original = store.event
    def fail_end(origin, kind, resource, token, detail):
        if resource == "six_strict_source_end":
            raise OSError("end event rejected")
        return original(origin, kind, resource, token, detail)
    monkeypatch.setattr(store, "event", fail_end)
    with pytest.raises(OSError, match="end event rejected"):
        m.execute(store, factory, None, source_sha256=sha, producer_sha256=own, permit_ram=True)
    assert len(created) == 6 and events(store)[-1]["resource"] == "six_producer_failure"
    assert store.sealed and (store.output / m.terminal.COMPLETION).exists()


def test_exact_source_array_comparison():
    a = {"x": [np.array([1, 2], dtype=np.float32)]}
    assert m._equal(a, deepcopy(a))
    assert not m._equal(a, {"x": [np.array([1, 2], dtype=np.float64)]})
    assert not m._equal(a, {"x": [np.array([1, 3], dtype=np.float32)]})


def test_complete_synthetic_saved_chain_strict_consumer(tmp_path, monkeypatch):
    """Synthetic traces, no GPU/physics; buffered writes deliberately do not test fsync."""
    def buffered(store, name, data, category, append=False):
        store.check(category, len(data))
        path = store.output / name
        with path.open("ab" if append else "xb") as stream:
            assert stream.write(data) == len(data)
        store.used[category] += len(data)
    monkeypatch.setattr(m.BoundStore, "write", buffered)
    limits = {key: 256 * 1024**2 for key in m.transport.base.LIMITS}
    store = m.BoundStore(tmp_path / "complete", limits=limits, cap=sum(limits.values()) + 1024**2,
                         reserve=1024**2, floor=1024**2, wall=600)
    warm = base_fixture.m.old.kernel.moving.warmup_commands()
    root, _ = feedback_fixture.setup()
    roots = [deepcopy(root), deepcopy(root)]
    sources = []
    for root in roots:
        trace = []
        for step in range(11):
            image = np.full((224, 224, 3), step, np.uint8)
            trace.append(dict(receipt=dict(step=step, overlay_index=0, snapshot={"fixture": step},
                                           state=[float(step)],
                                           pixel_sha256=m.transport.base.fsha(image.tobytes()))))
        sources.append(dict(root_sha256=root["binding"]["root_sha256"], warm_trace=trace,
                            packets={"off": {"raw_action_history": warm.copy()}}))
    prepared = dict(tasks=roots)
    data = dict(roots=sources, source_plan={})
    contract = dict(roots=[dict(task="fixture0"), dict(task="fixture1")],
                    expected_model_hash="a" * 64, precision={"fixture": True}, scientific_pass=False)
    sha, own = m.strict.core.digest(contract), m.own_sha()
    monkeypatch.setattr(m.strict, "source_material", lambda: (deepcopy(contract), prepared, data))

    class Raw:
        def close(self):
            pass
    class Reset:
        @staticmethod
        def TracedEnv(env, store, episode):
            return env
    def meter_call(meter, name, region):
        previous = meter.region
        meter.region = region
        try:
            meter.call(name, lambda: None)
        finally:
            meter.region = previous
    def observe(capture, row, phase, step):
        image = np.full((224, 224, 3), step, np.uint8)
        name = f'{row["episode_id"]}_{phase}_{step}.png'
        capture.frame(name, image)
        frame = capture.frames[name]
        receipt = dict(phase=phase, step=step, path=str(store.output / name),
                       file_sha256=frame["sha256"], pixel_sha256=frame["pixel_sha256"],
                       snapshot={"fixture": step}, state=[float(step)])
        capture.emit("fresh_observation", episode_id=row["episode_id"], receipt=receipt)
        return image, frame
    class Native:
        @staticmethod
        def setup(env, source, plan, task, row, meter, capture):
            meter_call(meter, "reset", "setup")
            meter_call(meter, "reset", "setup")
            observe(capture, row, "warmup", 0)
            for step in range(1, 11):
                meter_call(meter, "public_step", "warmup")
                observe(capture, row, "warmup", step)
                capture.emit("warmup_return", episode_id=row["episode_id"], terminated=False, truncated=False)
            meter_call(meter, "goal_bind", "goal_bind")
            observe(capture, row, "plan", 0)
            capture.emit("initial_native_evaluation", episode_id=row["episode_id"], success=False)
            capture.emit("matched_root_accepted", episode_id=row["episode_id"], root_sha256=source["root_sha256"])
            return "goal", "goal_rgb", warm[-5:].copy()

        class Session:
            def __init__(self, env, goal, rgb, meter, capture, row, model, torch, device):
                self.meter, self.capture, self.row = meter, capture, row
                self.episode_id = row["episode_id"]
            def packet(self, start, raw):
                for index, action in enumerate(raw):
                    step = start + index + 1
                    detail = dict(episode_id=self.episode_id, step=step, raw_bits=action.tobytes().hex())
                    self.capture.emit("authorized_dispatch", **detail)
                    meter_call(self.meter, "public_step", "prefix" if start == 0 else "suffix")
                    self.image, self.frame = observe(self.capture, self.row, "plan", step)
                    self.capture.emit("control_return", **detail, terminated=False, truncated=False)
                png = (store.output / self.frame["file"]).read_bytes()
                return dict(raw=raw.copy(), rgb_bytes=png, end=start + 5, stopped=False,
                            episode_id=self.episode_id)
            def encode(self, boundary, png):
                prep = encoder.original.pixels.preprocess_pixels(self.image)
                hidden = np.zeros((1, 257, 192), np.float32)
                out = np.full((1, 192), boundary / 100, np.float32)
                arrays = dict(pixels=self.image, preprocessed=prep, encoder_input=prep.reshape(1, 3, 224, 224),
                              encoder_hidden=hidden, projector_input=hidden[:, 0, :],
                              projector_output=out, latent=out.reshape(192))
                refs = {role: store.array("encoding", role, value) for role, value in arrays.items()}
                for resource in ("encoder", "projector"):
                    token = store.index
                    store.event("encoding", "intent", resource, token, dict(boundary=boundary))
                    store.event("encoding", "return", resource, token, dict(boundary=boundary))
                store.event("encoding", "record", "encoding_complete", store.index,
                            dict(boundary=boundary, frame=self.frame, arrays=refs))
                return dict(latent=out.reshape(192), input_sha256=self.frame["sha256"],
                            end=boundary, episode_id=self.episode_id)
    factory = m.native_factory.Factory(prepared, data, store, None, None, None,
                                      meter_module=owner.module(construct=lambda meter: Raw()),
                                      native_module=Native, reset_module=Reset)
    def backend(task, sink):
        return m.strict.old_saved.host.prior.RecordingBackend(
            joint_fixture.BufferedRAM(task, sink), task, sink, lambda: None)
    output = m.execute(store, factory, backend, source_sha256=sha, producer_sha256=own, permit_ram=True)
    assert all(row["status"] == "callback_feedback_complete_not_certified" for row in output["rows"])
    result = m.consume(store.output, output["terminal"]["report_sha256"],
                       output["terminal"]["completion_sha256"], sha, own)
    assert result["producer_binding_checked"] and len(result["source_math"]["slots"]) == 6
    assert not result["physical_certified"] and not result["scientific_pass"]
    with (store.output / "events.jsonl").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="inventory"):
        m.consume(store.output, output["terminal"]["report_sha256"],
                  output["terminal"]["completion_sha256"], sha, own)
