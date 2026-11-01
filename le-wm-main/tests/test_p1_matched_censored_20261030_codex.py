"""CPU-only event replay and substitutes; no simulator, CUDA, or production."""
import copy
import hashlib
import json
import shutil
import struct
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p1_matched_censored_runner_20261030_codex as run
from experiments.spectral_reachability import p1_matched_censored_consumer_20261030_codex as consume


@pytest.fixture(scope="module")
def original():
    path = run.proof.PREPARE
    p = run.native.read(path, cap=32 * 1024**2)
    return p["anchors"], p["probe_sets"]


@pytest.fixture(scope="module")
def complete_shard(tmp_path_factory, original):
    dest = tmp_path_factory.mktemp("censored_saved_replay").resolve()
    source = run.proof.DIRECTORY.parent / "anchor_00"
    report = run.native.read(source / "report.json", cap=64*1024**2)
    for path in source.glob("*.png"):
        shutil.copyfile(path, dest / path.name)
    lines = (source / "journal.jsonl").read_bytes().splitlines()
    last = json.loads(lines[-1])
    last["status"] = "recorded_not_consumed"
    del last["event_sha256"]
    last["event_sha256"] = run.native.source.digest(last)
    lines[-1] = run.native.raw(last).rstrip(b"\n")
    journal = b"\n".join(lines) + b"\n"
    (dest / "journal.jsonl").write_bytes(journal)
    report["schema"] = run.NAME + ".shard.v1"
    report["status"] = "recorded_not_consumed"
    report["journal_chain_tail"] = last["event_sha256"]
    report["journal_sha256"] = hashlib.sha256(journal).hexdigest()
    blob = run.native.raw(report)
    (dest / "report.json").write_bytes(blob)
    return dest, hashlib.sha256(blob).hexdigest(), original[0][0], original[1][0]


def test_saved_complete_shard_replay(complete_shard):
    path, sha, anchor, probes = complete_shard
    result = consume.check_shard(path, anchor, probes, sha)
    assert len(result["statuses"]) == len(result["endpoints"]) == 21
    assert result["statuses"] == ["probe_completed"] * 21


def test_external_report_hash_and_probe_substitution_rejected(complete_shard):
    path, sha, anchor, probes = complete_shard
    with pytest.raises(ValueError):
        consume.check_shard(path, anchor, probes, "0" * 64)
    altered = copy.deepcopy(probes)
    altered[0]["raw5_float32_hex"] = altered[1]["raw5_float32_hex"]
    with pytest.raises(ValueError):
        consume.check_shard(path, anchor, altered, sha)


def test_saved_censor_full_shard_and_tampering(tmp_path, complete_shard):
    source, _, anchor, probes = complete_shard
    dest = tmp_path / "censor_fixture"
    shutil.copytree(source, dest)
    report = run.native.read(dest / "report.json", cap=64*1024**2)
    events = [json.loads(x) for x in (dest / "journal.jsonl").read_bytes().splitlines()]
    row, probe = report["rows"][20], probes[20]
    trans = next(e for e in events if e["kind"] == "returned_transition" and e.get("branch") == 20
                 and e.get("phase") == "probe" and e.get("step") == 5)
    frame = next(e for e in events if e["kind"] == "fresh_observation" and e.get("branch") == 20
                 and e["receipt"]["phase"] == "probe" and e["receipt"]["step"] == 5
                 and e["receipt"]["overlay_index"] == 0)
    removed_start = frame["index"] + 1
    state = copy.deepcopy(trans["state"])
    state[1] = 529.0
    snap = copy.deepcopy(trans["snapshot"])
    snap["state"] = state
    snap["state_float64_bits"] = struct.pack("<7d", *state).hex()
    snap["bodies"]["agent"]["position"][1] = 529.0
    trans["state"], trans["snapshot"] = state, copy.deepcopy(snap)
    frame["receipt"]["state"], frame["receipt"]["snapshot"] = state, copy.deepcopy(snap)
    frame["pre"]["snapshot"], frame["post"]["snapshot"] = copy.deepcopy(snap), copy.deepcopy(snap)
    row["responses"].pop()
    row["trajectory"].pop()
    row["PNG"] -= 2
    row["last_observation"] = copy.deepcopy(frame["receipt"])
    trial = copy.deepcopy(row)
    trial.update(status="engineering_failed", error=dict(stage="branch", type="Rejected",
                                                        message="fresh RGB/canvas guard"))
    image = consume._image(dest, frame["receipt"])
    evidence = run.proof.incident(events, anchor, probe, trial, image)
    keys = ("anchor", "probe", "step", "action_float32_hex", "returned_state",
            "out_of_bounds", "frame_sha256", "blue_pixels")
    censor = {k: evidence[k] for k in keys}
    row["status"], row["censor"] = "censored_out_of_domain", censor
    scope = {k: frame[k] for k in ("branch", "anchor_index", "condition", "arm", "candidate_role", "probe")}
    events = [e for e in events if not removed_start <= e["index"] <= removed_start + 5]
    offset = next(i for i, e in enumerate(events) if e["kind"] == "branch_closed" and e.get("branch") == 20)
    events.insert(offset, dict(kind="censored_out_of_domain", proof=censor, **scope))
    events[offset + 1]["status"] = "censored_out_of_domain"
    report["counts"]["explicit_render"]["attempted"] -= 2
    report["counts"]["explicit_render"]["completed"] -= 2
    events[-1]["counts"] = report["counts"]
    previous, ids = None, {}
    for i, event in enumerate(events):
        event["index"], event["previous_sha256"] = i, previous
        if event["kind"] == "operation_dispatch_intent":
            old_id = event["receipt"]["index"]
            ids[old_id] = len(ids)
            event["receipt"]["index"] = ids[old_id]
        elif event["kind"] == "operation_returned":
            event["receipt"]["index"] = ids[event["receipt"]["index"]]
        event.pop("event_sha256", None)
        event["event_sha256"] = run.native.source.digest(event)
        previous = event["event_sha256"]
    report["receipts"] = sorted([e["receipt"] for e in events if e["kind"] == "operation_returned"],
                                key=lambda x: x["index"])
    raw = b"".join(run.native.raw(e) for e in events)
    (dest / "journal.jsonl").write_bytes(raw)
    report["journal_chain_tail"], report["journal_sha256"] = previous, hashlib.sha256(raw).hexdigest()
    report_raw = run.native.raw(report)
    (dest / "report.json").write_bytes(report_raw)
    sha = hashlib.sha256(report_raw).hexdigest()
    result = consume.check_shard(dest, anchor, probes, sha)
    assert result["statuses"][-1] == "censored_out_of_domain" and result["endpoints"][-1] is None
    report["rows"][20]["censor"]["action_float32_hex"] = "00" * 8
    bad = run.native.raw(report)
    (dest / "report.json").write_bytes(bad)
    with pytest.raises(ValueError, match="censor proof"):
        consume.check_shard(dest, anchor, probes, hashlib.sha256(bad).hexdigest())


def test_guard_is_not_suppressed_by_name():
    try:
        raise run.base.moving.Rejected("fresh RGB/canvas guard")
    except run.base.moving.Rejected as exc:
        assert not run._guard_origin(exc)
    state = np.array([4., 529., 4., 4., 0., 0., 0.])
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    try:
        run.base.moving.guard(rgb, state)
    except run.base.moving.Rejected as exc:
        assert run._guard_origin(exc)


def test_censor_proof_requires_real_guard_and_recorded_transition(tmp_path, original):
    directory = run.proof.DIRECTORY
    report = run.native.read(directory / "report.json", cap=64*1024**2)
    events = consume.saved.events(directory / "journal.jsonl", report["journal_chain_tail"])
    row = report["rows"][16]
    frame = directory / row["last_observation"]["png"]
    shutil.copyfile(frame, tmp_path / frame.name)
    store = SimpleNamespace(path=tmp_path, events=events)
    state = np.array(row["last_observation"]["state"])
    black = np.zeros((224, 224, 3), dtype=np.uint8)
    try:
        run.base.moving.guard(black, state)
    except run.base.moving.Rejected as exc:
        proof = run.censored_proof(store, original[0][5], original[1][5][16], row, exc)
        assert proof["step"] == 5 and proof["blue_pixels"] == 0
        broken = copy.deepcopy(row)
        broken["last_observation"]["state"][1] = 400
        with pytest.raises(ValueError):
            run.censored_proof(store, original[0][5], original[1][5][16], broken, exc)


@pytest.mark.parametrize("failure", [False, True])
def test_fixed_shard_continuation_and_failstop(tmp_path, original, monkeypatch, failure):
    anchor, probes = original[0][0], original[1][0]
    monkeypatch.setattr(run.base, "validate_env", lambda body: None)
    monkeypatch.setattr(run.base, "native_hooks", lambda *args: nullcontext())
    seen = []
    class Env:
        def __init__(self):
            self.unwrapped = SimpleNamespace(render_action=False, render_size=224, render_mode="rgb_array")
        def close(self):
            return None
    def fake_episode(env, root, source_plan, plan, anchor, probe, row, meter, store):
        seen.append(probe["index"])
        if failure and probe["index"] == 3:
            raise ValueError("unrelated engineering failure")
        row["status"] = "censored_out_of_domain" if probe["index"] == 1 else "probe_completed"
        row["warmup_returned"], row["own_returned"], row["PNG"] = 10, 10, 66
        for kind, n in dict(reset=2, public_step=20, physics_step=204, set_state=4,
                            goal_setter=3, goal_render=2, goal_bind=1, explicit_render=66).items():
            meter.counts[kind]["attempted"] += n
            meter.counts[kind]["completed"] += n
    monkeypatch.setattr(run, "episode", fake_episode)
    result = run.execute_shard(anchor, probes, {"roots": [None, None], "source_plan": None},
                               tmp_path / "shard", Env, {})
    if failure:
        assert result["status"] == "partial_or_failed" and seen == [0, 1, 2, 3]
        assert result["rows"][4]["status"] == "not_started"
    else:
        assert result["status"] == "recorded_not_consumed" and seen == list(range(21))
        assert result["rows"][1]["status"] == "censored_out_of_domain"


def test_prepare_binds_all_new_files_and_fixed_roster(monkeypatch, original):
    anchors, banks = original
    monkeypatch.setattr(run.old, "build", lambda: dict(fixed_anchors=16, fixed_branches=336,
                                                         anchors=anchors, probe_sets=banks, bindings={}))
    p = run.build()
    assert len(p["bindings"]) == 4
    assert all(p["bindings"][str(f)] == run.runner.physical.sha(f) for f in run.source_files())
    assert not p["P1_admission"] and not p["independent_confirmation"]
    banks = copy.deepcopy(banks)
    banks[0][1]["index"] = 0
    monkeypatch.setattr(run.old, "build", lambda: dict(fixed_anchors=16, fixed_branches=336,
                                                         anchors=anchors, probe_sets=banks, bindings={}))
    with pytest.raises(ValueError, match="roster"):
        run.build()


def test_exclusive_write(tmp_path):
    path = tmp_path / "new.json"
    sha = run._write_new(path, {"fixture": True})
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        run._write_new(path, {"fixture": False})


@pytest.mark.parametrize("fail_at", [None, 5])
def test_full_roster_orchestration_with_stub_shards(tmp_path, original, monkeypatch, fail_at):
    anchors, banks = original
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "verify_prepare", lambda *_: dict(original=dict(anchors=anchors, probe_sets=banks),
                                                               bindings={}))
    monkeypatch.setattr(run.runner.physical, "verify", lambda *_: (None, {}))
    monkeypatch.setattr(run.native, "constructor_contract", lambda: None)
    seen = []
    def fake_shard(anchor, probes, data, path, factory, bindings):
        seen.extend((anchor["index"], p["index"]) for p in probes)
        path.mkdir()
        status = "partial_or_failed" if anchor["index"] == fail_at else "recorded_not_consumed"
        run._write_new(path / "report.json", dict(status=status))
        return dict(status=status)
    monkeypatch.setattr(run, "execute_shard", fake_shard)
    output = results / (run.NAME + "_run_fixture")
    prepared = tmp_path / "fixture_prepare.json"
    prepared.write_bytes(b"fixture")
    expected = hashlib.sha256(b"fixture").hexdigest()
    result = run.run(str(prepared), expected, output, authorization=run.AUTH)
    report = json.loads((output / "run_report.json").read_bytes())
    assert hashlib.sha256((output / "run_report.json").read_bytes()).hexdigest() == result["sha256"]
    assert result["status"] == report["status"]
    if fail_at is None:
        assert len(seen) == 336 and report["status"] == "recorded_not_consumed"
    else:
        assert len(seen) == 126 and report["status"] == "partial_or_failed"
    with pytest.raises(FileExistsError if fail_at is None else ValueError):
        # Authorization is checked even if an output already exists.
        if fail_at is None:
            run._write_new(output / "run_report.json", {})
        else:
            run.run(str(prepared), expected, output, authorization="wrong")


@pytest.mark.parametrize(("status", "exit_code"), [("recorded_not_consumed", 0), ("partial_or_failed", 1)])
def test_cli_reports_partial_failure_with_nonzero(monkeypatch, capsys, status, exit_code):
    monkeypatch.setattr(sys, "argv", ["runner", "run", "--output", "/x", "--prepared", "/p",
                                     "--sha256", "a"*64, "--authorization", run.AUTH])
    monkeypatch.setattr(run, "run", lambda *a, **kw: dict(sha256="b"*64, status=status))
    assert run.main() == exit_code
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == status and printed["sha256"] == "b"*64
