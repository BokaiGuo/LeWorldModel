"""Supervised saved-RGB refresh CUDA execution and strict saved-only acceptance."""
import argparse
import math
import os
import re
import signal
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p5_rgb_instrumented_supervision_20261030_codex_v1 as wiring
from experiments.spectral_reachability import p5_rgb_observation_refresh_20261030_codex_v1 as refresh

bridge, runner, gate, sup, acc = wiring.abi.bridge, wiring.runner, wiring.gate, wiring.sup, wiring.acc
NAME = "p5_rgb_refresh_supervision_20261030_codex_v1"
MODULE = "experiments.spectral_reachability." + NAME
AUTH = "MAIN_THREAD_APPROVED_SUPERVISED_REFRESH_164_CUDA"
CHILD_AUTH = "SUPERVISED_REFRESH_164_CUDA_CHILD"
REFRESH_SHA = "856497a80798947c6f10742c2fb22a653ce325d84786c3023ee03ec465dd8d73"
WIRING_SHA = "f1f3f1d0bc962b6f153ace297f12da0f6ff63eb990252b96a8a63d62d96ffee7"
PREP_SHA = "46bfde766c6ce6d99c95cba0c37909c4437dcfbdb9e5e049da02ca0d83403409"
PREP_PATH = bridge.ROOT / "results" / (refresh.NAME + "_PREP01.json")
BASE = {"seq", "prev", "kind", "hash"}
IDENTITY = {"branch", "root_slot", "condition", "task_id", "selection_digest", "horizon",
            "previous_plan_step", "current_plan_step", "target_plan_step", "previous_sha256",
            "image_sha256", "action_sha256", "op"}


def check(value, message):
    refresh.check(value, message)


def source_sha():
    return bridge.digest(Path(__file__).read_bytes())


def preflight(rebuild=True):
    check(refresh.digest(Path(refresh.__file__).read_bytes()) == REFRESH_SHA and
          wiring.source_sha() == WIRING_SHA, "source pins")
    base = wiring.preflight()
    raw = refresh.checked_path(PREP_PATH, PREP_SHA, 8 * 1024 * 1024)
    prepared = bridge.read_json(raw)
    check(prepared["source_sha256"]["consumer_source"] == REFRESH_SHA and
          prepared["source_sha256"]["checkpoint"] == bridge.FILES["checkpoint"][1],
          "prepared sources")
    if rebuild:
        check(raw == bridge.canonical(refresh.prepare(*refresh.sources())), "prepared exact rebuild")
    return base, prepared


def snapshot(directory, phase, state, journal, producer_sha):
    gate.write_new(directory / ("state_" + phase + ".json"),
                   {"schema": NAME + ".snapshot", "phase": phase, "state": state,
                    "baseline_sha256": wiring.BASE_SHA, "prepared_sha256": PREP_SHA,
                    "producer_sha256": producer_sha, "anchor_seq": journal.seq - 1, "anchor_hash": journal.prev})


class InstrumentedBackend:
    def __init__(self, backend, journal, directory, base, capture=acc.capture):
        self.backend, self.journal, self.directory, self.base, self.capture = backend, journal, directory, base, capture
        self.encodes = self.steps = 0
        self.before = None

    def metadata(self):
        return self.backend.metadata()

    def encode(self, values):
        check(self.encodes == self.steps and self.encodes < 820, "refresh encode order")
        if self.encodes == 0:
            self.before = self.capture(self.backend.model, self.backend.torch)
            acc.check_snapshots(self.base, self.before, self.before)
            snapshot(self.directory, "before", self.before, self.journal, source_sha())
        result = self.backend.encode(values)
        self.encodes += 1
        return result

    def step(self, values, index):
        check(index == 0 and self.encodes == self.steps + 1, "refresh transition order")
        result = self.backend.step(values, index)
        self.steps += 1
        if self.steps == 820:
            after = self.capture(self.backend.model, self.backend.torch)
            acc.check_snapshots(self.base, self.before, after)
            snapshot(self.directory, "after", after, self.journal, source_sha())
        return result


def identity(row, step, op):
    return {"branch": row["index"], "root_slot": row["root_slot"], "condition": row["condition"],
            "task_id": row["task_id"], "selection_digest": row["selection_digest"], "horizon": step["horizon"],
            "previous_plan_step": step["previous_plan_step"], "current_plan_step": step["current_plan_step"],
            "target_plan_step": step["target_plan_step"],
            "previous_sha256": step["previous_rgb"]["model_input_chw_float32_sha256"],
            "image_sha256": step["current_rgb"]["model_input_chw_float32_sha256"],
            "action_sha256": step["action_sha256"], "op": op}


def review_journal(prepared, raw, expected_sha):
    check(bridge.digest(raw) == expected_sha and len(raw) <= 16 * 1024 * 1024 and raw.endswith(b"\n"),
          "journal SHA/bounds")
    lines = raw.splitlines()
    check(len(lines) == 3282, "complete fixed call count")
    events, prev = [], "0" * 64
    for seq, line in enumerate(lines):
        e = bridge.read_json(line)
        check(bridge.canonical(e).rstrip(b"\n") == line and e.get("seq") == seq and e.get("prev") == prev,
              "journal canonical chain")
        body = dict(e)
        claim = body.pop("hash", None)
        check(claim == bridge.digest(bridge.canonical(body)), "event hash")
        prev = claim
        events.append(e)
    start, end = events[0], events[-1]
    check(set(start) == BASE | {"schema", "prepared_sha256", "source_sha256", "max_seconds", "backend",
                                "branch_denominator", "pair_denominator"}
          and start["kind"] == "start" and start["schema"] == refresh.NAME + ".journal"
          and start["prepared_sha256"] == PREP_SHA and start["source_sha256"] == REFRESH_SHA
          and start["branch_denominator"] == 164 and start["pair_denominator"] == 80
          and type(start["max_seconds"]) is int and 1 <= start["max_seconds"] <= 3600, "start identity")
    backend = start["backend"]
    check(set(backend) == gate.BACKEND and backend["device"] == "cuda:0"
          and backend["checkpoint_sha256"] == bridge.FILES["checkpoint"][1]
          and backend["strict_load"] is True and backend["frozen"] is True and backend["eval"] is True
          and all(isinstance(backend[k], str) and backend[k] for k in ("device_name", "torch_version", "cuda_version")),
          "backend self-report")
    check(len(prepared["rows"]) == 164 and len(prepared["pairs"]) == 80, "prepared denominator")
    rows, pos = [], 1
    for i, row in enumerate(prepared["rows"]):
        check((row["index"], row["root_slot"], row["condition"]) ==
              (i, i // 82, ("off", "on")[(i // 41) % 2]) and len(row["steps"]) == 5, "row order")
        outputs = []
        for h, step in enumerate(row["steps"], 1):
            check((step["horizon"], step["previous_plan_step"], step["current_plan_step"], step["target_plan_step"])
                  == (h, h + 3, h + 4, h + 5), "step order")
            recorded = {}
            for op in ("encode_state", "predict_next"):
                a, r = events[pos:pos + 2]
                pos += 2
                expected = identity(row, step, op)
                check(set(a) == BASE | IDENTITY and set(r) == BASE | IDENTITY | {"output"}
                      and a["kind"] == "attempt" and r["kind"] == "returned"
                      and all(a[k] == r[k] == v for k, v in expected.items()), "operation identity")
                output = r["output"]
                check(set(output) == {"values", "sha256"}, "output envelope")
                array = np.asarray(output["values"], dtype="<f4")
                check(array.shape == (6,) and np.isfinite(array).all() and
                      bridge.digest(array.tobytes()) == output["sha256"], "output bytes")
                recorded[op] = output
            outputs.append({"horizon": h, **recorded})
        rows.append({"index": i, "root_slot": row["root_slot"], "condition": row["condition"], "steps": outputs})
    check(pos == len(events) - 1 and set(end) == BASE | {"status", "elapsed_seconds"} and
          end["kind"] == "end" and end["status"] == "complete" and
          type(end["elapsed_seconds"]) in (int, float) and math.isfinite(end["elapsed_seconds"]) and
          0 <= end["elapsed_seconds"], "complete end")
    return events, rows


def check_snapshot(path, phase, events, base, producer_sha):
    check(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024 * 1024, "snapshot file")
    raw = path.read_bytes()
    obj = bridge.read_json(raw)
    check(bridge.canonical(obj) == raw and set(obj) ==
          {"schema", "phase", "state", "baseline_sha256", "prepared_sha256", "producer_sha256", "anchor_seq", "anchor_hash"}
          and obj["schema"] == NAME + ".snapshot" and obj["phase"] == phase
          and obj["baseline_sha256"] == wiring.BASE_SHA and obj["prepared_sha256"] == PREP_SHA
          and obj["producer_sha256"] == producer_sha, "snapshot identity")
    seq = obj["anchor_seq"]
    check(type(seq) is int and seq == (1 if phase == "before" else 3279), "snapshot sequence")
    event = events[seq]
    check(event["kind"] == "attempt" and event["hash"] == obj["anchor_hash"] and
          (event["branch"], event["horizon"], event["op"]) ==
          ((0, 1, "encode_state") if phase == "before" else (163, 5, "predict_next")), "snapshot anchor")
    return obj["state"], bridge.digest(raw)


def review_run(directory, base, prepared, producer_sha, expected_journal_sha=None):
    path = directory / "events.jsonl"
    check(directory.is_dir() and not directory.is_symlink() and path.is_file() and not path.is_symlink()
          and path.stat().st_size <= 16 * 1024 * 1024, "journal path")
    raw = path.read_bytes()
    sha = bridge.digest(raw)
    check(expected_journal_sha is None or sha == expected_journal_sha, "external journal SHA")
    events, rows = review_journal(prepared, raw, sha)
    before, bsha = check_snapshot(directory / "state_before.json", "before", events, base, producer_sha)
    after, asha = check_snapshot(directory / "state_after.json", "after", events, base, producer_sha)
    acc.check_snapshots(base, before, after)
    return {"journal_sha256": sha, "before_sha256": bsha, "after_sha256": asha,
            "condition_counts": {"off": {"denominator": 82, "complete": 82},
                                 "on": {"denominator": 82, "complete": 82}},
            "encode_returned": 820, "predict_returned": 820, "rows": rows}


def paths(suffix):
    check(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", suffix) is not None, "suffix")
    results = bridge.ROOT / "results"
    check(results.resolve() == results and not results.is_symlink(), "results path")
    return results / (NAME + "_supervisor_" + suffix), results / (NAME + "_run_" + suffix)


def command(suffix, runner_seconds, code_sha):
    return [str(bridge.ROOT / ".venv/bin/python"), "-m", MODULE, "child",
            "--authorization", CHILD_AUTH, "--run-suffix", suffix,
            "--runner-seconds", str(runner_seconds), "--source-sha", code_sha]


def child(args):
    check(args.authorization == CHILD_AUTH and args.source_sha == source_sha()
          and 1 <= args.runner_seconds <= 3600, "child authorization")
    _, directory = paths(args.run_suffix)
    check(not directory.exists() and not directory.is_symlink(), "run directory exists")
    base, prepared = preflight()
    for row in prepared["rows"]:
        for step in row["steps"]:
            refresh.materialize(step)
    checkpoint, _ = runner.verify_sources()
    backend = runner.CudaBackend(checkpoint, prepared, 0)
    journal = runner.Journal(directory)
    try:
        status = refresh.execute(prepared, PREP_SHA, InstrumentedBackend(backend, journal, directory, base),
                                 journal, args.runner_seconds)
    finally:
        journal.close()
    print("child status:", status)
    return 0 if status == "complete" else 2


def launch(args):
    check(args.authorization == AUTH and sup.UUID.fullmatch(args.gpu_uuid) and
          1 <= args.min_free_mib <= args.min_total_mib and 1 <= args.hard_seconds <= 7200
          and 1 <= args.runner_seconds <= 3600, "launch arguments")
    supervision, run_dir = paths(args.run_suffix)
    check(not run_dir.exists() and not run_dir.is_symlink(), "run directory exists")
    base, prepared = preflight()
    code_sha = source_sha()
    supervision.mkdir()
    runner.sync_dir(supervision.parent)
    log = supervision / "supervisor.jsonl"
    sup.emit(log, {"event": "created", "producer_sha256": code_sha, "baseline_sha256": wiring.BASE_SHA,
                   "prepared_sha256": PREP_SHA, "gpu_uuid": args.gpu_uuid, "run_suffix": args.run_suffix,
                   "min_free_mib": args.min_free_mib, "min_total_mib": args.min_total_mib,
                   "hard_seconds": args.hard_seconds, "runner_seconds": args.runner_seconds, "retry": False})
    runner.sync_dir(supervision)
    execution, attempted = None, False
    try:
        gpu = sup.query_gpu(args.gpu_uuid)
        sup.admit(gpu, args.min_free_mib, args.min_total_mib)
        sup.emit(log, {"event": "admitted", "snapshot": gpu})
        cmd = command(args.run_suffix, args.runner_seconds, code_sha)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(bridge.ROOT))
        sup.emit(log, {"event": "launch_attempt", "command": cmd})
        attempted = True
        with (supervision / "stdout.log").open("xb") as out, (supervision / "stderr.log").open("xb") as err:
            runner.sync_dir(supervision)
            def interrupt(signum, frame):
                raise InterruptedError(f"signal {signum}")
            previous = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
            try:
                execution = sup.supervise(cmd, env, bridge.ROOT, out, err, args.hard_seconds,
                                          lambda pid: sup.emit(log, {"event": "launched", "pid": pid}))
            finally:
                for s, handler in previous.items():
                    signal.signal(s, handler)
            for f in (out, err):
                f.flush()
                os.fsync(f.fileno())
        status = {"status": "partial", "execution": execution, "retry": False}
        if execution["status"] == "exited" and execution["returncode"] == 0:
            check(math.isfinite(execution["elapsed_seconds"]) and execution["elapsed_seconds"] <= args.hard_seconds
                  and source_sha() == code_sha, "deadline/source")
            preflight()
            status.update(status="complete_self_report_only", review=review_run(run_dir, base, prepared, code_sha))
    except Exception as exc:
        status = {"status": "partial_or_not_started", "retry": False, "error_type": type(exc).__name__,
                  "error": str(exc)[:500], "writer_may_be_alive": bool(attempted and
                         (execution is None or execution["returncode"] is None))}
    sup.emit(log, {"event": "finished", **status})
    print(status["status"])
    return 0 if status["status"] == "complete_self_report_only" else 2


def validate_supervisor(events, suffix, producer_sha):
    check([e.get("event") for e in events] == ["created", "admitted", "launch_attempt", "launched", "finished"],
          "supervisor sequence")
    created, admitted, launch_event, launched, finished = events
    check(created["producer_sha256"] == producer_sha and created["baseline_sha256"] == wiring.BASE_SHA
          and created["prepared_sha256"] == PREP_SHA and created["run_suffix"] == suffix
          and created["retry"] is False and finished["retry"] is False and
          type(created["hard_seconds"]) is int and 1 <= created["hard_seconds"] <= 7200 and
          type(created["runner_seconds"]) is int and 1 <= created["runner_seconds"] <= 3600 and
          sup.UUID.fullmatch(created["gpu_uuid"]) and admitted["snapshot"]["uuid"] == created["gpu_uuid"]
          and admitted["snapshot"]["mig_mode"] in ("Disabled", "N/A", "[N/A]")
          and admitted["snapshot"]["compute_mode"] in ("Default", "Exclusive_Process")
          and launch_event["command"] == command(suffix, created["runner_seconds"], producer_sha),
          "supervisor binding")
    sup.admit(admitted["snapshot"], created["min_free_mib"], created["min_total_mib"])
    execution = finished["execution"]
    check(finished["status"] == "complete_self_report_only" and execution["status"] == "exited"
          and execution["returncode"] == 0 and execution["pid"] == launched["pid"]
          and type(execution["elapsed_seconds"]) in (int, float) and math.isfinite(execution["elapsed_seconds"])
          and 0 <= execution["elapsed_seconds"] <= created["hard_seconds"], "supervisor completion")
    return finished["review"]


def accept(args):
    supervision, run_dir = paths(args.run_suffix)
    check(supervision.is_dir() and not supervision.is_symlink(), "supervisor directory")
    code_sha = source_sha()
    base, prepared = preflight()
    raw = acc.read_new_file(supervision / "supervisor.jsonl", args.supervisor_sha)
    check(raw.endswith(b"\n"), "supervisor truncated")
    events = [bridge.read_json(line) for line in raw.splitlines()]
    check(all(bridge.canonical(e).rstrip(b"\n") == line for e, line in zip(events, raw.splitlines(), strict=True)),
          "supervisor canonical")
    expected_review = validate_supervisor(events, args.run_suffix, code_sha)
    review = review_run(run_dir, base, prepared, code_sha, args.journal_sha)
    check(review == expected_review, "supervisor review mismatch")
    check(re.fullmatch(re.escape(NAME + "_accept_") + r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\.json",
                       args.output_name) is not None, "accept name")
    report = {"schema": NAME + ".accept", "status": "self_report_consistency_only_not_scientific",
              "producer_sha256": code_sha, "refresh_source_sha256": REFRESH_SHA, "baseline_sha256": wiring.BASE_SHA,
              "prepared_sha256": PREP_SHA, "supervisor_sha256": args.supervisor_sha,
              "review": review, "P5_admission": False,
              "boundary": "saved self-report consistency only; no independent hardware proof or scientific admission"}
    print(gate.write_new(supervision.parent / args.output_name, report))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("launch")
    for k in ("authorization", "gpu-uuid", "run-suffix"):
        p.add_argument("--" + k, required=True)
    for k in ("min-free-mib", "min-total-mib", "hard-seconds", "runner-seconds"):
        p.add_argument("--" + k, type=int, required=True)
    p = sub.add_parser("child")
    for k in ("authorization", "run-suffix", "source-sha"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--runner-seconds", type=int, required=True)
    p = sub.add_parser("accept")
    for k in ("run-suffix", "supervisor-sha", "journal-sha", "output-name"):
        p.add_argument("--" + k, required=True)
    args = ap.parse_args()
    return {"launch": launch, "child": child, "accept": accept}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
