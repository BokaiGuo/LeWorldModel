"""Supervised CUDA runner with durable before/after state evidence; no automatic retry."""
import argparse
import math
import os
import re
import signal
import sys
from pathlib import Path

from experiments.spectral_reachability import p5_rgb_checkpoint_acceptance_20261030_codex_v1 as acc

abi, runner, gate, sup = acc.abi, acc.runner, acc.gate, acc.sup
NAME = "p5_rgb_instrumented_supervision_20261030_codex_v1"
MODULE = "experiments.spectral_reachability." + NAME
AUTH = "MAIN_THREAD_APPROVED_INSTRUMENTED_164_RGB_CUDA"
CHILD_AUTH = "SUPERVISED_INSTRUMENTED_164_RGB_CUDA_CHILD"
ACC_SHA = "479ad89a6ce1b319146a489770c308525d00df0885a6aa6918c2ee692768f541"
BASE_SHA = "853f22a07ff01693422e242bf032619cfe7d96548a1740b70062d92f69ccf126"
BASE_PATH = abi.bridge.ROOT / ("results/" + acc.NAME + "_BASELINE02.json")


def source_sha():
    return abi.bridge.digest(Path(__file__).read_bytes())


def preflight():
    abi.check(abi.bridge.digest(Path(acc.__file__).read_bytes()) == ACC_SHA, "acceptance source SHA")
    sup.checked_gate()
    raw = acc.read_new_file(BASE_PATH, BASE_SHA)
    base = abi.bridge.read_json(raw)
    abi.check(base["source_sha256"]["acceptance_source"] == ACC_SHA and
              base["checkpoint_sha256"] == abi.bridge.FILES["checkpoint"][1] and
              base["prepared_sha256"] == runner.PREP_SHA and base["strict_load"] is True,
              "checkpoint baseline identity")
    return base


def save_snapshot(directory, phase, state, journal, code_sha):
    record = {"schema": NAME + ".snapshot", "phase": phase, "state": state,
              "baseline_sha256": BASE_SHA, "prepared_sha256": runner.PREP_SHA,
              "producer_sha256": code_sha, "anchor_seq": journal.seq - 1,
              "anchor_hash": journal.prev}
    gate.write_new(directory / ("state_" + phase + ".json"), record)


class InstrumentedBackend:
    def __init__(self, backend, journal, directory, base, capture=acc.capture, code_sha=None):
        self.backend, self.journal, self.directory, self.base = backend, journal, directory, base
        self.capture, self.code_sha = capture, code_sha or source_sha()
        self.encodes, self.transitions, self.before = 0, 0, None

    def metadata(self):
        return self.backend.metadata()

    def encode(self, values):
        abi.check(self.encodes < 164 and self.transitions == 5 * self.encodes, "encode ordering")
        if self.encodes == 0:
            self.before = self.capture(self.backend.model, self.backend.torch)
            acc.check_snapshots(self.base, self.before, self.before)
            save_snapshot(self.directory, "before", self.before, self.journal, self.code_sha)
        self.encodes += 1
        return self.backend.encode(values)

    def step(self, values, index):
        abi.check(self.encodes > 0 and index == self.transitions % 5 and
                  self.transitions // 5 == self.encodes - 1, "transition ordering")
        output = self.backend.step(values, index)
        self.transitions += 1
        if self.transitions == 820:
            abi.check(self.encodes == 164, "last branch")
            after = self.capture(self.backend.model, self.backend.torch)
            acc.check_snapshots(self.base, self.before, after)
            save_snapshot(self.directory, "after", after, self.journal, self.code_sha)
        return output


def checked_snapshot(path, phase, journal_events, base, code_sha):
    abi.check(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024 * 1024,
              phase + " snapshot missing or oversize")
    raw = path.read_bytes()
    obj = abi.bridge.read_json(raw)
    abi.check(abi.bridge.canonical(obj) == raw and set(obj) ==
              {"schema", "phase", "state", "baseline_sha256", "prepared_sha256", "producer_sha256",
               "anchor_seq", "anchor_hash"} and obj["schema"] == NAME + ".snapshot" and
              obj["phase"] == phase and obj["baseline_sha256"] == BASE_SHA and
              obj["prepared_sha256"] == runner.PREP_SHA and obj["producer_sha256"] == code_sha,
              phase + " snapshot identity")
    seq = obj["anchor_seq"]
    abi.check(type(seq) is int and 0 <= seq < len(journal_events), "snapshot anchor sequence")
    event = journal_events[seq]
    target = (0, 0, "encode_state") if phase == "before" else (163, 5, "predict_next")
    abi.check(event["hash"] == obj["anchor_hash"] and event["kind"] == "attempt" and
              (event["branch"], event["step"], event["op"]) == target, "snapshot anchor")
    return obj, abi.bridge.digest(raw)


def review_run(run_dir, base, code_sha):
    journal_path = run_dir / "events.jsonl"
    abi.check(run_dir.is_dir() and not run_dir.is_symlink() and journal_path.is_file() and
              not journal_path.is_symlink() and journal_path.stat().st_size <= 16 * 1024 * 1024, "journal file")
    data = journal_path.read_bytes()
    digest = abi.bridge.digest(data)
    reviewed = gate.strict_review(runner.load_prepared(), data, digest)
    abi.check(reviewed["status"] == "complete_journal_verified" and
              reviewed["condition_counts"] == {"off": {"denominator": 82, "complete": 82},
                                               "on": {"denominator": 82, "complete": 82}}, "complete journal")
    events = [abi.bridge.read_json(line) for line in data.splitlines()]
    before, before_sha = checked_snapshot(run_dir / "state_before.json", "before", events, base, code_sha)
    after, after_sha = checked_snapshot(run_dir / "state_after.json", "after", events, base, code_sha)
    acc.check_snapshots(base, before["state"], after["state"])
    return {"journal_sha256": digest, "before_sha256": before_sha, "after_sha256": after_sha,
            "condition_counts": reviewed["condition_counts"]}


def child(args):
    abi.check(args.authorization == CHILD_AUTH and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_suffix) and
        1 <= args.runner_seconds <= 3600 and args.source_sha == source_sha(), "child authority")
    base = preflight()
    prepared = runner.load_prepared()
    checkpoint, _ = runner.verify_sources()
    for row in prepared["rows"]:
        abi.materialize(row)
    directory = abi.bridge.ROOT / "results" / (NAME + "_run_" + args.run_suffix)
    abi.check(not directory.exists() and not directory.is_symlink(), "run directory exists")
    backend = runner.CudaBackend(checkpoint, prepared, 0)
    journal = runner.Journal(directory)
    try:
        status = runner.run_protocol(prepared, InstrumentedBackend(backend, journal, directory, base),
                                     journal, args.runner_seconds)
    finally:
        journal.close()
    print("child status:", status)
    return 0 if status == "complete" else 2


def launch(args):
    abi.check(args.authorization == AUTH and sup.UUID.fullmatch(args.gpu_uuid) and
              re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_suffix) and
              1 <= args.min_free_mib <= args.min_total_mib and 1 <= args.hard_seconds <= 7200 and
              1 <= args.runner_seconds <= 3600, "launch arguments")
    base = preflight()
    code_sha = source_sha()
    results = abi.bridge.ROOT / "results"
    directory = results / (NAME + "_supervisor_" + args.run_suffix)
    run_dir = results / (NAME + "_run_" + args.run_suffix)
    abi.check(not run_dir.exists() and not run_dir.is_symlink(), "run directory exists")
    directory.mkdir()
    runner.sync_dir(results)
    events = directory / "supervisor.jsonl"
    sup.emit(events, {"event": "created", "producer_sha256": code_sha, "baseline_sha256": BASE_SHA,
                      "prepared_sha256": runner.PREP_SHA, "gpu_uuid": args.gpu_uuid,
                      "min_free_mib": args.min_free_mib, "min_total_mib": args.min_total_mib,
                      "hard_seconds": args.hard_seconds, "runner_seconds": args.runner_seconds,
                      "run_suffix": args.run_suffix, "retry": False})
    runner.sync_dir(directory)
    attempted, execution = False, None
    try:
        gpu = sup.query_gpu(args.gpu_uuid)
        sup.admit(gpu, args.min_free_mib, args.min_total_mib)
        sup.emit(events, {"event": "admitted", "snapshot": gpu})
        command = [str(abi.bridge.ROOT / ".venv/bin/python"), "-m", MODULE, "child",
                   "--authorization", CHILD_AUTH, "--run-suffix", args.run_suffix,
                   "--runner-seconds", str(args.runner_seconds), "--source-sha", code_sha]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(abi.bridge.ROOT))
        sup.emit(events, {"event": "launch_attempt", "command": command})
        attempted = True
        with (directory / "stdout.log").open("xb") as out, (directory / "stderr.log").open("xb") as err:
            runner.sync_dir(directory)
            def interrupt(signum, frame):
                raise InterruptedError(f"signal {signum}")
            previous = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
            try:
                execution = sup.supervise(command, env, abi.bridge.ROOT, out, err, args.hard_seconds,
                                          lambda pid: sup.emit(events, {"event": "launched", "pid": pid}))
            finally:
                for s, handler in previous.items():
                    signal.signal(s, handler)
            for f in (out, err):
                f.flush()
                os.fsync(f.fileno())
        status = {"status": "partial", "execution": execution, "retry": False}
        if execution["status"] == "exited" and execution["returncode"] == 0:
            abi.check(math.isfinite(execution["elapsed_seconds"]) and
                      execution["elapsed_seconds"] <= args.hard_seconds, "external deadline exceeded")
            abi.check(source_sha() == code_sha, "producer source changed")
            preflight()
            status["review"] = review_run(run_dir, base, code_sha)
            status["status"] = "complete_self_report_only"
        elif execution["returncode"] is not None:
            status["evidence"] = sup.journal_summary(run_dir)
        else:
            status["evidence"] = {"status": "not_read_writer_may_be_alive"}
    except Exception as exc:
        status = {"status": "partial_or_not_started", "retry": False,
                  "error_type": type(exc).__name__, "error": str(exc)[:500],
                  "evidence": (sup.journal_summary(run_dir) if execution is not None and
                               execution["returncode"] is not None else
                               {"status": "not_read_writer_may_be_alive" if attempted else "not_started"})}
    sup.emit(events, {"event": "finished", **status})
    print(status)
    return 0 if status["status"] == "complete_self_report_only" else 2


def accept(args):
    abi.check(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_suffix), "suffix")
    base = preflight()
    results = abi.bridge.ROOT / "results"
    directory = results / (NAME + "_supervisor_" + args.run_suffix)
    run_dir = results / (NAME + "_run_" + args.run_suffix)
    abi.check(not directory.is_symlink(), "supervisor path")
    raw = acc.read_new_file(directory / "supervisor.jsonl", args.supervisor_sha)
    abi.check(raw.endswith(b"\n"), "supervisor log truncated")
    events = [abi.bridge.read_json(line) for line in raw.splitlines()]
    abi.check([e["event"] for e in events] == ["created", "admitted", "launch_attempt", "launched", "finished"],
              "supervisor sequence")
    created, admitted, launch_event, launched, finished = events
    code_sha = source_sha()
    command = [str(abi.bridge.ROOT / ".venv/bin/python"), "-m", MODULE, "child", "--authorization", CHILD_AUTH,
               "--run-suffix", args.run_suffix, "--runner-seconds", str(created["runner_seconds"]),
               "--source-sha", code_sha]
    abi.check(created["producer_sha256"] == code_sha and created["baseline_sha256"] == BASE_SHA and
              created["prepared_sha256"] == runner.PREP_SHA and created["run_suffix"] == args.run_suffix and
              created["retry"] is False and finished["retry"] is False and launch_event["command"] == command and
              type(created["hard_seconds"]) is int and 1 <= created["hard_seconds"] <= 7200 and
              type(created["runner_seconds"]) is int and 1 <= created["runner_seconds"] <= 3600 and
              admitted["snapshot"]["uuid"] == created["gpu_uuid"] and sup.UUID.fullmatch(created["gpu_uuid"]) and
              admitted["snapshot"]["mig_mode"] in ("Disabled", "N/A", "[N/A]") and
              admitted["snapshot"]["compute_mode"] in ("Default", "Exclusive_Process"), "supervisor identity")
    sup.admit(admitted["snapshot"], created["min_free_mib"], created["min_total_mib"])
    execution = finished["execution"]
    abi.check(finished["status"] == "complete_self_report_only" and execution["status"] == "exited" and
              execution["returncode"] == 0 and execution["pid"] == launched["pid"] and
              isinstance(execution["elapsed_seconds"], (int, float)) and
              math.isfinite(execution["elapsed_seconds"]) and
              0 <= execution["elapsed_seconds"] <= created["hard_seconds"], "supervisor completion")
    review = review_run(run_dir, base, code_sha)
    abi.check(review == finished["review"] and review["journal_sha256"] == args.journal_sha,
              "independent journal SHA/review")
    output = args.output_name
    abi.check(Path(output).name == output and output.startswith(NAME + "_accept_") and output.endswith(".json"),
              "exclusive acceptance name")
    report = {"schema": NAME + ".accept", "status": "self_report_consistency_only_not_scientific",
              "producer_sha256": code_sha, "baseline_sha256": BASE_SHA, "supervisor_sha256": args.supervisor_sha,
              "review": review, "P5_admission": False,
              "boundary": "self-reported snapshots and journal; no independent hardware or scientific validation"}
    print(gate.write_new(results / output, report))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("launch")
    for k in ("authorization", "gpu-uuid", "run-suffix"):
        a.add_argument("--" + k, required=True)
    for k in ("min-free-mib", "min-total-mib", "hard-seconds", "runner-seconds"):
        a.add_argument("--" + k, type=int, required=True)
    c = sub.add_parser("child")
    for k in ("authorization", "run-suffix", "source-sha"):
        c.add_argument("--" + k, required=True)
    c.add_argument("--runner-seconds", type=int, required=True)
    v = sub.add_parser("accept")
    for k in ("run-suffix", "supervisor-sha", "journal-sha", "output-name"):
        v.add_argument("--" + k, required=True)
    args = ap.parse_args()
    return {"launch": launch, "child": child, "accept": accept}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
