"""External fixed-panel CUDA launch supervision. Does not train or access future labels."""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from experiments.spectral_reachability import p5_rgb_cuda_review_gate_20261030_codex_v1 as gate

abi, runner = gate.abi, gate.runner
NAME = "p5_rgb_cuda_supervisor_20261030_codex_v1"
AUTH = "MAIN_THREAD_APPROVED_SUPERVISED_FIXED_164_RGB_CUDA"
GATE_SHA = "e557fd97b980c2f68da25733419c1be9225cc0096aca18864999147f5eb30d16"
UUID = re.compile(r"GPU-[0-9a-fA-F-]{36}\Z")


def query_gpu(uuid, query=subprocess.run):
    """Query a physical GPU by UUID, never use an unstable numeric index."""
    abi.check(UUID.fullmatch(uuid) is not None, "physical GPU UUID required; MIG is not supported")
    def one(fields):
        result = query(["nvidia-smi", "-i", uuid, "--query-gpu=" + fields,
                        "--format=csv,noheader,nounits"], capture_output=True,
                       text=True, timeout=10, check=True)
        lines = result.stdout.strip().splitlines()
        abi.check(len(lines) == 1, "expected exactly one GPU")
        return [v.strip() for v in lines[0].split(",")]
    fields = one("uuid,memory.total,memory.free,compute_mode")
    abi.check(len(fields) == 4 and fields[0] == uuid and fields[1].isdigit() and
              fields[2].isdigit() and fields[3] in ("Default", "Exclusive_Process"),
              "GPU identity, memory, or compute mode")
    mig = one("mig.mode.current")
    abi.check(len(mig) == 1 and mig[0] in ("Disabled", "N/A", "[N/A]"), "MIG must be disabled")
    return {"uuid": uuid, "total_mib": int(fields[1]), "free_mib": int(fields[2]),
            "compute_mode": fields[3], "mig_mode": mig[0]}


def admit(snapshot, min_free_mib, min_total_mib):
    abi.check(type(min_free_mib) is int and type(min_total_mib) is int and
              1 <= min_free_mib <= min_total_mib, "positive reviewed memory thresholds")
    abi.check(snapshot["total_mib"] >= min_total_mib and
              snapshot["free_mib"] >= min_free_mib and
              0 <= snapshot["free_mib"] <= snapshot["total_mib"], "GPU memory admission denied")


def kill_group(child):
    """Only the process group created by Popen(start_new_session=True)."""
    try:
        os.killpg(child.pid, signal.SIGKILL)
        return "sigkill_sent"
    except ProcessLookupError:
        return "group_absent"


def supervise(command, env, cwd, out_file, err_file, timeout_seconds, on_started=lambda pid: None):
    """No retry; hard SIGKILL request at deadline, including child descendants."""
    started = time.monotonic()
    child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                             stdout=out_file, stderr=err_file, start_new_session=True,
                             close_fds=True)
    status, kill = "interrupted", None
    try:
        on_started(child.pid)
        try:
            rc = child.wait(timeout=max(0, timeout_seconds - (time.monotonic() - started)))
            status = "exited"
        except subprocess.TimeoutExpired:
            status = "timeout"
            kill = kill_group(child)
            try:
                rc = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                rc = None
    except BaseException:
        kill_group(child)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise
    return {"status": status, "returncode": rc, "kill": kill,
            "elapsed_seconds": time.monotonic() - started, "pid": child.pid}


def emit(path, event):
    with path.open("ab") as f:
        f.write(abi.bridge.canonical(event))
        f.flush()
        os.fsync(f.fileno())


def journal_summary(run_dir):
    path = run_dir / "events.jsonl"
    if not path.exists():
        return {"status": "no_journal"}
    if run_dir.is_symlink() or path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        return {"status": "journal_unreadable_or_oversize"}
    raw = path.read_bytes()
    digest = abi.bridge.digest(raw)
    try:
        reviewed = gate.strict_review(runner.load_prepared(), raw, digest)
        return {"status": reviewed["status"], "sha256": digest,
                "condition_counts": reviewed["condition_counts"]}
    except Exception as exc:
        return {"status": "journal_not_verified", "sha256": digest,
                "error_type": type(exc).__name__, "error": str(exc)[:500]}


def checked_gate():
    path = Path(gate.__file__)
    abi.check(not path.is_symlink() and abi.bridge.digest(path.read_bytes()) == GATE_SHA, "gate source SHA")
    gate.preflight()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--gpu-uuid", required=True)
    ap.add_argument("--min-free-mib", type=int, required=True)
    ap.add_argument("--min-total-mib", type=int, required=True)
    ap.add_argument("--hard-seconds", type=int, required=True)
    ap.add_argument("--runner-seconds", type=int, required=True)
    ap.add_argument("--run-suffix", required=True)
    args = ap.parse_args()
    abi.check(args.authorization == AUTH, "main-thread authorization required")
    abi.check(UUID.fullmatch(args.gpu_uuid) is not None and
              1 <= args.runner_seconds <= 3600 and 1 <= args.hard_seconds <= 7200 and
              re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_suffix), "arguments")
    # Validate thresholds before invoking the device query.
    abi.check(1 <= args.min_free_mib <= args.min_total_mib, "memory thresholds")
    checked_gate()
    results = abi.bridge.ROOT / "results"
    supervision = results / (NAME + "_" + args.run_suffix)
    run_name = runner.NAME + "_" + args.run_suffix
    run_dir = results / run_name
    abi.check(not run_dir.exists() and not run_dir.is_symlink(), "runner output exists")
    supervision.mkdir()
    runner.sync_dir(results)
    events = supervision / "supervisor.jsonl"
    emit(events, {"event": "created", "prepared_sha256": runner.PREP_SHA,
                  "gate_sha256": GATE_SHA, "gpu_uuid": args.gpu_uuid,
                  "min_free_mib": args.min_free_mib, "min_total_mib": args.min_total_mib,
                  "hard_seconds": args.hard_seconds, "runner_seconds": args.runner_seconds,
                  "run_name": run_name, "retry": False})
    runner.sync_dir(supervision)
    status = {"status": "not_started", "retry": False}
    launch_attempted = False
    try:
        snapshot = query_gpu(args.gpu_uuid)
        admit(snapshot, args.min_free_mib, args.min_total_mib)
        emit(events, {"event": "admitted", "snapshot": snapshot})
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(abi.bridge.ROOT))
        command = [str(abi.bridge.ROOT / ".venv/bin/python"), "-m",
                   "experiments.spectral_reachability.p5_rgb_cuda_runner_20261030_codex_v1",
                   "--authorization", runner.AUTH, "--device-index", "0",
                   "--max-seconds", str(args.runner_seconds), "--output-name", run_name]
        emit(events, {"event": "launch_attempt", "command": command})
        launch_attempted = True
        with (supervision / "stdout.log").open("xb") as out, (supervision / "stderr.log").open("xb") as err:
            runner.sync_dir(supervision)
            def interrupt(signum, frame):
                raise InterruptedError(f"supervisor signal {signum}")
            prior = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                execution = supervise(command, env, abi.bridge.ROOT, out, err, args.hard_seconds,
                                      lambda pid: emit(events, {"event": "launched", "pid": pid}))
            finally:
                for sig, handler in prior.items():
                    signal.signal(sig, handler)
            out.flush()
            err.flush()
            os.fsync(out.fileno())
            os.fsync(err.fileno())
        status = {"status": "partial" if execution["status"] != "exited" or execution["returncode"] != 0
                  else "child_exited_zero", "execution": execution, "retry": False}
    except Exception as exc:
        status = {"status": "partial_or_not_started", "error_type": type(exc).__name__,
                  "error": str(exc)[:500], "retry": False}
    if launch_attempted and ("execution" not in status or status["execution"]["returncode"] is None):
        status["journal"] = {"status": "not_read_writer_may_be_alive"}
    else:
        status["journal"] = journal_summary(run_dir)
    if status["status"] == "child_exited_zero" and status["journal"]["status"] != "complete_journal_verified":
        status["status"] = "partial"
    if status["status"] == "child_exited_zero":
        status["status"] = "complete_self_report_only"
    emit(events, {"event": "finished", **status})
    print(json.dumps(status, sort_keys=True))
    return 0 if status["status"] == "complete_self_report_only" else 2


if __name__ == "__main__":
    sys.exit(main())
