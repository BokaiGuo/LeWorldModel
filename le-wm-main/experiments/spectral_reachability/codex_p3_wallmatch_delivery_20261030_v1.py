"""Fixed per-process deadline exploration using the sealed queued host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.spectral_reachability import codex_p3_wallmatch_host_20261030_v1 as host
from experiments.spectral_reachability import codex_p3_queued_cost_batch_20261030_v1 as old

NAME = "codex_p3_wallmatch_delivery_20261030_v1"
SECONDS = 30.0
ROOT = host.ROOT
SLOTS = tuple(old.SLOTS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def material(prepare, prepare_sha):
    host.require(digest(prepare) == prepare_sha, "host prepare hash")
    files = [Path(__file__).resolve(),
             ROOT / "tests/test_codex_p3_wallmatch_20261030_v1.py",
             ROOT / "innerdocs/codex_p3_wallmatch_protocol_20261030_v1.md"]
    return {"schema": NAME + ".contract", "host_prepare": str(Path(prepare).resolve()),
            "host_prepare_sha256": prepare_sha,
            "sources": {str(p): digest(p) for p in files},
            "slots": [list(x) for x in SLOTS], "wall_seconds": SECONDS,
            "wall_scope": "process launch through return including cooperative queue",
            "consumer_outside_deadline": True, "retry": False, "scientific_pass": False}


def exclusive(path, prefix):
    p = Path(path)
    host.require(p.is_absolute() and p.parent == ROOT / "results" and
                 p.name.startswith(NAME + prefix) and not os.path.lexists(p), "exclusive output")
    return p


def run_deadline(cmd, seconds):
    start = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=max(0, seconds - (time.monotonic() - start)))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
    elapsed = time.monotonic() - start
    return {"exit_code": proc.returncode, "wall_seconds": elapsed,
            "deadline_exceeded": timed_out or elapsed > seconds,
            "stdout": stdout[-4096:], "stderr": stderr[-8192:]}


def execute(contract_path, contract_sha, out):
    host.require(digest(contract_path) == contract_sha, "contract hash")
    contract = json.loads(Path(contract_path).read_bytes())
    host.require(contract == material(contract["host_prepare"], contract["host_prepare_sha256"]),
                 "contract or source mismatch")
    out = exclusive(out, "_RUN_")
    paths = [(ROOT / "results" / (host.NAME + f"_RUN_WALL_P{p}_{a}_01"),
              ROOT / "results" / (host.NAME + f"_CONSUME_WALL_P{p}_{a}_01.json"))
             for p, a in SLOTS]
    host.require(all(not os.path.lexists(x) for pair in paths for x in pair), "slot outputs exist")
    out.mkdir(mode=0o700)
    ledger = host.journal.Journal(out / "events.jsonl", 2 * 1024**2, 128 * 1024)
    rows = []
    failure = None
    try:
        ledger.event({"kind": "batch_started", "contract_sha256": contract_sha,
                      "slots": contract["slots"], "scientific_pass": False})
        for (point, arm), (run, receipt) in zip(SLOTS, paths, strict=True):
            ledger.event({"kind": "slot_attempt", "point": point, "arm": arm,
                          "state": "not_a_result", "run": str(run)})
            cmd = [sys.executable, "-B", "-m", "experiments.spectral_reachability." + host.NAME,
                   "run", "--prepare", contract["host_prepare"], "--sha256",
                   contract["host_prepare_sha256"], "--point", str(point), "--arm", arm,
                   "--output", str(run)]
            attempt = run_deadline(cmd, SECONDS)
            row = {"point": point, "arm": arm, "attempt": attempt,
                   "delivered_and_saved": False, "run": str(run), "receipt": None}
            if attempt["exit_code"] == 0 and not attempt["deadline_exceeded"]:
                try:
                    report_path = run / "report.json"
                    report_sha = digest(report_path)
                    report = json.loads(report_path.read_bytes())
                    host.require(report["status"] == "panel_completed_unverified" and
                                 report["point"] == point and report["arm"] == arm and
                                 report["model_before"] == report["model_after"], "host report")
                    cmd = [sys.executable, "-B", "-m", "experiments.spectral_reachability." + host.NAME,
                           "consume", "--run", str(run), "--sha256", report_sha,
                           "--output", str(receipt)]
                    checked = subprocess.run(cmd, text=True, capture_output=True)
                    host.require(checked.returncode == 0, "saved consumer failed: " + checked.stderr[-2048:])
                    saved = json.loads(receipt.read_bytes())
                    host.require(saved["status"] == "semantic_saved_checked_not_scientifically_admitted"
                                 and len(saved["rows"]) == 1, "saved status")
                    actual = saved["rows"][0]
                    scheduler = actual["attempted"]["scheduler"]
                    host.require(scheduler == 1, "scheduler charge")
                    caps = host.bridge.caps(point)
                    host.require(set(actual["attempted"]) == set(caps) and
                                 all(type(actual["attempted"][k]) is int and
                                     0 <= actual["attempted"][k] <= v for k, v in caps.items()) and
                                 report["elapsed_seconds"] < host.WALL and
                                 report["peak_allocated_bytes"] <= 3*1024**3 and
                                 report["peak_reserved_bytes"] <= 4*1024**3, "axis and host caps")
                    events = host.journal.read(run / "events.jsonl", report["journal_tail"], host.CAP)
                    timing = old.intervals(events, point, arm)
                    row.update(delivered_and_saved=True, report_sha256=report_sha,
                               receipt=str(receipt), receipt_sha256=digest(receipt),
                               allocation=actual["allocation"], actual=actual["attempted"],
                               caps=caps, lock_wait_seconds=report["lock_wait_seconds"],
                               child_wall_seconds=report["elapsed_seconds"],
                               cuda_preflight_seconds=report["cuda_preflight_seconds"],
                               model_load_seconds=report["model_load_seconds"],
                               model_load_process_io_delta=report["model_load_process_io_delta"],
                               peak_allocated_bytes=report["peak_allocated_bytes"],
                               peak_reserved_bytes=report["peak_reserved_bytes"], **timing)
                except Exception as exc:
                    row["consumer_error"] = {"type": type(exc).__name__, "message": str(exc)[:2048]}
            rows.append(row)
            ledger.event({"kind": "slot_result", "row": row})
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    try:
        ledger.append({"kind": "batch_terminal", "completed_slots": len(rows),
                       "failure": failure, "scientific_pass": False}, terminal=True)
        result = {"schema": NAME + ".report", "contract_sha256": contract_sha,
                  "rows": rows, "error": failure, "journal_tail": ledger.tail,
                  "scientific_pass": False, "total_flops": "unknown", "energy": "unknown"}
    finally:
        ledger.close()
    host.write_new(out / "report.json", result)
    print(json.dumps({"slots": len(rows), "delivered": sum(x["delivered_and_saved"] for x in rows),
                      "error": failure}))
    host.require(failure is None and len(rows) == 6 and
                 all(row["delivered_and_saved"] for row in rows), "batch incomplete; preserved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--host-prepare")
    parser.add_argument("--host-sha")
    parser.add_argument("--contract")
    parser.add_argument("--sha256")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        value = material(args.host_prepare, args.host_sha)
        path = exclusive(args.output, "_PREP_")
        print(json.dumps({"path": str(path), "sha256": host.write_new(path, value)}))
    else:
        execute(args.contract, args.sha256, args.output)


if __name__ == "__main__":
    main()
