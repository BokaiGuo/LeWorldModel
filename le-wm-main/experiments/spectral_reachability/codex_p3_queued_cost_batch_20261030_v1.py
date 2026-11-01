"""Six independent host processes, durable outer elapsed-time and budget record."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.spectral_reachability import codex_p3_queued_cost_host_20261030_v1 as host

STEM = "codex_p3_queued_cost_batch_20261030_v1"
SLOTS = [(p, a) for p in (20, 30) for a in ("uniform", "signal", "reverse")]


def intervals(events, point, arm):
    times = [e.get("wall_mono_ns") for e in events]
    host.require(all(type(t) is int and t > 0 for t in times) and
                 all(a <= b for a, b in zip(times, times[1:])), "event monotonic clock")
    allocations = [e for e in events if e["kind"] == "allocation_frozen"]
    ends = [e for e in events if e["kind"] == "arm_complete"]
    nominal = [e for e in events if e["kind"] == "nominal_request"]
    signals = [e for e in events if e["kind"] == "signal_complete"]
    host.require(len(allocations) == len(ends) == 1 and
                 len(nominal) == len(signals) == (0 if arm == "uniform" else 2), "phase census")
    require_scope = lambda event: event["scope"][:2] == [point, arm]
    host.require(all(require_scope(e) for e in allocations + ends + nominal + signals), "phase identity")
    return dict(signal_seconds_including_journal=(0.0 if not nominal else
                (signals[-1]["wall_mono_ns"]-nominal[0]["wall_mono_ns"])/1e9),
                search_seconds_including_journal=(ends[0]["wall_mono_ns"]-
                allocations[0]["wall_mono_ns"])/1e9)


def check_caps(point, row, report):
    caps = {**{k: 2*point+8 for k in ("action_encoder", "predict", "predictor", "pred_proj")},
            "jacobian": 2, "vjp_directions": 384, "grad_api": 384,
            "raw_backward_hook": 384, "svd": 2, "metric": point+4}
    host.require(set(row["attempted"]) == set(caps) and
                 all(type(row["attempted"][k]) is int and 0 <= row["attempted"][k] <= v
                     for k, v in caps.items()), "axis caps")
    host.require(row["point"] == report["point"] == point and row["arm"] == report["arm"],
                 "slot report identity")
    host.require(report["elapsed_seconds"] < 1800 and
                 report["peak_allocated_bytes"] <= 3*1024**3 and
                 report["peak_reserved_bytes"] <= 4*1024**3, "host cap")
    return caps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = host.ROOT / "results"
    out = Path(args.output)
    host.require(out.is_absolute() and out.parent == root and out.name.startswith(STEM + "_RUN_") and
                 not os.path.lexists(out), "exclusive batch output")
    host.require(host.prior.sha(args.prepare) == args.sha256, "external prepare SHA")
    paths = [(root / (host.NAME + f"_RUN_P{p}_{a}_01"),
              root / (host.NAME + f"_CONSUME_P{p}_{a}_01.json")) for p, a in SLOTS]
    host.require(all(not os.path.lexists(path) for pair in paths for path in pair),
                 "all child outputs must be new")
    out.mkdir(mode=0o700)
    ledger = host.journal.Journal(out / "events.jsonl", 2*1024**2, 128*1024)
    rows, failure = [], None
    try:
        ledger.event(dict(kind="batch_started", slots=[list(s) for s in SLOTS],
                          prepare_sha256=args.sha256, scientific_pass=False))
        for (point, arm), (run, receipt) in zip(SLOTS, paths, strict=True):
            ledger.event(dict(kind="process_attempt", point=point, arm=arm, output=str(run),
                              queue_state="not_a_result"))
            cmd = [sys.executable, "-B", "-m", "experiments.spectral_reachability."+host.NAME,
                   "run", "--prepare", args.prepare, "--sha256", args.sha256,
                   "--point", str(point), "--arm", arm, "--output", str(run)]
            started = time.monotonic_ns()
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            elapsed = (time.monotonic_ns() - started)/1e9
            ledger.event(dict(kind="process_return", point=point, arm=arm, exit_code=proc.returncode,
                              process_wall_seconds=elapsed, stdout=proc.stdout[-4096:],
                              stderr=proc.stderr[-8192:]))
            host.require(proc.returncode == 0, "process failed; partial preserved; no retry")
            report_path = run / "report.json"
            report_sha = host.prior.sha(report_path)
            report = host.prior.prior.read(report_path)
            host.require(report["status"] == "panel_completed_unverified" and report["error"] is None and
                         report["model_before"] == report["model_after"], "model/terminal")
            cmd = [sys.executable, "-B", "-m", "experiments.spectral_reachability."+host.NAME,
                   "consume", "--run", str(run), "--sha256", report_sha, "--output", str(receipt)]
            verify = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            host.require(verify.returncode == 0, "independent consumer failed; no retry")
            saved = host.prior.prior.read(receipt)
            host.require(saved["status"] == "semantic_saved_checked_not_scientifically_admitted" and
                         len(saved["rows"]) == 1, "saved semantics")
            row = saved["rows"][0]
            caps = check_caps(point, row, report)
            events = host.journal.read(run / "events.jsonl", report["journal_tail"], host.CAP)
            timing = intervals(events, point, arm)
            result = dict(point=point, arm=arm, process_wall_seconds=elapsed,
                          lock_wait_seconds=report["lock_wait_seconds"],
                          child_wall_seconds=report["elapsed_seconds"],
                          cuda_preflight_seconds=report["cuda_preflight_seconds"],
                          model_load_seconds=report["model_load_seconds"],
                          model_load_process_io_delta=report["model_load_process_io_delta"],
                          algorithm_seconds_including_journal=report["algorithm_seconds_including_journal"],
                          peak_allocated_bytes=report["peak_allocated_bytes"],
                          peak_reserved_bytes=report["peak_reserved_bytes"], **timing,
                          caps=caps, actual=row["attempted"], allocation=row["allocation"],
                          report_sha256=report_sha, report_path=str(report_path),
                          consumer_sha256=host.prior.sha(receipt), consumer_path=str(receipt))
            rows.append(result)
            ledger.event(dict(kind="slot_checked", result=result))
    except BaseException as exc:
        failure = dict(type=type(exc).__name__, message=str(exc)[:2048])
        try:
            ledger.append(dict(kind="batch_failure", error=failure, completed=len(rows),
                               retry_authorized=False), terminal=True)
        except BaseException as err:
            failure["journal_failure"] = repr(err)
    try:
        if failure is None:
            ledger.append(dict(kind="batch_complete", slots=len(rows), scientific_pass=False), terminal=True)
        result = dict(schema=STEM+".report", status="measured_not_scientifically_admitted" if failure is None
                      else "partial_or_failed", prepare_sha256=args.sha256, rows=rows, error=failure,
                      journal_tail=ledger.tail, axes_nested_not_additive=True,
                      e2e_wall_equality_certified=False, standalone_gpu_energy="unknown",
                      standalone_flops="unknown", scientific_pass=False)
    finally:
        ledger.close()
    host.write_new(out / "report.json", result)
    print(result["status"], len(rows))
    host.require(failure is None, "batch failed; retained; no retry")


if __name__ == "__main__":
    main()
