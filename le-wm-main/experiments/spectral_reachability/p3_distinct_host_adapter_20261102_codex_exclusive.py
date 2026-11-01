"""P3 fixed-quota admission around the existing host-CUDA runner.

The existing runner performs every real model operation and resource charge.
This adapter constrains its emitted allocation and candidate order; it does not
run the separate three-axis BoundaryAdapter or add fictitious model calls.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import traceback

from . import p3_distinct_quota_scheduler_20261102_codex_exclusive as scheduler
from . import p3_effective_equal_budget_20261101_parallel as existing

host = existing.host
ROOT = host.ROOT
NAME = "p3_distinct_host_adapter_20261102_codex_exclusive"
AUTHORITY = "EXPLICIT_P3_DISTINCT_HOST_CUDA_REPEATED_KNOWN_PANEL"
PREVIOUS_REPORT = existing.PRIMARY_REPORT_SHA256


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular source file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QuotaMonitor:
    """Admission check over the real runner's event stream, before next dispatch."""

    def __init__(self, sink):
        require(callable(sink), "durable sink")
        self.sink = sink
        self.contract = {row["budget_point"]: row for row in scheduler.protocol()["allocations"]}
        self.roster = [(point, arm) for point in (20, 30) for arm in ("uniform", "signal", "reverse")]
        self.position = 0
        self.active = None
        self.attempts = [0, 0]
        self.completed = [0, 0]
        self.pending = None
        self.candidates = [[], []]
        self.closed = False

    def __call__(self, event):
        # Save the actual observed event first. A rejected allocation is retained.
        self.sink(event)
        kind = event.get("kind")
        if kind == "allocation_frozen":
            require(not self.closed and self.active is None and self.position < len(self.roster),
                    "allocation lifecycle")
            point, arm = self.roster[self.position]
            require(event["scope"] == [point, arm, "search", 0], "fixed arm order")
            expected = ([(point + 4) // 2] * 2 if arm == "uniform"
                        else self.contract[point][arm])
            require(event["quota"] == expected, "online allocation differs from frozen strategy")
            self.active = (point, arm, expected)
            self.attempts, self.completed, self.candidates = [0, 0], [0, 0], [[], []]
        elif kind == "candidate_attempt":
            require(self.active is not None and self.pending is None, "candidate lifecycle")
            point, arm, quota = self.active
            scope = event["scope"]
            root = scope[3]
            require(type(root) is int and root in (0, 1)
                    and scope == [point, arm, "search", root]
                    and (root == 0 or self.completed[0] == quota[0])
                    and event["index"] == self.attempts[root] < quota[root], "candidate order/quota")
            self.attempts[root] += 1
            self.pending = (root, event["index"], copy.deepcopy(event["commands"]))
        elif kind == "candidate_complete":
            require(self.active is not None and self.pending is not None, "candidate completion lifecycle")
            root, index, commands = self.pending
            point, arm, _ = self.active
            candidate = event["candidate"]
            require(event["scope"] == [point, arm, "search", root]
                    and candidate["index"] == index and candidate["commands"] == commands
                    and type(candidate["cost"]) in (int, float)
                    and math.isfinite(candidate["cost"]), "candidate completion identity")
            self.completed[root] += 1
            self.candidates[root].append(candidate)
            self.pending = None
        elif kind == "arm_complete":
            require(self.active is not None and self.pending is None, "arm completion lifecycle")
            point, arm, quota = self.active
            row = event["row"]
            require(self.attempts == self.completed == quota
                    and row["point"] == point and row["arm"] == arm
                    and row["allocation"] == quota
                    and row["attempted"] == row["returned"]
                    and row["winners"] == [min(items, key=lambda x: (x["cost"], x["index"]))
                                           for items in self.candidates],
                    "arm completion, real accounting or winner mismatch")
            self.position += 1
            self.active = None
        elif kind == "experiment_complete":
            require(self.position == len(self.roster) and self.active is None
                    and self.pending is None and event["arms"] == len(self.roster),
                    "complete roster")
            self.closed = True


def check_events(events):
    monitor = QuotaMonitor(lambda _event: None)
    for event in events:
        monitor(event)
    require(monitor.closed, "incomplete observed panel")
    return {"status": "quota_monitor_complete", "P3_admission": False, "C3_admission": False}


def build():
    frozen = scheduler.protocol()
    plan, tasks, banks = host.material()
    require(plan["budget_points"] == [20, 30] and plan["roster"] == ["uniform", "signal", "reverse"]
            and plan["gpu_uuid"] == host.UUID and plan["scientific_pass"] is False
            and plan["physical_release"] is False, "existing host plan")
    # Prove the prior panel exists and its saved semantics reconstruct, before
    # considering any repeat. No claim that the new adapter has run.
    report, events, _ = existing.verify_primary_run(existing.PRIMARY_RUN, PREVIOUS_REPORT)
    check_events(events)
    require(report["status"] == "panel_completed_unverified", "previous completed run")
    return {"schema": NAME + ".prepare.v1", "plan": plan, "scheduler": frozen,
            "adapter_sha256": sha(__file__), "scheduler_sha256": sha(scheduler.__file__),
            "historical_report_sha256": PREVIOUS_REPORT,
            "duplicate_known_panel": True, "candidate_selection": "existing native signal and runner",
            "measurement": "existing runner resource attempts/returns, hooks and saved consumer",
            "physical_release": False, "scientific_pass": False, "P3_admission": False,
            "C3_admission": False, "gate_modified": False}


def launch(prepared, prepared_sha, output, *, authority, repeat_known_panel):
    require(authority == AUTHORITY and repeat_known_panel is True,
            "explicit authority and acknowledgement of repeated known panel")
    prepared = Path(prepared).resolve(strict=True)
    require(sha(prepared) == prepared_sha, "prepared SHA")
    contract = host.prior.prior.read(prepared)
    require(contract == build(), "prepared source drift")
    out = Path(output)
    require(out.is_absolute() and out.parent == ROOT / "results"
            and out.name.startswith(NAME + "_RUN_") and not os.path.lexists(out), "exclusive output")
    require(shutil.disk_usage(out.parent).free >= host.CAP + host.FLOOR, "disk admission")
    plan = contract["plan"]
    _, tasks, _ = host.prior.verify(host.PREVIOUS, host.PREVIOUS_SHA)
    banks = host.bridge.banks(host.prior.dependency().old.BALL)
    require(plan["task_sha256"] == [host.prior.digest(t) for t in tasks]
            and plan["bank_sha256"] == [host.prior.digest(b) for b in banks], "task/bank identity")
    # Cooperative lock is nonblocking; it is not a device-wide reservation.
    with host.lock.gpu_lock(host.LOCK):
        snapshot = host.gpu_snapshot()
        out.mkdir(mode=0o700)
        host.write_new(out / "prepare.json", plan)
        host.write_new(out / "adapter_binding.json", {
            "prepared": str(prepared), "prepared_sha256": prepared_sha,
            "adapter_sha256": sha(__file__), "P3_admission": False})
        pid = os.fork()
        if pid == 0:
            try:
                fd = next(int(x) for x in os.listdir("/proc/self/fd")
                          if x.isdigit() and int(x) > 2 and host._same_lock(int(x)))
                original = host.bridge.execute_host
                def adapted(model, device, task_values, sink, limits, sync):
                    return original(model, device, task_values, QuotaMonitor(sink), limits, sync)
                host.bridge.execute_host = adapted
                report = host.child(plan, tasks, banks, out, fd, snapshot)
                os._exit(0 if report["status"] == "panel_completed_unverified" else 1)
            except BaseException:
                traceback.print_exc()
                os._exit(1)
        _, status = os.waitpid(pid, 0)
        require(os.waitstatus_to_exitcode(status) == 0, "child failed; preserve exclusive run")
    report_sha = sha(out / "report.json")
    _report, events, _consumed = existing.verify_primary_run(out, report_sha)
    check_events(events)
    return {"run": str(out), "report_sha256": report_sha,
            "status": "saved_semantics_and_quota_verified_not_admitted",
            "P3_admission": False, "C3_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "verify-existing"))
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--prepared-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authority")
    parser.add_argument("--repeat-known-panel", action="store_true")
    args = parser.parse_args()
    if args.action == "verify-existing":
        _report, events, _ = existing.verify_primary_run(existing.PRIMARY_RUN, PREVIOUS_REPORT)
        print(json.dumps(check_events(events), sort_keys=True))
    elif args.action == "prepare":
        require(args.output and args.output.is_absolute() and args.output.parent == ROOT / "results"
                and args.output.name.startswith(NAME + "_PREP_") and not os.path.lexists(args.output)
                and not args.prepared and not args.authority and not args.repeat_known_panel,
                "exclusive prepare arguments")
        print(json.dumps({"prepared": str(args.output), "sha256": host.write_new(args.output, build())}))
    else:
        require(args.prepared and args.prepared_sha256 and args.output, "run arguments")
        print(json.dumps(launch(args.prepared, args.prepared_sha256, args.output,
                                authority=args.authority, repeat_known_panel=args.repeat_known_panel)))


if __name__ == "__main__":
    main()
