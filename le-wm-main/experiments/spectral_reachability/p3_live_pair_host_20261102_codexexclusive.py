"""Exclusive known-root P3 point-20 host-CUDA launcher and saved consumer."""
import argparse
import json
import os
from pathlib import Path
import shutil
import traceback

from . import codex_p3_online_budget_host_20261030_v1 as host
from . import p3_live_distinct_pair_20261102_codexexclusive as live
from . import p3_live_pair_saved_20261102_codexexclusive as saved

NAME = "p3_live_pair_host_20261102_codexexclusive"
AUTHORITY = "KNOWN_ROOT_P3_POINT20_HOST_CUDA_DEVELOPMENT"
ROOT = host.ROOT


def build():
    base, tasks, _ = host.material()
    banks = live.kernel.banks(live.kernel.native.BALL)
    require = host.require
    require(base["task_sha256"] == [host.prior.digest(task) for task in tasks]
            and base["bank_sha256"] == [host.prior.digest(bank) for bank in banks],
            "task/bank differs from existing host plan")
    paths = [Path(__file__).resolve(), Path(live.__file__).resolve(),
             Path(saved.__file__).resolve(), Path(live.kernel.__file__).resolve(),
             Path(live.quota_kernel.__file__).resolve()]
    inventory = {row["path"]: row for row in base["sources"]}
    for row in host.identity.inventory(paths):
        inventory[row["path"]] = row
    base["sources"] = sorted(inventory.values(), key=lambda x: x["path"])
    return {"schema": NAME + ".prepare.v1", "host_plan": base, "point": 20,
            "arms": ["signal", "reverse"], "known_roots": True,
            "equal_total_compute": False, "physical_release": False,
            "independent_outcome": False, "P3_admission": False}, tasks, banks


def output(path, prefix):
    p = Path(path)
    host.require(p.is_absolute() and p.parent == ROOT / "results"
                 and p.name.startswith(NAME + prefix) and not os.path.lexists(p),
                 "exclusive result path required")
    return p


def load_prepare(path, sha):
    host.require(host.prior.sha(path) == sha, "external prepare SHA")
    plan = host.prior.prior.read(path)
    host.identity.verify(plan["host_plan"]["sources"])
    current, tasks, banks = build()
    host.require(plan == current, "prepare drift")
    return plan, tasks, banks


def run(plan_path, sha, result_path, authority):
    host.require(authority == AUTHORITY, "explicit development run authority")
    plan, tasks, banks = load_prepare(plan_path, sha)
    out = output(result_path, "_RUN_")
    host.require(shutil.disk_usage(out.parent).free >= host.CAP + host.FLOOR, "disk admission")
    with host.lock.gpu_lock(host.LOCK):
        snapshot = host.gpu_snapshot()
        out.mkdir(mode=0o700)
        host.write_new(out / "prepare.json", plan["host_plan"])
        host.write_new(out / "pair_prepare.json", plan)
        pid = os.fork()
        if pid == 0:
            try:
                fd = next(int(x) for x in os.listdir("/proc/self/fd")
                          if x.isdigit() and int(x) > 2 and host._same_lock(int(x)))
                # Change only the child-local bridge reference; host.child owns
                # model hashing, UUID preflight, journal and memory/wall caps.
                def execute(model, device, values, sink, limits, sync):
                    return live.execute_host(model, device, values, sink, limits, sync, point=20)
                host.bridge.execute_host = execute
                report = host.child(plan["host_plan"], tasks, banks, out, fd, snapshot)
                os._exit(0 if report["status"] == "panel_completed_unverified" else 1)
            except BaseException:
                traceback.print_exc()
                os._exit(1)
        _, status = os.waitpid(pid, 0)
    host.require(os.waitstatus_to_exitcode(status) == 0, "child failed; preserve run directory")
    return {"run": str(out), "report_sha256": host.prior.sha(out / "report.json")}


def consume(run_path, report_sha, prepare_sha):
    out = Path(run_path)
    host.require(out.is_absolute() and out.parent == ROOT / "results"
                 and out.name.startswith(NAME + "_RUN_") and out.is_dir() and not out.is_symlink(),
                 "exclusive run directory")
    host.require(host.prior.sha(out / "report.json") == report_sha, "report SHA")
    plan, tasks, banks = load_prepare(out / "pair_prepare.json", prepare_sha)
    report = host.prior.prior.read(out / "report.json")
    host.require(report["status"] == "panel_completed_unverified"
                 and report["prepare_sha256"] == host.prior.sha(out / "prepare.json")
                 and host.prior.prior.read(out / "prepare.json") == plan["host_plan"]
                 and report["model_before"] == report["model_after"] == plan["host_plan"]["expected_model_hash"]
                 and report["gpu_snapshot"]["uuid"] == host.UUID, "host identity and status")
    result = saved.consume(out / "events.jsonl", report["journal_tail"], host.CAP, tasks, banks)
    result["run_report_sha256"] = report_sha
    result["pair_prepare_sha256"] = prepare_sha
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "consume"))
    parser.add_argument("--output")
    parser.add_argument("--prepare")
    parser.add_argument("--sha256")
    parser.add_argument("--report-sha256")
    parser.add_argument("--run")
    parser.add_argument("--authority")
    args = parser.parse_args()
    if args.action == "prepare":
        plan, _, _ = build()
        path = output(args.output, "_PREP_")
        print(json.dumps({"prepare": str(path), "sha256": host.write_new(path, plan)}))
    elif args.action == "run":
        print(json.dumps(run(args.prepare, args.sha256, args.output, args.authority)))
    else:
        result = consume(args.run, args.report_sha256, args.sha256)
        path = output(args.output, "_CONSUME_")
        print(json.dumps({"output": str(path), "sha256": host.write_new(path, result),
                          "status": result["status"]}))


if __name__ == "__main__":
    main()
