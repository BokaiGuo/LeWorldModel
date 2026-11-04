"""New-venv P3 host wiring; prepare/verify are CPU-only, run requires coordination."""
import argparse
import json
import os
from pathlib import Path
import shutil
import traceback

from . import p3_venv_migration_binding_20261102_codexexclusive as binding
from . import p3_live_pair_host_20261102_codexexclusive as pair
from . import moving_history_planners_cuda_20261027_v1 as tasks_loader

host, live, saved, ROOT = pair.host, pair.live, pair.saved, pair.ROOT
NAME = "p3_venv_live_host_20261102_codexexclusive"
AUTHORITY = "NEW_VENV_KNOWN_ROOT_P3_HOST_CUDA_COORDINATED"
PREDECESSOR_SHA = "47cc8ff5a32ef3ce36b0dd6807b7c1ff68fea3736137d0e948122d050b2e2846"


def require(value, message):
    if not value:
        raise ValueError(message)


def material(binding_path, binding_sha):
    """Reconstruct authenticated saved inputs; no neural/physical operations."""
    binding.verify(binding_path, binding_sha)
    env = json.loads(Path(binding_path).read_text())
    require(host.prior.sha(host.PREVIOUS) == PREDECESSOR_SHA, "predecessor SHA")
    predecessor = host.prior.prior.read(host.PREVIOUS)
    require(len(predecessor["slots"]) == 2, "two fixed ordered roots")
    tasks = []
    for index, slot in enumerate(predecessor["slots"]):
        require(slot["source_slot"] == index, "task slot order")
        loaded = tasks_loader.load_export(slot["task_export"], slot["task_sha256"])
        require(loaded["task_id"] == slot["task_id"]
                and loaded["gn_task"]["task_contract_sha256"] == slot["task_contract_sha256"],
                "task identity")
        tasks.append(loaded["gn_task"])
    banks = live.kernel.banks(live.kernel.native.BALL)
    old = host.prior.prior.read(binding.OLD)
    require([host.prior.digest(x) for x in tasks] == old["task_sha256"], "task digest")
    require([host.prior.digest(x) for x in banks] == old["bank_sha256"], "bank digest")
    require(predecessor["expected_model_hash"] == old["expected_model_hash"]
            == env["historical_expected_tensor_hash"], "declared tensor binding")
    host.identity.verify(old["sources"])
    sources = {row["path"]: row for row in old["sources"]}
    for row in env["selected_sources"] + [p["entry"] for p in env["packages"]] + env["checkpoint"]:
        sources[row["path"]] = row
    for row in host.identity.inventory([Path(__file__).resolve(), Path(binding.__file__).resolve(),
                                        Path(tasks_loader.__file__).resolve(), Path(binding_path).resolve()]):
        sources[row["path"]] = row
    host_plan = dict(old, sources=sorted(sources.values(), key=lambda row: row["path"]))
    plan = {"schema": NAME + ".prepare.v1", "point": 20, "arms": ["signal", "reverse"],
            "new_environment_binding": str(Path(binding_path).resolve()),
            "new_environment_binding_sha256": binding_sha,
            "old_new_environment_equivalence": "unknown", "historical_prepare_sha256": binding.OLD_SHA,
            "predecessor_sha256": PREDECESSOR_SHA, "host_plan": host_plan,
            "task_sha256": [host.prior.digest(x) for x in tasks],
            "bank_sha256": [host.prior.digest(x) for x in banks],
            "known_roots": True, "source_closure_complete": False,
            "physical_release": False, "independent_outcome": False,
            "equal_total_compute": False, "P3_admission": False}
    return plan, tasks, banks


def verify(path, sha):
    require(host.prior.sha(path) == sha, "external prepare SHA")
    plan = host.prior.prior.read(path)
    current, tasks, banks = material(plan["new_environment_binding"], plan["new_environment_binding_sha256"])
    require(plan == current, "prepare drift")
    return plan, tasks, banks


def output(path, prefix):
    path = Path(path)
    require(path.is_absolute() and path.parent == ROOT / "results"
            and path.name.startswith(NAME + prefix) and not os.path.lexists(path), "exclusive output")
    return path


def launch(prepared, sha, destination, authority):
    require(authority == AUTHORITY, "explicit coordinated CUDA authority required")
    plan, tasks, banks = verify(prepared, sha)
    out = output(destination, "_RUN_")
    require(shutil.disk_usage(out.parent).free >= host.CAP + host.FLOOR, "disk admission")
    with host.lock.gpu_lock(host.LOCK):
        snapshot = host.gpu_snapshot()
        out.mkdir(mode=0o700)
        host.write_new(out / "prepare.json", plan["host_plan"])
        host.write_new(out / "venv_prepare.json", plan)
        pid = os.fork()
        if pid == 0:
            try:
                fd = next(int(x) for x in os.listdir("/proc/self/fd")
                          if x.isdigit() and int(x) > 2 and host._same_lock(int(x)))
                def execute(model, device, values, sink, limits, sync):
                    return live.execute_host(model, device, values, sink, limits, sync, point=20)
                host.bridge.execute_host = execute
                report = host.child(plan["host_plan"], tasks, banks, out, fd, snapshot)
                os._exit(0 if report["status"] == "panel_completed_unverified" else 1)
            except BaseException:
                traceback.print_exc()
                os._exit(1)
        _, status = os.waitpid(pid, 0)
    require(os.waitstatus_to_exitcode(status) == 0, "child failed; retain run")
    return {"run": str(out), "report_sha256": host.prior.sha(out / "report.json")}


def consume(run_path, report_sha, prepare_sha):
    out = Path(run_path)
    require(out.is_absolute() and out.parent == ROOT / "results" and not out.is_symlink()
            and out.name.startswith(NAME + "_RUN_") and out.is_dir(), "exclusive run")
    require(host.prior.sha(out / "report.json") == report_sha, "report SHA")
    plan, tasks, banks = verify(out / "venv_prepare.json", prepare_sha)
    report = host.prior.prior.read(out / "report.json")
    require(report["status"] == "panel_completed_unverified"
            and report["prepare_sha256"] == host.prior.sha(out / "prepare.json")
            and host.prior.prior.read(out / "prepare.json") == plan["host_plan"]
            and report["model_before"] == report["model_after"] == plan["host_plan"]["expected_model_hash"]
            and report["gpu_snapshot"]["uuid"] == host.UUID, "host binding")
    result = saved.consume(out / "events.jsonl", report["journal_tail"], host.CAP, tasks, banks)
    result.update(old_new_environment_equivalence="unknown", independent_outcome=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "verify", "run", "consume"))
    parser.add_argument("--binding")
    parser.add_argument("--binding-sha")
    parser.add_argument("--prepare")
    parser.add_argument("--sha256")
    parser.add_argument("--output")
    parser.add_argument("--authority")
    parser.add_argument("--run")
    parser.add_argument("--report-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        plan, _, _ = material(args.binding, args.binding_sha)
        path = output(args.output, "_PREP_")
        print(json.dumps({"path": str(path), "sha256": host.write_new(path, plan),
                          "execution_authorized": False}))
    elif args.action == "verify":
        plan, tasks, banks = verify(args.prepare, args.sha256)
        print(json.dumps({"status": "new_venv_preflight_verified", "tasks": len(tasks),
                          "banks": len(banks), "execution_authorized": False,
                          "old_new_environment_equivalence": plan["old_new_environment_equivalence"]}))
    elif args.action == "run":
        print(json.dumps(launch(args.prepare, args.sha256, args.output, args.authority)))
    else:
        result = consume(args.run, args.report_sha, args.sha256)
        path = output(args.output, "_CONSUME_")
        print(json.dumps({"path": str(path), "sha256": host.write_new(path, result)}))


if __name__ == "__main__":
    main()
