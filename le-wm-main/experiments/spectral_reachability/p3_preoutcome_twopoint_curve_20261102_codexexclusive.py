"""Known-root two-point P3 curve: reuse saved 20, prepare new 30; no physics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import traceback

from . import p3_preoutcome_cuda_host_20261102_codexexclusive as base
from . import p3_preoutcome_saved_consumer_20261102_codexexclusive as saved
from . import codex_p3_quota_journal_20261030_v1 as host_journal

ctl, host, ROOT = base.control, base.host, base.ROOT
NAME = "p3_preoutcome_twopoint_curve_20261102_codexexclusive"
AUTH = "P3_PREOUTCOME_KNOWN_ROOT_POINT30_HOST_CUDA_COORDINATED"
POINT20 = ROOT / "results/p3_preoutcome_cuda_host_20261102_codexexclusive_RUN_01"
POINT20_PREP = "f5f3c8caa423a413f972967f5a86c0cc9ce1e615c71139d51e74ee852ba6a11a"
POINT20_REPORT = "202ffc7a6c8fb7f26301a137028862a08cd4e014ca95c4387220b4adc49d8729"
POINT20_CONSUME = ROOT / "results/p3_preoutcome_cuda_host_20261102_codexexclusive_CONSUME_01.json"
POINT20_CONSUME_SHA = "85b16d3875cf360059fd166d6f961588dea94c4988d8e725ee8ce07380f2bed7"
CAP30 = {"observation_encode": 0, "action_encode": 64, "predict_calls": 64,
         "predict_samples": 64, "signal_compute": 2, "candidate_evaluate": 30}


def build():
    old20, tasks, bank = base.verify(POINT20 / "preoutcome_prepare.json", POINT20_PREP)
    prior = saved.consume(POINT20, POINT20_REPORT, POINT20_PREP, base.verify)
    ctl.require(host.prior.sha(POINT20_CONSUME) == POINT20_CONSUME_SHA
                and json.loads(POINT20_CONSUME.read_bytes()) == prior, "point20 saved consume")
    source = dict(old20["host_plan"])
    inventory = {v["path"]: v for v in source["sources"]}
    for item in host.identity.inventory([Path(__file__), Path(base.__file__), Path(saved.__file__),
                                         Path(ctl.__file__)]):
        inventory[item["path"]] = item
    source["sources"] = sorted(inventory.values(), key=lambda x: x["path"])
    schedule30 = ctl.freeze(30, CAP30, bank, old20["schedule"]["binding"])
    ctl.require(schedule30["bank_sha256"] == old20["schedule"]["bank_sha256"], "shared fixed bank")
    return ({"schema": NAME + ".prepare.v1", "host_plan": source,
             "points": [20, 30], "schedule20": old20["schedule"], "schedule30": schedule30,
             "point20_report_sha256": POINT20_REPORT, "point20_prepare_sha256": POINT20_PREP,
             "point20_consumer_sha256": POINT20_CONSUME_SHA,
             "point20_scope": "exposed historical run reused; not rerun",
             "point30_scope": "new known-root development run on same fixed candidate bank",
             "curve_contract_created_after_point20_outcome": True,
             "common_per_arm_cap_within_each_point": True, "equal_total_compute": False,
             "independent_roots": False, "physical_release": False,
             "P3_admission": False, "C3_admission": False}, tasks, bank)


def verify(path, sha):
    ctl.require(host.prior.sha(path) == sha, "external prepare SHA")
    value = json.loads(Path(path).read_bytes())
    current, tasks, bank = build()
    ctl.require(value == current, "source or prepare drift")
    return value, tasks, bank


def output(path, prefix):
    value = Path(path)
    ctl.require(value.is_absolute() and value.parent == ROOT / "results"
                and value.name.startswith(NAME + prefix) and not os.path.lexists(value), "exclusive output")
    return value


def launch(prepared, sha, destination, authority):
    ctl.require(authority == AUTH, "explicit coordinated CUDA authority")
    plan, tasks, bank = verify(prepared, sha)
    out = output(destination, "_RUN_")
    ctl.require(shutil.disk_usage(out.parent).free >= host.CAP + host.FLOOR, "disk admission")
    with host.lock.gpu_lock(host.LOCK):
        snapshot = host.gpu_snapshot()
        out.mkdir(mode=0o700)
        host.write_new(out / "prepare.json", plan["host_plan"])
        host.write_new(out / "curve_prepare.json", plan)
        pid = os.fork()
        if pid == 0:
            try:
                fd = next(int(x) for x in os.listdir("/proc/self/fd")
                          if x.isdigit() and int(x) > 2 and host._same_lock(int(x)))
                def execute(model, device, values, sink, limits, sync):
                    signal, score = base.callbacks(model, device, values)
                    def checked_sync():
                        limits()
                        sync()
                        limits()
                    result = ctl.run_pair(plan["schedule30"], bank, {a: model for a in ctl.ARMS},
                                          {a: checked_sync for a in ctl.ARMS},
                                          signal, score, out / "controller.jsonl")
                    sink({"kind": "curve30_completed", "controller_sha256": host.prior.sha(out / "controller.jsonl"),
                          "result": result})
                    return result
                host.bridge.execute_host = execute
                report = host.child(plan["host_plan"], tasks, bank, out, fd, snapshot)
                os._exit(0 if report["status"] == "panel_completed_unverified" else 1)
            except BaseException:
                traceback.print_exc()
                os._exit(1)
        _, status = os.waitpid(pid, 0)
    ctl.require(os.waitstatus_to_exitcode(status) == 0, "child failed; retain run and costs")
    return {"run": str(out), "report_sha256": host.prior.sha(out / "report.json")}


def assemble(plan, checked20, events20, checked30=None, events30=None, failure=None):
    """Describe full arm costs; never substitute unknown/failed work with zero."""
    def point(number, checked, events, scope):
        if checked is None:
            return {"point": number, "status": "missing_or_failed", "failure": failure,
                    "resources": "unknown_in_full", "winners": None, "scope": scope}
        actual = events[-1]["result"]
        return {"point": number, "status": "saved_consistency_checked", "scope": scope,
                "allocation": checked["allocation"], "winners": checked["winners"],
                "signals_recorded": checked["signals_recorded"],
                "resources": actual["resources"], "cost_units_not_fused": True}
    p20 = point(20, checked20, events20, "previously exposed saved model run")
    p30 = point(30, checked30, events30, "new development execution")
    signal_equal = None if checked30 is None else checked20["signals_recorded"] == checked30["signals_recorded"]
    return {"schema": NAME + ".curve.v1", "points": [p20, p30],
            "status": "both_saved_consistency_checked" if checked30 is not None else "partial_curve",
            "signal_values_identical_across_points": signal_equal,
            "same_fixed_bank_sha256": plan["schedule20"]["bank_sha256"],
            "root_scope": "same two previously exposed development roots, not independent",
            "curve_contract_created_after_point20_outcome": True, "physical_outcomes": "not_measured",
            "wall_by_point": "unknown: host elapsed available separately in individual reports",
            "equal_total_compute": False, "flops": "unknown", "energy": "unknown",
            "P3_admission": False, "C3_admission": False}


def consume(run, report_sha, prepared, prepare_sha):
    plan, _, bank = verify(prepared, prepare_sha)
    checked20 = saved.consume(POINT20, POINT20_REPORT, POINT20_PREP, base.verify)
    events20 = saved.read(POINT20 / "controller.jsonl")
    run = Path(run)
    ctl.require(run.is_absolute() and run.parent == ROOT / "results"
                and run.name.startswith(NAME + "_RUN_") and run.is_dir() and not run.is_symlink(), "run path")
    ctl.require(host.prior.sha(run / "report.json") == report_sha, "external report SHA")
    ctl.require(host.prior.sha(run / "curve_prepare.json") == prepare_sha
                and json.loads((run / "curve_prepare.json").read_bytes()) == plan, "run prepare")
    report = json.loads((run / "report.json").read_bytes())
    ctl.require(report["prepare_sha256"] == host.prior.sha(run / "prepare.json")
                and json.loads((run / "prepare.json").read_bytes()) == plan["host_plan"], "host prepare")
    if report["status"] != "panel_completed_unverified":
        failure = {"report_status": report["status"], "error": report.get("error"),
                   "controller_journal_present": (run / "controller.jsonl").is_file(),
                   "unobserved_work": "unknown"}
        if (run / "controller.jsonl").is_file():
            try:
                prefix = saved.read(run / "controller.jsonl")
                failure["verified_prefix_events"] = len(prefix)
                failure["terminal_event"] = prefix[-1]
            except Exception as exc:
                failure["controller_error"] = str(exc)
        result = assemble(plan, checked20, events20, failure=failure)
    else:
        ctl.require(report["model_before"] == report["model_after"] == plan["host_plan"]["expected_model_hash"]
                    and report["gpu_snapshot"]["uuid"] == host.UUID, "host identity")
        events = host_journal.read(run / "events.jsonl", report["journal_tail"], host.CAP)
        ctl.require(len(events) == 1 and events[0]["kind"] == "curve30_completed"
                    and events[0]["controller_sha256"] == host.prior.sha(run / "controller.jsonl"), "host journal")
        controller_events = saved.read(run / "controller.jsonl")
        checked30 = saved.verify_events(controller_events, plan["schedule30"], bank)
        ctl.require(events[0]["result"] == controller_events[-1]["result"], "host result")
        result = assemble(plan, checked20, events20, checked30, controller_events)
    result["point30_report_sha256"] = report_sha
    result["prepare_sha256"] = prepare_sha
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "verify", "preview", "run", "consume"))
    p.add_argument("--output")
    p.add_argument("--prepare")
    p.add_argument("--sha256")
    p.add_argument("--run")
    p.add_argument("--report-sha256")
    p.add_argument("--authority")
    a = p.parse_args()
    if a.action == "prepare":
        plan, _, _ = build()
        path = output(a.output, "_PREP_")
        print(json.dumps({"path": str(path), "sha256": host.write_new(path, plan), "cuda_used": False}))
    elif a.action == "verify":
        plan, tasks, bank = verify(a.prepare, a.sha256)
        print(json.dumps({"status": "prepared_only", "points": plan["points"], "tasks": len(tasks),
                          "banks": len(bank), "execution_authorized": False}))
    elif a.action == "preview":
        plan, _, _ = verify(a.prepare, a.sha256)
        checked = saved.consume(POINT20, POINT20_REPORT, POINT20_PREP, base.verify)
        result = assemble(plan, checked, saved.read(POINT20 / "controller.jsonl"),
                          failure={"reason": "point30_not_started", "unobserved_work": "unknown"})
        result["prepare_sha256"] = a.sha256
        path = output(a.output, "_PREVIEW_")
        print(json.dumps({"path": str(path), "sha256": host.write_new(path, result),
                          "status": result["status"]}))
    elif a.action == "run":
        print(json.dumps(launch(a.prepare, a.sha256, a.output, a.authority)))
    else:
        result = consume(a.run, a.report_sha256, a.prepare, a.sha256)
        path = output(a.output, "_CONSUME_")
        print(json.dumps({"path": str(path), "sha256": host.write_new(path, result),
                          "status": result["status"]}))


if __name__ == "__main__":
    main()
