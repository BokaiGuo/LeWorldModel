"""Two-root physical bridge using frozen model handoffs and historical guards."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from . import p2_multi_joint_development_20261102_codexexclusive as model
from . import p2_multi_joint_saved_20261102_codexexclusive as model_saved
from . import p2_joint_physical_20261102_codexexclusive as one
from . import p2_joint_physical_saved_20261102_codexexclusive as one_saved

NAME = "p2_multi_joint_physical_20261102_codexexclusive"
ROOT, RESULTS = model.ROOT, model.RESULTS
physical = one.physical
HISTORY = one.HISTORY
SESSION = RESULTS / (NAME + "_SESSION_01")
AUTH = "P2_TWO_ROOT_FIXED_WINNERS_PHYSICS_ONLY"


def build_recipe(case, handoff, model_report_sha):
    old = model.read_checked(case["old_physical_prep"], case["old_physical_prep_sha256"])
    model.need(old["job_index"] == case["base"] and
               physical.digest(old["prefix_reference"]) == case["prefix_reference_sha256"]
               and handoff["base"] == case["base"] and handoff["root"] == case["root"]
               and handoff["task_id"] == case["task_id"]
               and handoff["native_root_sha256"] == case["native_root_sha256"]
               and handoff["context_sha256"] == case["context_sha256"]
               and handoff["goal_sha256"] == case["goal_sha256"], "physical source/handoff")
    template, plans = old["plans"][0], []
    model.need([a["arm"] for a in handoff["arms"]] == list(model.matched.ARMS), "arm order")
    for index, selected in enumerate(handoff["arms"]):
        raw = bytes.fromhex(selected["full25_float32_hex"])
        model.need(len(raw) == 200 and raw[:120].hex() == case["prefix_float32_hex"]
                   and model.host.base.sha(raw) == selected["full25_sha256"]
                   and model.host.base.sha(raw[120:]) == selected["suffix_sha256"], "fixed actions")
        physical.raw25(np.frombuffer(raw, dtype="<f4").reshape(25, 2).astype(float).tolist())
        witness = {"schema": NAME + ".winner_reference_v1", "arm": selected["arm"],
                   "model_report_sha256": model_report_sha, "candidate_index": selected["candidate_index"],
                   "iteration": selected["iteration"], "model_score": selected["model_score"],
                   "suffix_sha256": selected["suffix_sha256"], "input_context": deepcopy(old["observed_context"]),
                   "scope": "new optimized winner reference, not historical model witness"}
        witness["new_witness_sha256"] = physical.digest(witness)
        p = deepcopy(template)
        p.update(index=index, policy=selected["arm"],
                 commands=np.frombuffer(raw, dtype="<f4").reshape(25, 2).astype(float).tolist(),
                 raw_float32_hex=raw.hex(), raw_float32_sha256=selected["full25_sha256"],
                 new_model_plan=witness)
        p.pop("selection_digest")
        p["selection_digest"] = physical.digest(p)
        plans.append(p)
    old["plans"] = plans
    return old


def build(model_run, report_sha):
    receipt = model_saved.consume(model_run, report_sha)
    cases, _ = model.input_cases()
    recipes = []
    for case, row in zip(cases, receipt["roots"], strict=True):
        path = Path(model_run) / f"root_{case['base']}_handoff.json"
        handoff = model.read_checked(path, row["handoff_sha256"])
        recipes.append({"case": case, "handoff_sha256": row["handoff_sha256"],
                        "prepared": build_recipe(case, handoff, report_sha)})
    sources = {}
    for path in (Path(__file__), Path(one.__file__), Path(one_saved.__file__),
                 Path(model.__file__), Path(model_saved.__file__),
                 Path(__file__).parent / "p2_multi_joint_physical_saved_20261102_codexexclusive.py"):
        sources[str(path)] = model.host.base.file_sha(path)
    model.need(model.host.base.file_sha(HISTORY) == one.isolation.recovery.HISTORICAL_SHA, "historical overlay")
    return {"schema": NAME + ".plan_v1", "model_run": str(model_run), "model_report_sha256": report_sha,
            "model_plan_sha256": receipt["plan_sha256"], "recipes": recipes, "sources": sources,
            "historical_p0_sha256": one.isolation.recovery.HISTORICAL_SHA,
            "physical_caps_per_root_pair": physical.contract()["caps"], "root_wall_seconds": physical.WALL,
            "output_cap_per_root": physical.CAP, "session": str(SESSION), "authorization": AUTH,
            "protocol": "same root and task per pair; warm10 + guarded prefix15 + winner10; native stop censored",
            "failure": "native stop continues; engineering error stops remaining roots; no retries or padding",
            "historical_sra_authenticated": False, "equal_total_compute": False,
            "independent_confirmation": False, "P2_admission": False}


def path_new(path, kind):
    path = Path(path)
    model.need(path.is_absolute() and path.parent == RESULTS and path.parent.resolve() == RESULTS
               and path.name.startswith(NAME + kind) and not os.path.lexists(path), "exclusive path")
    return path


def verify(path, digest):
    path = Path(path)
    model.need(path.is_absolute() and path.parent == RESULTS and not path.is_symlink()
               and path.name.startswith(NAME + "_PLAN_"), "plan path")
    plan = model.read_checked(path, digest)
    model.need(plan == build(plan["model_run"], plan["model_report_sha256"])
               and model.host.base.file_sha(path) == digest, "physical plan/source drift")
    return plan


def execute_one(recipe, data, store, plan_sha):
    return one.execute_pair(recipe["prepared"], data, store, lambda: make_env(), plan_sha)


def make_env():
    import gymnasium as gym
    import stable_worldmodel.envs
    return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                    with_target=False, render_action=False)


def child(plan_path, digest, auth):
    model.need(auth == AUTH, "physical authorization")
    plan = verify(plan_path, digest)
    target = str(one.isolation.recovery.TARGET)
    model.need(model.host.base.file_sha(target) == one.isolation.recovery.HISTORICAL_SHA and
               any(len(parts := line.split()) > 6 and parts[4] == target and "ro" in parts[5].split(",")
                   for line in Path("/proc/self/mountinfo").read_text().splitlines()), "historical read-only mount")
    model.need(SESSION.is_dir() and not SESSION.is_symlink(), "exclusive session")
    status = [{"base": r["case"]["base"], "status": "not_started"} for r in plan["recipes"]]
    session = {"schema": NAME + ".session_v1", "plan_sha256": digest, "rows": status,
               "status": "partial_or_failed", "error": None, "physical_execution": True,
               "equal_total_compute": False, "independent_confirmation": False, "P2_admission": False}
    started = time.monotonic()
    try:
        with model.host.native.locking.gpu_lock(one.executor.LOCK):
            for index, recipe in enumerate(plan["recipes"]):
                case = recipe["case"]
                old, data = physical.verify(case["old_physical_prep"], case["old_physical_prep_sha256"])
                prepared = recipe["prepared"]
                model.need({k: v for k, v in prepared.items() if k != "plans"} ==
                           {k: v for k, v in old.items() if k != "plans"}
                           and prepared["source_slots"] == [r["slot"] for r in data["roots"]]
                           and prepared["source_roots"] == physical.native.allmode.root_metadata(data)
                           and prepared["warm_traces"] == [r["warm_trace"] for r in data["roots"]],
                           "physical source reconstruction")
                path = SESSION / (physical.NAME + f"_MULTI_B{case['base']}_01")
                store = physical.Store(path)
                status[index]["status"] = "running"
                store.put("binding.json", {"plan_sha256": digest, "base": case["base"],
                                           "old_plan_sha256": case["old_physical_prep_sha256"]})
                rows, meter, errors = execute_one(recipe, data, store, digest)
                try:
                    physical.verify(case["old_physical_prep"], case["old_physical_prep_sha256"])
                    verify(plan_path, digest)
                    events = [physical.native.decode(x) for x in (path / "journal.jsonl").read_bytes().splitlines()]
                    one.executor._audit_recorded(prepared, rows, events)
                    if time.monotonic() - meter.started >= physical.WALL:
                        raise TimeoutError("physical root wall boundary")
                except BaseException as exc:
                    errors.append(physical.native.base.error("postcheck", exc))
                result_status = "recorded_development_unconsumed" if not errors else "partial_or_failed"
                store.finalizing = True
                store.emit("joint_finished", status=result_status, errors=errors, counts=meter.counts)
                report = {"schema": NAME + ".root_report_v1", "status": result_status, "rows": rows,
                          "errors": errors, "counts": meter.counts, "operation_receipts": meter.receipts,
                          "plan_sha256": digest, "journal_sha256": physical.sha(path / "journal.jsonl"),
                          "wall_seconds": time.monotonic() - meter.started, "consumer_verified": False,
                          "new_neural_calls": 0, "historical_sra_authenticated": False,
                          "independent_confirmation": False, "equal_total_compute": False,
                          "P1_admission": False, "P2_admission": False}
                store.put("report.json", report, final=True)
                status[index].update(status=result_status, output=str(path),
                                     report_sha256=physical.sha(path / "report.json"))
                if errors:
                    raise RuntimeError("root physical failure; remaining roots not started")
        verify(plan_path, digest)
        session["status"] = "recorded_development_unconsumed"
    except Exception as exc:
        session["error"] = {"type": type(exc).__name__, "message": str(exc)[:1500]}
        for row in status:
            if row["status"] == "running":
                row["status"] = "failed_partial"
    session["wall_seconds"] = time.monotonic() - started
    sha = model.host.base.write_new(SESSION / "session_report.json", session)
    print(json.dumps({"status": session["status"], "session_report_sha256": sha}))
    if session["error"]:
        raise SystemExit(1)


def run_host(path, digest, auth):
    model.need(auth == AUTH, "physical authorization")
    verify(path, digest)
    SESSION.mkdir(mode=0o700)
    physical.paired.fsync_dir(SESSION.parent)
    cmd = ["bwrap", "--die-with-parent", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
           "--ro-bind", str(HISTORY), str(one.isolation.recovery.TARGET), "--bind", str(SESSION), str(SESSION),
           "--bind", str(one.executor.LOCK), str(one.executor.LOCK), "--setenv", "TMPDIR", str(SESSION),
           "--setenv", "CUDA_VISIBLE_DEVICES", "", "--setenv", "OMP_NUM_THREADS", "1",
           "--setenv", "OPENBLAS_NUM_THREADS", "1", "--chdir", str(ROOT), "--",
           str(ROOT / ".venv/bin/python"), "-B", "-m", "experiments.spectral_reachability." + NAME,
           "run-child", "--plan", str(path), "--plan-sha256", digest, "--authorization", auth]
    result = subprocess.run(cmd, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--model-run", required=True)
    prep.add_argument("--model-report-sha256", required=True)
    prep.add_argument("--output", required=True)
    for name in ("check", "run-host", "run-child"):
        q = sub.add_parser(name)
        q.add_argument("--plan", required=True)
        q.add_argument("--plan-sha256", required=True)
        if name != "check":
            q.add_argument("--authorization", required=True)
    a = p.parse_args()
    if a.command == "prepare":
        path = path_new(a.output, "_PLAN_")
        print(json.dumps({"plan": str(path), "sha256": model.host.base.write_new(
            path, build(a.model_run, a.model_report_sha256))}))
    elif a.command == "check":
        verify(a.plan, a.plan_sha256)
        print(json.dumps({"status": "verified_no_physics"}))
    elif a.command == "run-host":
        run_host(a.plan, a.plan_sha256, a.authorization)
    else:
        child(a.plan, a.plan_sha256, a.authorization)


if __name__ == "__main__":
    main()
