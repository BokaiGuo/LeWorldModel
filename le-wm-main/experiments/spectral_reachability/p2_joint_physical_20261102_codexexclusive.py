"""Presealed known-root, fixed-winner physical development executor.

Preparation never creates an environment; execution requires explicit authorization.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from . import p2_joint_saved_20261102_codexexclusive as saved
from . import p2_joint_host_20261102_codexexclusive as host
from . import p2_conditional_physical_handoff_20261102_exclusive as prior
from . import p2_conditional_physical_executor_tmpfix_20261102_exclusive as executor
from . import p2_b0_isolated_consume_20261102_exclusive as isolation

NAME = "p2_joint_physical_20261102_codexexclusive"
ROOT, RESULTS = host.base.ROOT, host.RESULTS
RUN = RESULTS / "p2_joint_host_20261102_codexexclusive_RUN_02"
REPORT_SHA = "d991f0e57e2802ba09885eb29c18d2c95c836933db6ab5325c54cfeba11ea7c1"
PLAN = RESULTS / (NAME + "_PLAN_01.json")
HISTORY = RESULTS / (NAME + "_HISTORICAL_P0_01.py")
SESSION = RESULTS / (NAME + "_SESSION_01")
AUTH = "P2_JOINT_CONDITIONAL_FIXED_WINNERS_B0_PHYSICS_ONLY"
physical = executor.physical
ARMS = ("composition", "no_composition_sra")


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def build():
    model_audit = saved.consume(RUN, REPORT_SHA)
    report = json.loads(prior.checked(RUN / "report.json", REPORT_SHA))
    fixed = host.native.verified_input()
    require(report["result"]["context_sha256"] == fixed["context_sha256"]
            and report["result"]["goal_sha256"] == fixed["goal_sha256"], "model context/goal")
    original = json.loads(prior.checked(prior.old.PREP, prior.old.PREP_SHA))
    context = json.loads(fixed["context_json"])
    require(all(context[k] == original["observed_context"][k]
                for k in ("layout", "latent_history", "action_history")), "observed context")
    proof = json.loads(prior.checked(prior.isolation.OUTPUT, prior.PHYSICAL_PROOF_SHA))
    require(proof["status"] == "consumer_returned_success" and
            proof["consumer_result"]["status"] == "verified_known_root_physical_not_admitted" and
            proof["contract_sha256"] == prior.isolation.CONTRACT_SHA and
            proof["physical_report_sha256"] == prior.isolation.REPORT_SHA, "prior isolation proof")
    bundle, _, _, actions = prior.old.launch.wire.load_capture(
        prior.old.launch.CAPTURE / "capture.json", prior.old.launch.CAPTURE_SHA)
    prefix = np.asarray(actions, dtype="<f4").tobytes()
    require(bundle["base"] == 0 and bundle["source"]["fresh_root"] is False
            and original["job_index"] == 0 and len(prefix) == 120
            and len(original["prefix_reference"]) == 106, "B0 prefix")
    events = [physical.native.decode(line) for line in
              (prior.old.capture.RUN / "journal.jsonl").read_bytes().splitlines()]
    old_report = json.loads(prior.checked(prior.old.capture.RUN / "report.json", prior.old.capture.REPORT_SHA))
    require(host.base.file_sha(prior.old.capture.RUN / "journal.jsonl") == old_report["journal_sha256"],
            "historical journal identity")
    refs = [physical.normalized_prefix_event(e) for e in events
            if e.get("branch") == 0 and physical.is_prefix_event(e)]
    require(refs == original["prefix_reference"], "prefix reference")
    for frame in bundle["frames"]:
        pixels = [r["receipt"]["pixel_sha256"] for r in refs
                  if r["kind"] == "fresh_observation" and r["receipt"]["phase"] == "plan"
                  and r["receipt"]["step"] == frame["step"] and r["receipt"]["overlay_index"] == 0]
        require(pixels == [frame["pixel_sha256"]], "capture RGB and guarded prefix")
    for p in original["plans"]:
        require(np.asarray(p["commands"], dtype="<f4").tobytes()[:120] == prefix
                and p["native_root_sha256"] == bundle["source"]["root_sha256"]
                and p["task_id"] == bundle["source"]["task_id"], "root/task/prefix identity")
    template = original["plans"][0]
    plans, selected = [], []
    for index, arm in enumerate(ARMS):
        winner = report["result"]["model_selected_winners"][arm]
        suffix = bytes.fromhex(winner["raw_float32_hex"])
        require(len(suffix) == 80, "selected action size")
        host.native.optimizer.raw_actions(np.frombuffer(suffix, dtype="<f4").reshape(10, 2))
        raw = prefix + suffix
        witness = {"schema": NAME + ".winner_reference_v1", "arm": arm,
                   "model_report_sha256": REPORT_SHA, "model_plan_sha256": model_audit["plan_sha256"],
                   "candidate_index": winner["candidate_index"], "iteration": winner["iteration"],
                   "model_score": winner["model_score"], "suffix_sha256": hashlib.sha256(suffix).hexdigest(),
                   "input_context": deepcopy(original["observed_context"]),
                   "scope": "new optimized winner reference, not historical model witness"}
        witness["new_witness_sha256"] = physical.digest(witness)
        p = deepcopy(template)
        p.update(index=index, policy=arm, commands=np.frombuffer(raw, dtype="<f4").reshape(25, 2).astype(float).tolist(),
                 raw_float32_hex=raw.hex(), raw_float32_sha256=hashlib.sha256(raw).hexdigest(),
                 new_model_plan=witness)
        p.pop("selection_digest")
        p["selection_digest"] = physical.digest(p)
        plans.append(p)
        selected.append({"arm": arm, "method": "conditional_suffix" if index == 0 else "joint_gauss_newton_not_historical_sra",
                         **winner, "suffix_sha256": hashlib.sha256(suffix).hexdigest(),
                         "full25_sha256": hashlib.sha256(raw).hexdigest()})
    sources = host.base.source_bindings()
    for module in (saved, host, prior, executor, isolation, physical):
        sources[str(Path(module.__file__))] = host.base.file_sha(module.__file__)
    for name in (NAME + ".py", "p2_joint_physical_saved_20261102_codexexclusive.py"):
        path = Path(__file__).parent / name
        sources[str(path)] = host.base.file_sha(path)
    doc = ROOT.parent / "docs/PC_P2_JOINT_PHYSICAL_20261102_CODEXEXCLUSIVE.md"
    sources[str(doc)] = host.base.file_sha(doc)
    return {"schema": NAME + ".plan_v1", "model_report_sha256": REPORT_SHA,
            "model_plan_sha256": model_audit["plan_sha256"], "model_journal_tail": model_audit["summary"]["ledger_tail_sha256"],
            "old_prepared_sha256": prior.old.PREP_SHA, "capture_sha256": prior.old.launch.CAPTURE_SHA,
            "historical_p0_sha256": isolation.recovery.HISTORICAL_SHA,
            "root_sha256": bundle["source"]["root_sha256"], "task_id": bundle["source"]["task_id"],
            "context_sha256": fixed["context_sha256"], "goal_sha256": fixed["goal_sha256"],
            "prefix_sha256": hashlib.sha256(prefix).hexdigest(), "prefix_reference_sha256": physical.digest(refs),
            "selected": selected, "plans": plans, "source_sha256": sources,
            "caps": physical.contract()["caps"], "wall_seconds_dispatch_boundary": physical.WALL,
            "session": str(SESSION), "lock": str(executor.LOCK), "authorization": AUTH,
            "protocol": "fresh B0 per arm; exact warm10, guarded common prefix15, fixed winner10; no padding",
            "failure_policy": "native stop censored; engineering error stops later arms; no retry or imputation",
            "historical_sra_authenticated": False, "equal_total_compute": False,
            "independent_confirmation": False, "P1_admission": False, "P2_admission": False}


def prepare():
    require(not any(os.path.lexists(p) for p in (PLAN, HISTORY, SESSION)), "exclusive outputs")
    value = build()
    raw = isolation.historical_bytes()
    require(host.base.sha(raw) == isolation.recovery.HISTORICAL_SHA, "historical source")
    with HISTORY.open("xb") as stream:
        require(stream.write(raw) == len(raw), "historical source write")
        stream.flush()
        os.fsync(stream.fileno())
    return host.base.write_new(PLAN, value)


def verify(digest):
    value = json.loads(prior.checked(PLAN, digest))
    prior.checked(HISTORY, isolation.recovery.HISTORICAL_SHA)
    require(value == build() and host.base.file_sha(PLAN) == digest, "plan/source drift")
    return value


def derived(original, plan):
    p = deepcopy(original)
    p["plans"] = deepcopy(plan["plans"])
    for index, item in enumerate(p["plans"]):
        require(item["index"] == index and item["policy"] == ARMS[index]
                and item["selection_digest"] == physical.digest(
                    {k: v for k, v in item.items() if k != "selection_digest"}), "new plan")
    return p


def execute_pair(p, data, store, factory, digest):
    meter = physical.Meter(store)
    rows, errors, seen = physical.rows_for(p), [], []
    store.emit("joint_started", plan_sha256=digest)
    for row, plan in zip(rows, p["plans"], strict=True):
        env = None
        meter.where = dict(branch=row["index"], **{k: row[k] for k in
                           ("root_slot", "condition", "arm", "selection_digest")})
        row["status"] = "running"
        try:
            env = meter.call("constructor", factory)
            require(all(env is not x and env.unwrapped is not x.unwrapped for x in seen), "fresh environment")
            seen.append(env)
            physical.validate_env(env.unwrapped)
            require(env.unwrapped.render_action is False and env.unwrapped.render_size == 224
                    and env.unwrapped.render_mode == "rgb_array", "RGB224")
            with physical.native_hooks(env, meter, row["secondary_errors"]):
                physical.episode(env, data["roots"][plan["root_slot"]], data["source_plan"],
                                 plan, row, meter, store, p)
            require(not row["secondary_errors"], "hook restoration")
        except physical.native.allmode.NativeStop as exc:
            row.update(status="native_stopped", stop_reason=str(exc))
        except BaseException as exc:
            item = physical.native.base.error("branch", exc)
            row.update(status="engineering_failed", first_error=item)
            errors.append(item)
        finally:
            if env is not None:
                try:
                    meter.call("close", env.close, cleanup=True)
                except BaseException as exc:
                    item = physical.native.base.error("close", exc)
                    row["secondary_errors"].append(item)
                    errors.append(item)
                    row["status"] = "engineering_failed"
            for phase, label in (("warmup", "warmup"), ("plan", "own")):
                ops = [x for x in meter.receipts if x["resource"] == "public_step"
                       and x.get("branch") == row["index"] and x["phase"] == phase]
                row[label + "_attempted"] = len(ops)
                row[label + "_returned"] = sum(x["status"] == "returned" for x in ops)
        store.emit("branch_closed", receipt=row)
        if errors or row["secondary_errors"]:
            break
    errors.extend(meter.secondary)
    return rows, meter, errors


def child(digest, authorization):
    require(authorization == AUTH, "physical authorization")
    plan = verify(digest)
    require(host.base.file_sha(isolation.recovery.TARGET) == isolation.recovery.HISTORICAL_SHA,
            "historical read-only overlay")
    target = str(isolation.recovery.TARGET)
    require(any(len(parts := line.split()) > 6 and parts[4] == target and "ro" in parts[5].split(",")
                for line in Path("/proc/self/mountinfo").read_text().splitlines()), "read-only mount")
    require(SESSION.is_dir() and not SESSION.is_symlink(), "exclusive session")
    with host.native.locking.gpu_lock(executor.LOCK):
        original, data = physical.verify(prior.old.PREP, prior.old.PREP_SHA)
        p = derived(original, plan)
        require(p["source_slots"] == [r["slot"] for r in data["roots"]] and
                p["source_roots"] == physical.native.allmode.root_metadata(data) and
                p["warm_traces"] == [r["warm_trace"] for r in data["roots"]], "native source identity")
        path = SESSION / (physical.NAME + "_JOINT_01")
        store = physical.Store(path)
        store.put("binding.json", {"plan_sha256": digest, "old_plan_sha256": prior.old.PREP_SHA})
        def factory():
            import gymnasium as gym
            import stable_worldmodel.envs
            return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                            with_target=False, render_action=False)
        started = time.monotonic()
        rows, meter, errors = execute_pair(p, data, store, factory, digest)
        try:
            physical.verify(prior.old.PREP, prior.old.PREP_SHA)
            verify(digest)
            events = [physical.native.decode(x) for x in (path / "journal.jsonl").read_bytes().splitlines()]
            executor._audit_recorded(p, rows, events)
        except BaseException as exc:
            errors.append(physical.native.base.error("postcheck", exc))
        if time.monotonic() - meter.started >= physical.WALL:
            errors.append(physical.native.base.error("wall", TimeoutError("dispatch/postcheck boundary")))
        status = "recorded_development_unconsumed" if not errors else "partial_or_failed"
        store.finalizing = True
        store.emit("joint_finished", status=status, errors=errors, counts=meter.counts)
        report = {"schema": NAME + ".report_v1", "status": status, "rows": rows, "errors": errors,
                  "counts": meter.counts, "operation_receipts": meter.receipts, "plan_sha256": digest,
                  "journal_sha256": physical.sha(path / "journal.jsonl"), "wall_seconds": time.monotonic() - started,
                  "consumer_verified": False, "new_neural_calls": 0, "historical_sra_authenticated": False,
                  "independent_confirmation": False, "equal_total_compute": False, "P1_admission": False, "P2_admission": False}
        store.put("report.json", report, final=True)
        print(json.dumps({"status": status, "report_sha256": physical.sha(path / "report.json"), "output": str(path)}))
        if errors:
            raise SystemExit(1)


def run_host(digest, authorization):
    require(authorization == AUTH, "physical authorization")
    verify(digest)
    SESSION.mkdir(mode=0o700)
    physical.paired.fsync_dir(SESSION.parent)
    cmd = ["bwrap", "--die-with-parent", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
           "--ro-bind", str(HISTORY), str(isolation.recovery.TARGET), "--bind", str(SESSION), str(SESSION),
           "--bind", str(executor.LOCK), str(executor.LOCK), "--setenv", "TMPDIR", str(SESSION),
           "--setenv", "CUDA_VISIBLE_DEVICES", "", "--setenv", "OPENBLAS_NUM_THREADS", "1",
           "--setenv", "OMP_NUM_THREADS", "1", "--chdir", str(ROOT), "--",
           str(ROOT / ".venv/bin/python"), "-B", "-m", "experiments.spectral_reachability." + NAME,
           "run-child", "--plan-sha256", digest, "--authorization", authorization]
    result = subprocess.run(cmd, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    for name in ("check", "run-host", "run-child"):
        q = sub.add_parser(name)
        q.add_argument("--plan-sha256", required=True)
        if name != "check":
            q.add_argument("--authorization", required=True)
    a = p.parse_args()
    if a.command == "prepare":
        print(json.dumps({"plan": str(PLAN), "sha256": prepare()}))
    elif a.command == "check":
        verify(a.plan_sha256)
        print(json.dumps({"status": "verified_no_physical_execution"}))
    elif a.command == "run-host":
        run_host(a.plan_sha256, a.authorization)
    else:
        child(a.plan_sha256, a.authorization)


if __name__ == "__main__":
    main()
