"""Known-B0 conditional native physics executor; isolated historical source only.

Prepare is CPU-only. run-host starts a separate bubblewrap child only when
explicitly authorized; never call it without coordinating the shared lock.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from . import p2_conditional_physical_handoff_20261102_exclusive as handoff
from . import p2_conditional_native_recipe_20261102_exclusive as recipe
from . import p2_b0_isolated_consume_20261102_exclusive as isolation

ROOT = handoff.ROOT
RESULTS = handoff.RESULTS
physical = handoff.old.physical
NAME = "p2_conditional_physical_executor_20261102_exclusive"
SESSION = RESULTS / (NAME + "_SESSION_01")
PLAN = RESULTS / (NAME + "_PLAN_01.json")
HISTORY = RESULTS / (NAME + "_HISTORICAL_P0_01.py")
RECIPE = RESULTS / "p2_conditional_native_recipe_20261102_exclusive_01.json"
RECIPE_SHA = "6009cbaf5a44169e364f330492ece7ac8111fe4cf594fa93ade6477cbf2d9555"
AUTH = "P2_CONDITIONAL_B0_DEVELOPMENT_PHYSICS_ONLY"
LOCK = handoff.host.native.LOCK


def _checked_recipe():
    value = json.loads(handoff.checked(RECIPE, RECIPE_SHA))
    if value != recipe.build():
        raise ValueError("native recipe reconstruction")
    return value


def plan_value():
    _checked_recipe()
    history = isolation.historical_bytes()
    if handoff.host.sha(history) != isolation.recovery.HISTORICAL_SHA:
        raise ValueError("historical source mismatch")
    bundle, _, _, _ = handoff.old.launch.wire.load_capture(
        handoff.old.launch.CAPTURE / "capture.json", handoff.old.launch.CAPTURE_SHA)
    p = json.loads(handoff.checked(handoff.old.PREP, handoff.old.PREP_SHA))
    for frame in bundle["frames"]:
        found = [r["receipt"]["pixel_sha256"] for r in p["prefix_reference"]
                 if r["kind"] == "fresh_observation" and r["receipt"]["phase"] == "plan"
                 and r["receipt"]["step"] == frame["step"] and r["receipt"]["overlay_index"] == 0]
        if found != [frame["pixel_sha256"]]:
            raise ValueError("capture/physical reference mismatch")
    sources = [Path(__file__), Path(recipe.__file__), Path(handoff.__file__),
               Path(physical.__file__), Path(isolation.__file__)]
    return {"schema": NAME + ".plan_v1", "session": str(SESSION),
            "recipe_sha256": RECIPE_SHA, "handoff_sha256": recipe.PLAN_SHA,
            "historical_source_sha256": isolation.recovery.HISTORICAL_SHA,
            "historical_target": str(isolation.recovery.TARGET),
            "source_sha256": {str(path): handoff.host.file_sha(path) for path in sources},
            "cooperative_lock": str(LOCK), "authorization_required": AUTH,
            "execution": "two fresh B0 envs; exact warm10 and guarded prefix15; fixed suffix10",
            "failure": "no retries, preserve partial artifacts, no padding",
            "caps": physical.contract()["caps"], "independent_confirmation": False,
            "equal_total_compute": False, "P1_admission": False, "P2_admission": False}


def prepare():
    if any(os.path.lexists(path) for path in (PLAN, HISTORY, SESSION)):
        raise FileExistsError("exclusive plan/history/session already exists")
    plan = plan_value()
    with HISTORY.open("xb") as stream:
        raw = isolation.historical_bytes()
        if stream.write(raw) != len(raw):
            raise OSError("short historical source write")
        stream.flush()
        os.fsync(stream.fileno())
    return handoff.host.write_new(PLAN, plan)


def verify(expected):
    value = json.loads(handoff.checked(PLAN, expected))
    handoff.checked(HISTORY, isolation.recovery.HISTORICAL_SHA)
    if value != plan_value() or handoff.host.file_sha(PLAN) != expected:
        raise ValueError("executor plan or source drift")
    return value


def _derived(old_p, saved_recipe):
    """Old p must already have passed its own original complete verifier."""
    p = deepcopy(old_p)
    p["plans"] = deepcopy(saved_recipe["plans"])
    for index, plan in enumerate(p["plans"]):
        if (plan["index"] != index or plan["policy"] != handoff.ARMS[index]
                or plan["new_model_plan"]["scope"] != "new optimized winner reference, not historical model witness"
                or plan["selection_digest"] != physical.digest(
                    {k: v for k, v in plan.items() if k != "selection_digest"})):
            raise ValueError("conditional new winner reference")
    return p


def execute_pair(p, data, store, factory):
    """Runs authenticated plan; caller owns lock, input seals and output."""
    meter = physical.Meter(store)
    rows = physical.rows_for(p)
    errors, seen = [], []
    store.emit("conditional_started", recipe_sha256=RECIPE_SHA)
    for row, plan in zip(rows, p["plans"], strict=True):
        env = None
        meter.where = dict(branch=row["index"], **{k: row[k] for k in
                           ("root_slot", "condition", "arm", "selection_digest")})
        row["status"] = "running"
        try:
            env = meter.call("constructor", factory)
            physical.require(all(env is not prior and env.unwrapped is not prior.unwrapped
                                 for prior in seen), "fresh env per arm")
            seen.append(env)
            physical.validate_env(env.unwrapped)
            physical.require(env.unwrapped.render_action is False and env.unwrapped.render_size == 224
                             and env.unwrapped.render_mode == "rgb_array", "native RGB224")
            with physical.native_hooks(env, meter, row["secondary_errors"]):
                physical.episode(env, data["roots"][plan["root_slot"]], data["source_plan"],
                                 plan, row, meter, store, p)
            physical.require(not row["secondary_errors"], "hook restoration")
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
                ops = [op for op in meter.receipts if op["resource"] == "public_step"
                       and op.get("branch") == row["index"] and op["phase"] == phase]
                row[label + "_attempted"] = len(ops)
                row[label + "_returned"] = sum(op["status"] == "returned" for op in ops)
        store.emit("branch_closed", receipt=row)
        if errors or row["secondary_errors"]:
            break
    errors.extend(meter.secondary)
    return rows, meter, errors


def _audit_recorded(p, rows, events):
    for index, (row, plan) in enumerate(zip(rows, p["plans"], strict=True)):
        if row["status"] == "not_started":
            continue
        scoped = [e for e in events if e.get("branch", e.get("receipt", {}).get("branch")) == index]
        physical.audit_feedback(row, plan, scoped, p)
        physical.audit_labels(row, plan, scoped, p)
        expected = physical.raw25(plan["commands"])
        returned = [e for e in scoped if e["kind"] == "returned_transition" and e["phase"] == "plan"]
        physical.require(len(returned) <= 25 and len(returned) == row["own_returned"],
                         "actual return census")
        for step, event in enumerate(returned, 1):
            bits = expected[step - 1].tobytes().hex()
            physical.require(event["step"] == step and event["requested_bits"] == event["actual_bits"] == bits,
                             "actual fixed-action return")
        if row["status"] == "horizon_completed":
            physical.require(len(returned) == 25 and row["feedback_guard"] is not None,
                             "no incomplete full-horizon label")
        if row["status"] == "native_stopped":
            physical.require(not returned or returned[-1]["terminated"] or returned[-1]["truncated"]
                             or row["own_returned"] == 0, "stop not padded")


def run_child(expected, authorization):
    if authorization != AUTH:
        raise ValueError("explicit physics authorization required")
    verify(expected)
    if handoff.host.file_sha(isolation.recovery.TARGET) != isolation.recovery.HISTORICAL_SHA:
        raise ValueError("read-only historical view is required")
    target = str(isolation.recovery.TARGET)
    mounts = Path("/proc/self/mountinfo").read_text().splitlines()
    if not any(len(parts := line.split()) > 6 and parts[4] == target
               and "ro" in parts[5].split(",") for line in mounts):
        raise ValueError("historical source must be a read-only mounted file")
    if SESSION.is_symlink() or not SESSION.is_dir() or SESSION.parent.resolve() != RESULTS:
        raise ValueError("exclusive session directory required")
    with handoff.host.native.locking.gpu_lock(LOCK):
        # This validates the OLD plan and its original witnesses before deriving
        # the new separately authenticated witness; no old validation patched.
        original, data = physical.verify(handoff.old.PREP, handoff.old.PREP_SHA)
        p = _derived(original, _checked_recipe())
        if (p["source_slots"] != [r["slot"] for r in data["roots"]] or
                p["source_roots"] != physical.native.allmode.root_metadata(data) or
                p["warm_traces"] != [r["warm_trace"] for r in data["roots"]]):
            raise ValueError("native source identity")
        path = SESSION / (physical.NAME + "_CONDITIONAL_01")
        store = physical.Store(path)
        store.put("binding.json", {"plan_sha256": expected, "recipe_sha256": RECIPE_SHA,
                                   "old_plan_sha256": handoff.old.PREP_SHA})
        def factory():
            import gymnasium as gym
            import stable_worldmodel.envs
            return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                            with_target=False, render_action=False)
        started = time.monotonic()
        rows, meter, errors = execute_pair(p, data, store, factory)
        try:
            physical.verify(handoff.old.PREP, handoff.old.PREP_SHA)
            verify(expected)
            events = [physical.native.decode(x) for x in (path / "journal.jsonl").read_bytes().splitlines()]
            _audit_recorded(p, rows, events)
        except BaseException as exc:
            errors.append(physical.native.base.error("postcheck", exc))
        if time.monotonic() - meter.started >= physical.WALL:
            errors.append(physical.native.base.error("wall", TimeoutError("900 second dispatch/postcheck boundary")))
        status = "recorded_development_unconsumed" if not errors else "partial_or_failed"
        store.finalizing = True
        store.emit("conditional_finished", status=status, errors=errors, counts=meter.counts)
        report = {"schema": NAME + ".report_v1", "status": status, "rows": rows,
                  "errors": errors, "counts": meter.counts,
                  "operation_receipts": meter.receipts, "plan_sha256": expected,
                  "recipe_sha256": RECIPE_SHA, "journal_sha256": physical.sha(path / "journal.jsonl"),
                  "wall_seconds": time.monotonic() - started, "consumer_verified": False,
                  "new_neural_calls": 0, "independent_confirmation": False,
                  "equal_total_compute": False, "P1_admission": False, "P2_admission": False}
        store.put("report.json", report, final=True)
        print(json.dumps({"status": status, "report_sha256": physical.sha(path / "report.json"),
                          "output": str(path)}))
        if errors:
            raise SystemExit(1)


def run_host(expected, authorization):
    if authorization != AUTH:
        raise ValueError("explicit physics authorization required")
    verify(expected)
    SESSION.mkdir(mode=0o700)
    physical.paired.fsync_dir(SESSION.parent)
    cmd = ["bwrap", "--die-with-parent", "--ro-bind", "/", "/", "--dev", "/dev",
           "--proc", "/proc", "--ro-bind", str(HISTORY), str(isolation.recovery.TARGET),
           "--bind", str(SESSION), str(SESSION), "--bind", str(LOCK), str(LOCK),
           "--chdir", str(ROOT), "--", str(ROOT / ".venv/bin/python"), "-B", "-m",
           "experiments.spectral_reachability." + NAME, "run-child",
           "--plan-sha256", expected, "--authorization", authorization]
    child = subprocess.run(cmd, check=False)
    if child.returncode:
        raise SystemExit(child.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    for name in ("check", "run-host", "run-child"):
        command = sub.add_parser(name)
        command.add_argument("--plan-sha256", required=True)
        if name != "check":
            command.add_argument("--authorization", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps({"plan": str(PLAN), "sha256": prepare()}))
    elif args.command == "check":
        verify(args.plan_sha256)
        print(json.dumps({"status": "verified_no_physical_execution"}))
    elif args.command == "run-host":
        run_host(args.plan_sha256, args.authorization)
    else:
        run_child(args.plan_sha256, args.authorization)


if __name__ == "__main__":
    main()
