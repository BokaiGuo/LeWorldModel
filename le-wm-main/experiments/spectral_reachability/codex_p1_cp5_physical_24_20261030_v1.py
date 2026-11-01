"""Fixed known-root 24-row cp5 suffix physical exploration; no inference."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from . import codex_p1_three_prefix_h10_runner_20261030 as h10
from . import codex_p1_cp5_exploratory_saved_20261030_v1 as plans

ROOT = h10.ROOT
NAME = "codex_p1_cp5_physical_24_20261030_v1"
PLAN_DIR = ROOT/"results/codex_p1_cp5_exploratory_cuda_20261030_v1_RUN01"
MANIFEST_SHA = "fd54ebcdbf5847cc662bd019e69be34222bbb739be05651302fb11fa2adeca3b"
REPORT_SHA = "7b75ffb04404b4f6c3e4f22884eafcef7b38fb6829f9acf723996007ba01f09e"
RECEIPT = ROOT/"results/codex_p1_cp5_exploratory_saved_20261030_v1_RUN01.json"
RECEIPT_SHA = "d871135093687d66caa7fa02ccd28a1336f8c88c8c3dd8479895106b6afffdb4"
CAPS = {"constructor": 1, "close": 1, "reset": 2, "public_step": 25, "physics_step": 254,
        "set_state": 4, "goal_setter": 3, "goal_render": 2, "goal_bind": 1, "explicit_render": 81}
need = h10.require
sha = h10.runner.physical.sha


def make_roster(anchors, blind, events, receipt):
    need(len(anchors) == len(blind) == len(events) == len(receipt["rows"]) == 24, "24 rows")
    roster = []
    for i, (anchor, row, event, rec) in enumerate(zip(anchors, blind, events, receipt["rows"], strict=True)):
        need(anchor["index"] == row["index"] == event["index"] == rec["index"] == i
             and anchor["logical_pair"] == row["logical_pair"]
             and anchor["role"] == row["role"]
             and anchor["prefix5_float32_hex"] == row["prefix_raw_f32_hex"]
             and anchor["checkpoint"]["pixel_sha256"] == row["cp5_pixel_sha256"], "prefix/cp5 identity")
        raw = np.asarray(event["result"]["actions"], dtype="<f4")
        need(raw.shape == (1, 2, 5, 2) and np.isfinite(raw).all() and (abs(raw) <= 1).all(),
             "finite raw10 shape/bounds")
        bits = raw.reshape(10, 2).tobytes()
        roster.append({"index": i, "logical_pair": anchor["logical_pair"], "root_slot": anchor["root_slot"],
                       "condition": anchor["condition"], "arm": anchor["arm"], "role": anchor["role"],
                       "task_id": anchor["task_id"], "prefix5_float32_hex": anchor["prefix5_float32_hex"],
                       "cp5_file_sha256": anchor["checkpoint"]["file_sha256"],
                       "cp5_pixel_sha256": anchor["checkpoint"]["pixel_sha256"],
                       "plan_row_sha256": rec["row_sha256"], "raw10_float32_hex": bits.hex(),
                       "raw10_sha256": hashlib.sha256(bits).hexdigest()})
    return roster


def build():
    p = h10.design.verify(h10.INPUT, h10.INPUT_SHA)
    verified = plans.consume(PLAN_DIR, MANIFEST_SHA, REPORT_SHA)
    need(sha(RECEIPT) == RECEIPT_SHA and json.loads(RECEIPT.read_bytes()) == verified,
         "planner independent receipt")
    blind = json.loads(plans.run.BUNDLE.read_bytes())
    events = []
    for rec in verified["rows"]:
        path = PLAN_DIR/f"row_{rec['index']:02d}_result.json"
        need(sha(path) == rec["row_sha256"], "planned row SHA")
        events.append(json.loads(path.read_bytes()))
    roster = make_roster(p["anchors"], blind["rows"], events, verified)
    sources = dict(p["bindings"])
    for path in [h10.INPUT, RECEIPT, Path(__file__).resolve(), Path(plans.__file__).resolve(),
                 Path(__file__).with_name("codex_p1_cp5_physical_saved_20261030_v1.py").resolve(),
                 ROOT/"tests/test_codex_p1_cp5_physical_24_20261030_v1.py",
                 *h10.source_files()]:
        sources[str(path)] = sha(path)
    need(all(sha(path) == value for path, value in sources.items()), "source binding")
    return {"schema": NAME+".prepare", "roster": roster, "caps_per_row": CAPS,
            "h10_prepare_sha256": h10.INPUT_SHA, "plan_manifest_sha256": MANIFEST_SHA,
            "plan_report_sha256": REPORT_SHA, "plan_receipt_sha256": RECEIPT_SHA,
            "bindings": sources, "fixed_rows": 24, "physical_budget_wall_per_row_seconds": h10.WALL,
            "known_roots": True, "os_isolation_certified": False, "independent_confirmation": False,
            "P1_admission": False, "fairness_proven": False}


def prepare(path):
    path = Path(path)
    need(path.is_absolute() and path.parent == ROOT/"results" and path.name.startswith(NAME+"_prepare_")
         and not os.path.lexists(path), "exclusive prepare")
    return h10._write_new(path, build())


def verify(path, expected):
    path = Path(path)
    need(path.is_absolute() and path.is_file() and not path.is_symlink() and sha(path) == expected,
         "prepared hash")
    p = h10.native.read(path, cap=64*1024**2)
    need(p == build() and sha(path) == expected, "prepared/source drift")
    return p


class Meter(h10.Meter):
    def call(self, kind, fn, *, cleanup=False, **fields):
        need(kind in CAPS and self.counts[kind]["attempted"] < CAPS[kind], "single-row cap")
        return super().call(kind, fn, cleanup=cleanup, **fields)


def execute(anchor, item, data, path, factory):
    store = h10.ObservedStore(h10.Store(path))
    probe = {"index": 0, "name": "independently_planned_raw10",
             "raw10_float32_hex": item["raw10_float32_hex"], "raw10_sha256": item["raw10_sha256"]}
    row = h10.runner.row_for(anchor, probe)
    meter, errors = Meter(store), []
    meter.where = dict(branch=0, anchor_index=anchor["index"], condition=row["condition"],
                       arm=row["arm"], candidate_role=row["candidate_role"], probe=row["probe"])
    raw = np.frombuffer(bytes.fromhex(anchor["prefix5_float32_hex"]), dtype="<f4").reshape(5, 2)
    plan = {"condition": anchor["condition"], "commands": np.concatenate([raw, np.zeros((5, 2), np.float32)]).tolist()}
    env = None
    store.emit("run_started", anchor=anchor["index"], fixed_rows=1, caps=CAPS,
               raw10_sha256=item["raw10_sha256"])
    try:
        env = meter.call("constructor", factory)
        h10.base.validate_env(env.unwrapped)
        need(env.unwrapped.render_action is False and env.unwrapped.render_size == 224
             and env.unwrapped.render_mode == "rgb_array", "renderer")
        with h10.base.native_hooks(env, meter, row["secondary_errors"]):
            h10.episode(env, data["roots"][row["root_slot"]], data["source_plan"],
                        plan, anchor, probe, row, meter, store)
        need(not row["secondary_errors"], "hook cleanup")
    except BaseException as exc:
        row.update(status="engineering_failed", error=h10.base.error("branch", exc))
        errors.append(row["error"])
    finally:
        if env is not None:
            try:
                meter.call("close", env.close, cleanup=True)
            except BaseException as exc:
                errors.append(h10.base.error("close", exc))
    store.emit("branch_closed", branch=0, status=row["status"])
    valid = not errors and row["status"] in ("probe_completed", "native_stopped", "censored_out_of_domain")
    status = "recorded_not_consumed" if valid else "partial_or_failed"
    store.emit("run_finished", status=status, counts=meter.counts, errors=errors)
    report = {"schema": NAME+".row", "status": status, "row": row, "counts": meter.counts,
              "receipts": meter.receipts, "errors": errors, "journal_chain_tail": store.previous,
              "journal_sha256": sha(path/"journal.jsonl"), "wall_seconds": time.monotonic()-meter.started,
              "raw10_sha256": item["raw10_sha256"], "P1_admission": False}
    store.put("report.json", report, final=True)
    return report


def run(prepared, expected, output):
    p = verify(prepared, expected)
    old = h10.old
    source_report = h10.native.read(old.prep.DIRECTORY/"report.json", cap=64*1024**2)
    _, data = h10.runner.physical.verify(source_report["execution_binding"]["prepared"], old.prep.PREP_SHA)
    h10.native.constructor_contract()
    anchors = h10.native.read(h10.INPUT, cap=64*1024**2)["anchors"]
    output = Path(output)
    need(output.is_absolute() and output.parent == ROOT/"results" and
         output.name.startswith(NAME+"_run_") and not os.path.lexists(output) and
         shutil.disk_usage(output.parent).free >= 2*1024**3, "exclusive run/disk")
    output.mkdir(mode=0o700)
    binding_sha = h10._write_new(output/"binding.json", {"prepare_sha256": expected, "fixed_rows": 24})
    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
    shards = []
    failure = None
    try:
        for anchor, item in zip(anchors, p["roster"], strict=True):
            path = output/f"row_{anchor['index']:02d}"
            report = execute(anchor, item, data, path, factory)
            shards.append({"index": anchor["index"], "report_sha256": sha(path/"report.json"),
                           "status": report["status"]})
            if report["status"] != "recorded_not_consumed":
                failure = {"index": anchor["index"], "errors": report["errors"]}
                break
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    result = {"schema": NAME+".run", "status": "recorded_not_consumed" if len(shards) == 24 and not failure
              else "partial_or_failed", "shards": shards, "failure": failure, "fixed_rows": 24,
              "binding_sha256": binding_sha, "prepare_sha256": expected, "P1_admission": False,
              "independent_confirmation": False, "os_isolation_certified": False}
    digest = h10._write_new(output/"run_report.json", result)
    print(json.dumps({"status": result["status"], "sha256": digest, "rows": len(shards),
                      "failure": failure}, sort_keys=True), flush=True)
    return 0 if result["status"] == "recorded_not_consumed" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps({"sha256": prepare(args.output)}))
        return 0
    need(args.prepared and args.sha256, "prepare required")
    return run(args.prepared, args.sha256, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
