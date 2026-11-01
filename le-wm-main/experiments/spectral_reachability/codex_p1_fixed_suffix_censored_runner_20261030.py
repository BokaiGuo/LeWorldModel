"""Execute all 32 fixed CUDA plans in the original known-root physical setup."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from . import codex_p1_fixed_prefix_suffix_cuda_20261030_main as producer
from . import codex_p1_fixed_prefix_suffix_recheck_20261030_main as checks
from . import p1_backtrack_physical_runner_20261030 as prior
from . import codex_p1_fixed_suffix_censored_episode_20261030 as episode

NAME = "codex_p1_fixed_suffix_censored_runner_20261030"
ROOT, native, need, sha = prior.ROOT, prior.native, prior.require, prior.sha
CAPS = {k: 2 * v for k, v in prior.contract.contract()["caps"].items()}
OUTPUT_CAP = 1024 * 1024**2
FINAL_RESERVE = 32 * 1024**2
FLOOR = 512 * 1024**2
WALL = 7200


def plans_from_traces(report, binding, joined, traces):
    need(report["status"] == "complete" and len(report["rows"]) == len(binding["rows"]) == len(traces) == 8,
         "complete source roster")
    plans = []
    for i, (result, source, trace) in enumerate(zip(report["rows"], binding["rows"], traces, strict=True)):
        task = joined["task_rows"][source["slot"]]
        need(result["slot"] == source["slot"] and result["arm"] == source["arm"] and
             result["indices"] == source["indices"] and
             task["task_id"] == source["task"]["binding"]["task_id"], "task identity")
        need([e["kind"] for e in trace] == ["inputs"] + ["optimized"] * 4 + ["selection"],
             "trace roster")
        plans_raw = np.asarray(trace[-1]["payload"]["actions"], dtype=np.float32)
        need(plans_raw.shape == (4, 3, 5, 2) and np.isfinite(plans_raw).all() and
             (np.abs(plans_raw) <= 1).all(), "physical raw action domain")
        for j in range(4):
            candidate = plans_raw[j]
            need(np.array_equal(candidate[0], np.asarray(source["prefixes"][j], np.float32)),
                 "literal fixed prefix")
            raw = candidate.astype("<f4", copy=False).tobytes()
            plans.append({"index": len(plans), "slot": source["slot"], "root_slot": task["root_slot"],
                          "condition": task["condition"], "arm": source["arm"], "role": f"candidate_{j}",
                          "candidate_index": source["indices"][j], "task_id": task["task_id"],
                          "physical_root_sha256": task["native_root_sha256"],
                          "raw15_float32_hex": raw.hex(), "raw15_sha256": hashlib.sha256(raw).hexdigest()})
    need(len(plans) == 32, "fixed 32 branch denominator")
    return plans


def build():
    # This also verifies the complete original candidate panel and all 32
    # saved same-action successors before any physical action is released.
    check = checks.check()
    need(check["candidate_checks"] == 32, "saved model check")
    report = producer.source.saved.read_verified(checks.DIRECTORY / "report.json", checks.REPORT_SHA)
    binding_path = checks.DIRECTORY / "binding.json"
    need(sha(binding_path) == report["binding_sha256"], "model input binding")
    binding = native.read(binding_path, cap=64 * 1024**2)
    join_prepare, data = prior.material()
    joined = native.read(prior.JOIN, cap=32 * 1024**2)
    traces, bindings = [], deepcopy(join_prepare["bindings"])
    for i, row in enumerate(report["rows"]):
        path = checks.DIRECTORY / f"node_{i:02d}/trace.jsonl"
        need(sha(path) == row["trace_sha256"], "source trace hash")
        traces.append([json.loads(line) for line in path.read_text().splitlines()])
        bindings[str(path)] = row["trace_sha256"]
    protocol = ROOT.parent / "docs/D2P_P1_FIXED_SUFFIX_CENSORED_PROTOCOL_20261030_MAIN.md"
    for path in (Path(__file__), Path(episode.__file__), protocol, Path(checks.__file__), Path(producer.__file__),
                 checks.DIRECTORY / "report.json", binding_path):
        path = path.resolve()
        bindings[str(path)] = sha(path)
    plans = plans_from_traces(report, binding, joined, traces)
    need(all(sha(path) == value for path, value in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "plans": plans, "bindings": bindings, "caps": CAPS,
            "source_report_sha256": checks.REPORT_SHA, "fixed_rows": 32, "known_roots": 2,
            "candidate_policy": "all four optimized suffixes for every source arm; no outcome selection",
            "censor_policy": "stop branch only on proven plan-step finite canvas violation; never pad",
            "P1_admission": False, "independent_confirmation": False, "new_neural_calls": 0}, data


def exclusive(path, marker):
    path = Path(path)
    need(path.is_absolute() and path.parent == ROOT / "results" and
         path.name.startswith(NAME + "_" + marker + "_") and not path.exists() and not path.is_symlink(),
         "exclusive output")
    return path


class Store(prior.Store):
    def __init__(self, path):
        self.path = exclusive(path, "run")
        need(shutil.disk_usage(self.path.parent).free >= FLOOR + OUTPUT_CAP, "disk reserve")
        self.path.mkdir()
        native.paired.fsync_dir(self.path.parent)
        self.previous, self.index = None, 0

    def check(self, size=4096, final=False):
        files = list(self.path.iterdir())
        need(all(f.is_file() and not f.is_symlink() for f in files) and
             sum(f.stat().st_size for f in files) + size <= OUTPUT_CAP - (0 if final else FINAL_RESERVE) and
             shutil.disk_usage(self.path).free >= FLOOR + size, "output budget")


class Meter(prior.Meter):
    def __init__(self, store, started):
        self.store, self.started, self.where = store, started, {}
        self.receipts, self.secondary, self.fault = [], [], False
        self.counts = {k: dict(attempted=0, completed=0) for k in CAPS}
        self.active = {k: 0 for k in CAPS}

    def call(self, kind, fn, *, cleanup=False, **fields):
        need(not cleanup or kind == "close", "only close may bypass fault")
        if not cleanup:
            need(not self.fault and time.monotonic() - self.started < WALL, "wall/fault")
            self.store.check()
        count = self.counts[kind]
        need(count["attempted"] < CAPS[kind], "resource cap")
        receipt = dict(index=len(self.receipts), resource=kind, **deepcopy(self.where),
                       **deepcopy(fields), status="pending")
        try:
            self.store.emit("operation_dispatch_intent", receipt=receipt)
        except BaseException as exc:
            self.fault = True
            if not cleanup:
                raise
            self.secondary.append(native.base.error("cleanup_intent", exc))
        self.receipts.append(receipt)
        count["attempted"] += 1
        self.active[kind] += 1
        try:
            value = fn()
            count["completed"] += 1
            receipt["status"] = "returned"
        except BaseException as exc:
            receipt.update(status="failed", error=native.base.error(kind, exc))
            try:
                self.store.emit("operation_failed", receipt=receipt)
            except BaseException as extra:
                self.fault = True
                self.secondary.append(native.base.error("failure_journal", extra))
            raise
        finally:
            self.active[kind] -= 1
        try:
            self.store.emit("operation_returned", receipt=receipt)
        except BaseException as exc:
            self.fault = True
            if not cleanup:
                raise
            self.secondary.append(native.base.error("cleanup_return", exc))
        return value


def run_branches(current, data, store, meter, rows, errors, make):
    seen = []
    for plan, row in zip(current["plans"], rows, strict=True):
        meter.where = dict(branch=row["index"], candidate_role=row["role"],
                           **{k: row[k] for k in ("root_slot", "condition", "arm")})
        env = None
        row["status"] = "running"
        try:
            env = meter.call("constructor", lambda: make("swm/PushT-v1", render_mode="rgb_array",
                                                        resolution=224, with_target=False))
            need(all(env is not old and env.unwrapped is not old.unwrapped for old in seen), "fresh env")
            seen.append(env)
            native.validate_env(env.unwrapped)
            need(env.unwrapped.render_action is False and env.unwrapped.render_size == 224 and
                 env.unwrapped.render_mode == "rgb_array", "renderer")
            with native.native_hooks(env, meter, row["secondary_errors"]):
                episode.episode(env, data["roots"][plan["root_slot"]], data["source_plan"],
                                plan, row, meter, store)
            need(not row["secondary_errors"], "hook cleanup")
        except episode.CanvasCensored as exc:
            row.update(status="censored_out_of_domain", censor=exc.proof)
        except native.NativeStop as exc:
            row.update(status="native_stopped", stop_reason=str(exc))
        except BaseException as exc:
            item = native.base.error("branch", exc)
            row.update(status="engineering_failed", first_error=item)
            errors.append(item)
        finally:
            if env is not None:
                try:
                    meter.call("close", env.close, cleanup=True)
                except BaseException as exc:
                    errors.append(native.base.error("close", exc))
            for phase, label in (("warmup", "warmup"), ("plan", "own")):
                ops = [r for r in meter.receipts if r["resource"] == "public_step" and
                       r.get("branch") == row["index"] and r.get("phase") == phase]
                row[label + "_attempted"] = len(ops)
                row[label + "_returned"] = sum(r["status"] == "returned" for r in ops)
        store.emit("branch_closed", receipt=row)
        if errors or row["secondary_errors"]:
            break


def execute(prepared, expected, output):
    path = Path(prepared)
    need(path.is_absolute() and path.is_file() and not path.is_symlink() and sha(path) == expected, "prepare SHA")
    saved = native.read(path, cap=64 * 1024**2)
    current, data = build()
    need(saved == current and sha(path) == expected, "prepare/source drift")
    native.constructor_contract()
    from gymnasium import make
    import stable_worldmodel  # noqa: F401
    store = Store(output)
    meter = Meter(store, time.monotonic())
    rows = [prior.make_row(plan) for plan in current["plans"]]
    errors = []
    try:
        store.put("execution_binding.json", {"prepared": str(path), "prepare_sha256": expected})
        store.emit("run_started", fixed_rows=32, caps=CAPS)
        run_branches(current, data, store, meter, rows, errors, make)
    except BaseException as exc:
        errors.append(native.base.error("execution", exc))
    errors.extend(meter.secondary)
    try:
        need(all(sha(p) == h for p, h in current["bindings"].items()), "final source drift")
        need(not meter.secondary and not any(meter.active.values()) and
             all(v["attempted"] == v["completed"] for v in meter.counts.values()), "operation census")
        if not errors:
            c = {k: v["completed"] for k, v in meter.counts.items()}
            need(c["constructor"] == c["close"] == 32 and c["reset"] == 64 and
                 c["physics_step"] == 2 * c["reset"] + 10 * c["public_step"] and
                 c["set_state"] == 2 * c["reset"] and c["goal_render"] == c["reset"] and
                 c["goal_setter"] == c["reset"] + c["goal_bind"] and
                 c["public_step"] == sum(r["warmup_returned"] + r["own_returned"] for r in rows) and
                 c["explicit_render"] == sum(r["PNG"] for r in rows), "32-branch census")
    except BaseException as exc:
        errors.append(native.base.error("final_validation", exc))
    complete = not errors and all(r["status"] in ("horizon_completed", "native_stopped", "censored_out_of_domain")
                                  for r in rows)
    status = "physical_recorded_not_independently_consumed" if complete else "partial_or_failed"
    store.emit("run_finished", status=status, counts=meter.counts, errors=errors)
    report = {"schema": NAME + ".report", "status": status, "rows": rows, "errors": errors,
              "counts": meter.counts, "receipts": meter.receipts, "prepared": str(path),
              "prepare_sha256": expected, "journal_chain_tail": store.previous,
              "journal_sha256": sha(store.path / "journal.jsonl"), "fixed_rows": 32,
              "P1_admission": False, "independent_confirmation": False, "new_neural_calls": 0}
    store.put("report.json", report, final=True)
    return status, sha(store.path / "report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        path = exclusive(args.output, "prepare")
        value, _ = build()
        digest = producer.write_new(path, value)
        result = {"status": "prepared", "sha256": digest}
    else:
        status, digest = execute(args.prepared, args.sha, args.output)
        result = {"status": status, "sha256": digest}
    print(json.dumps(result, sort_keys=True))
    return int(result["status"] == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
