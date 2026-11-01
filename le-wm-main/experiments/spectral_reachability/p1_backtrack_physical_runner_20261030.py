"""Fixed 16-row known-root P1 physical exploration, with no neural calls.

This runner is not an independent confirmation, a capability probe, or P1
admission. A complete run still requires a separate strict saved consumer.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time

from experiments.spectral_reachability import agent_allmode_physical_20261028_v1 as native
from experiments.spectral_reachability import p1_backtrack_physical_contract_20261030 as contract
from experiments.spectral_reachability import p1_backtrack_physical_episode_20261030 as episode
from experiments.spectral_reachability import p1_backtrack_physical_source_join_20261030 as join
from experiments.spectral_reachability import p1_backtrack_physical_sequence_20261030 as flow
from experiments.spectral_reachability import p1_backtrack_prefix_guard_20261030 as guard

NAME = "p1_backtrack_physical_runner_20261030"
ROOT = native.ROOT
JOIN = ROOT / "results/p1_backtrack_physical_source_join_20261030_MAIN02.json"
JOIN_SHA = "cb82a39b8a464f16b7e9f71cd9d072f2d48c0887bf36d57c91ca50baf0f2af84"
AUTH = "FIXED16_KNOWN_ROOT_EXPLORATION"
require, sha = native.require, native.sha


def material():
    require(sha(JOIN) == JOIN_SHA, "fixed source join SHA")
    joined = native.read(JOIN, cap=32 * 1024**2)
    plans = contract.plans(joined)
    data = native.source_material()
    require(joined["source_plan"] == data["source_plan"] and
            joined["source_slots"] == [r["slot"] for r in data["roots"]] and
            joined["warm_traces"] == [r["warm_trace"] for r in data["roots"]] and
            joined["source_bindings"] == data["bindings"], "native source closure")
    metadata = native.root_metadata(data)
    for task in joined["task_rows"]:
        root = metadata[task["root_slot"]]
        require(task["native_root_sha256"] == root["root_sha256"] and
                task["native_goal"] == root["goal"] and task["native_root_state"] == root["root_state"] and
                task["renderer"] == root["renderer"] and
                task["goal_pixel_sha256"] == root["goal_pixels"][task["condition"]], "native task closure")
    bindings = deepcopy(data["bindings"])
    paths = [Path(__file__), Path(contract.__file__), Path(episode.__file__), Path(join.__file__),
             Path(flow.__file__), Path(guard.__file__), *native.own_files()]
    for path in paths:
        path = str(path.resolve())
        h = sha(path)
        require(path not in bindings or bindings[path] == h, "source seal collision")
        bindings[path] = h
    bindings[str(JOIN)] = JOIN_SHA
    for path, digest in bindings.items():
        require(sha(path) == digest, "input/source hash")
    return dict(schema=NAME + ".prepare", contract=contract.contract(), plans=plans,
                join_path=str(JOIN), join_sha256=JOIN_SHA, bindings=bindings,
                P1_admission=False, physical_authorized=False), data


def prepare(output):
    p, _ = material()
    output = Path(output)
    require(output.is_absolute() and output.parent == ROOT / "results" and
            output.name.startswith(NAME + "_prepare_") and not output.exists() and not output.is_symlink(),
            "exclusive prepare path")
    value = native.raw(p)
    with output.open("xb") as f:
        require(f.write(value) == len(value), "prepare write")
        f.flush()
        os.fsync(f.fileno())
    native.paired.fsync_dir(output.parent)
    return {"path": str(output), "sha256": sha(output), "physical_authorized": False}


def verify(prepared, expected):
    native.actual_hash(expected)
    prepared = Path(prepared)
    require(prepared.is_absolute() and prepared.is_file() and not prepared.is_symlink() and
            sha(prepared) == expected, "prepared external SHA")
    saved = native.read(prepared, cap=32 * 1024**2)
    current, data = material()
    require(saved == current and sha(prepared) == expected, "prepare/source/code drift")
    return saved, data


class Store(native.base.Store):
    def __init__(self, path):
        self.path = Path(path)
        c = contract.contract()
        require(self.path.is_absolute() and self.path.parent == ROOT / "results" and
                self.path.name.startswith(NAME + "_") and not self.path.exists() and
                not self.path.is_symlink() and
                shutil.disk_usage(self.path.parent).free >= c["disk_floor_bytes"] + c["output_cap_bytes"],
                "exclusive bounded output and disk reserve")
        self.path.mkdir()
        native.paired.fsync_dir(self.path.parent)
        self.previous, self.index = None, 0

    def check(self, size=4096, final=False):
        c = contract.contract()
        files = list(self.path.iterdir())
        require(all(f.is_file() and not f.is_symlink() for f in files) and
                sum(f.stat().st_size for f in files) + size <=
                c["output_cap_bytes"] - (0 if final else c["final_reserve_bytes"]) and
                shutil.disk_usage(self.path).free >= c["disk_floor_bytes"] + size, "output budget")

    def write(self, name, value, append=False, final=False):
        require(type(name) is str and Path(name).name == name and name not in ("", ".", "..") and
                not (self.path / name).is_symlink(), "flat output name")
        self.check(len(value), final)
        with (self.path / name).open("ab" if append else "xb") as f:
            require(f.write(value) == len(value), "short output")
            f.flush()
            os.fsync(f.fileno())
        native.paired.fsync_dir(self.path)
        return sha(self.path / name)


class Meter:
    def __init__(self, store, started):
        self.store, self.started, self.where = store, started, {}
        self.receipts, self.secondary, self.fault = [], [], False
        self.counts = {k: dict(attempted=0, completed=0) for k in contract.contract()["caps"]}
        self.active = {k: 0 for k in self.counts}

    def call(self, kind, fn, *, cleanup=False, **fields):
        require(not cleanup or kind == "close", "only close may bypass fault")
        if not cleanup:
            require(not self.fault and time.monotonic() - self.started < contract.contract()["wall_seconds"],
                    "wall/fault predispatch")
            self.store.check()
        count = self.counts[kind]
        require(count["attempted"] < contract.contract()["caps"][kind], "resource cap")
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


def make_row(plan):
    return dict(index=plan["index"], root_slot=plan["root_slot"], condition=plan["condition"],
                arm=plan["arm"], role=plan["role"], task_id=plan["task_id"], plan=deepcopy(plan),
                status="not_started", first_error=None, secondary_errors=[],
                warmup_attempted=0, warmup_returned=0, own_attempted=0, own_returned=0, PNG=0,
                initial_exact=False, trajectory=[], checkpoints=[], missing_checkpoints=[0, 5, 10, 15],
                last_observation=None, responses=[], prefix51=dict(complete=False, frames=[]),
                source_unpaired_on_steps=[])


def census(rows, counts):
    c = {k: v["completed"] for k, v in counts.items()}
    require(all(v["attempted"] == v["completed"] for v in counts.values()) and
            c["constructor"] == c["close"] == 16 and c["reset"] == 32 and
            c["physics_step"] == 2 * c["reset"] + 10 * c["public_step"] and
            c["set_state"] == 2 * c["reset"] and c["goal_render"] == c["reset"] and
            c["goal_setter"] == c["reset"] + c["goal_bind"] and
            c["public_step"] == sum(r["warmup_returned"] + r["own_returned"] for r in rows) and
            c["explicit_render"] == sum(r["PNG"] for r in rows), "actual operation census")


def execute(p, data, store, factory, prepared, expected, *, authorization):
    require(authorization == AUTH and p["contract"] == contract.contract() and
            len(p["plans"]) == 16, "explicit fixed16 authorization and contract")
    meter = Meter(store, time.monotonic())
    rows, errors, seen = [make_row(plan) for plan in p["plans"]], [], []
    binding = dict(prepared=str(prepared), prepare_sha256=expected, join_sha256=JOIN_SHA)
    try:
        store.put("execution_binding.json", binding)
        store.emit("run_started", binding=binding, contract=contract.contract())
        for plan, row in zip(p["plans"], rows, strict=True):
            meter.where = dict(branch=row["index"], candidate_role=row["role"], **{k: row[k] for k in
                               ("root_slot", "condition", "arm")})
            env = None
            row["status"] = "running"
            try:
                env = meter.call("constructor", factory)
                require(all(env is not old and env.unwrapped is not old.unwrapped for old in seen), "fresh env")
                seen.append(env)
                native.validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size == 224 and
                        env.unwrapped.render_mode == "rgb_array", "renderer")
                with native.native_hooks(env, meter, row["secondary_errors"]):
                    episode.episode(env, data["roots"][plan["root_slot"]], data["source_plan"], plan, row, meter, store)
                require(not row["secondary_errors"], "hook cleanup")
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
    except BaseException as exc:
        errors.append(native.base.error("execution", exc))
    source_ok = False
    try:
        for path, digest in p["bindings"].items():
            require(sha(path) == digest, "final source hash")
        source_ok = True
    except BaseException as exc:
        errors.append(native.base.error("final_inputs", exc))
    errors.extend(meter.secondary)
    if not errors:
        try:
            census(rows, meter.counts)
        except BaseException as exc:
            errors.append(native.base.error("census", exc))
    complete = not errors and source_ok and all(r["status"] in ("horizon_completed", "native_stopped") for r in rows)
    status = "physical_recorded_not_independently_consumed" if complete else "partial_or_failed"
    store.emit("run_finished", status=status, counts=meter.counts, errors=errors)
    report = dict(schema=NAME + ".report", status=status, rows=rows, errors=errors,
                  contract=contract.contract(), counts=meter.counts, receipts=meter.receipts,
                  execution_binding=binding, journal_chain_tail=store.previous,
                  journal_sha256=sha(store.path / "journal.jsonl"),
                  inputs_final_verified=source_ok, wall_seconds=time.monotonic() - meter.started,
                  fixed_rows=16, P1_admission=False, independent_confirmation=False, new_neural_calls=0)
    store.put("report.json", report, final=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    else:
        require(args.authorize_physics, "explicit physics flag")
        p, data = verify(args.prepared, args.prepare_sha)
        native.constructor_contract()
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
        print("Executing all fixed known-root candidates and controls; no P1 admission.", flush=True)
        result = execute(p, data, Store(args.output), factory, args.prepared, args.prepare_sha,
                         authorization=AUTH)
    print(json.dumps(result, sort_keys=True))
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
