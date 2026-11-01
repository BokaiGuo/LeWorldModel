"""Execute fixed 16 x 21 known-root physical probes; no neural inference.

Each anchor uses a separate exclusive shard, fixed order, and no retries.
Results require an independent saved audit before scientific interpretation.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import moving_overlay_capability_probe_physical_20261027_v1 as prior
from experiments.spectral_reachability import p1_backtrack_capability_prepare_20261030 as prep
from experiments.spectral_reachability import p1_backtrack_physical_runner_20261030 as physical

NAME = "p1_backtrack_capability_runner_20261030"
require = physical.require
CAP, FLOOR, FINAL = 256 * 1024**2, 512 * 1024**2, 16 * 1024**2
WALL = 1800
CAPS = dict(constructor=21, close=21, reset=42, public_step=420, physics_step=4284,
            set_state=84, goal_setter=63, goal_render=42, goal_bind=21, explicit_render=1386)


class Store(prior.base.Store):
    def __init__(self, path):
        self.path = path
        require(not path.exists() and not path.is_symlink() and
                shutil.disk_usage(path.parent).free >= CAP + FLOOR, "exclusive shard and disk reserve")
        path.mkdir()
        prior.base.paired.fsync_dir(path.parent)
        self.previous, self.index = None, 0

    def check(self, size=4096, final=False):
        files = list(self.path.iterdir())
        require(all(f.is_file() and not f.is_symlink() for f in files) and
                sum(f.stat().st_size for f in files) + size <= CAP - (0 if final else FINAL) and
                shutil.disk_usage(self.path).free >= FLOOR + size, "shard cap")

    def write(self, name, value, append=False, final=False):
        require(type(name) is str and Path(name).name == name and name not in ("", ".", "..") and
                not (self.path / name).is_symlink(), "flat file")
        self.check(len(value), final)
        with (self.path / name).open("ab" if append else "xb") as stream:
            require(stream.write(value) == len(value), "complete write")
            stream.flush()
            os.fsync(stream.fileno())
        prior.base.paired.fsync_dir(self.path)
        return physical.sha(self.path / name)


class Meter(physical.Meter):
    def __init__(self, store):
        super().__init__(store, time.monotonic())
        self.counts = {key: dict(attempted=0, completed=0) for key in CAPS}
        self.active = {key: 0 for key in CAPS}

    def call(self, kind, fn, *, cleanup=False, **fields):
        require(not cleanup or kind == "close", "only close may bypass fault")
        if not cleanup:
            require(not self.fault and time.monotonic() - self.started < WALL, "wall or journal fault")
            self.store.check()
        count = self.counts[kind]
        require(count["attempted"] < CAPS[kind], "per-shard operation cap")
        receipt = dict(index=len(self.receipts), resource=kind, **deepcopy(self.where), **deepcopy(fields),
                       status="pending")
        try:
            self.store.emit("operation_dispatch_intent", receipt=receipt)
        except BaseException:
            self.fault = True
            if not cleanup:
                raise
        self.receipts.append(receipt)
        count["attempted"] += 1
        self.active[kind] += 1
        try:
            result = fn()
            count["completed"] += 1
            receipt["status"] = "returned"
        except BaseException as exc:
            receipt.update(status="failed", error=prior.base.error(kind, exc))
            try:
                self.store.emit("operation_failed", receipt=receipt)
            except BaseException:
                self.fault = True
            raise
        finally:
            self.active[kind] -= 1
        try:
            self.store.emit("operation_returned", receipt=receipt)
        except BaseException:
            self.fault = True
            if not cleanup:
                raise
        return result


def row_for(anchor, probe):
    return dict(index=probe["index"], anchor_index=anchor["index"], root_slot=anchor["root_slot"],
                condition=anchor["condition"], arm=anchor["arm"], candidate_role=anchor["role"],
                task_id=anchor["task_id"], probe=probe["name"], status="not_started",
                start_verified=False, start_state=None, responses=[], trajectory=[], checkpoints=[],
                missing_checkpoints=[0, 5, 10], PNG=0, warmup_returned=0, own_attempted=0,
                own_returned=0, secondary_errors=[])


def execute_shard(anchor, probes, data, output, factory, bindings):
    store, rows = Store(output), [row_for(anchor, probe) for probe in probes]
    meter = Meter(store)
    errors, seen = [], []
    prefix = np.frombuffer(bytes.fromhex(anchor["prefix5_float32_hex"]), dtype="<f4").reshape(5, 2)
    plan = dict(condition=anchor["condition"], commands=np.concatenate([prefix, np.zeros((5, 2), np.float32)]).tolist())
    try:
        store.emit("run_started", anchor=anchor["index"], fixed_rows=21, caps=CAPS)
        for row, probe in zip(rows, probes, strict=True):
            meter.where = dict(branch=row["index"], anchor_index=anchor["index"],
                               condition=row["condition"], arm=row["arm"], candidate_role=row["candidate_role"],
                               probe=row["probe"])
            env = None
            try:
                env = meter.call("constructor", factory)
                require(all(env is not item and env.unwrapped is not item.unwrapped for item in seen), "fresh env")
                seen.append(env)
                prior.base.validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size == 224 and
                        env.unwrapped.render_mode == "rgb_array", "renderer")
                probe_actions = np.frombuffer(bytes.fromhex(probe["raw5_float32_hex"]),
                                              dtype="<f4").reshape(5, 2).copy()
                probe_data = dict(name=probe["name"], commands=probe_actions.tolist())
                with prior.base.native_hooks(env, meter, row["secondary_errors"]):
                    prior.episode(env, data["roots"][row["root_slot"]], data["source_plan"],
                                  plan, anchor, probe_data, row, meter, store)
                require(not row["secondary_errors"], "hook cleanup")
            except BaseException as exc:
                row.update(status="engineering_failed", error=prior.base.error("branch", exc))
                errors.append(dict(index=row["index"], **row["error"]))
            finally:
                if env is not None:
                    try:
                        meter.call("close", env.close, cleanup=True)
                    except BaseException as exc:
                        errors.append(prior.base.error("close", exc))
            store.emit("branch_closed", branch=row["index"], status=row["status"])
            if errors or row["secondary_errors"]:
                break
        for path, digest in bindings.items():
            require(physical.sha(path) == digest, "postrun source/code drift")
        if not errors:
            c = {k: v["completed"] for k, v in meter.counts.items()}
            require(all(v["attempted"] == v["completed"] for v in meter.counts.values()) and
                    c["constructor"] == c["close"] == 21 and c["reset"] == 42 and
                    c["physics_step"] == 2 * c["reset"] + 10 * c["public_step"] and
                    c["set_state"] == 2 * c["reset"] and c["goal_bind"] == 21 and
                    c["goal_render"] == c["reset"] and c["goal_setter"] == c["reset"] + c["goal_bind"] and
                    c["public_step"] == sum(r["warmup_returned"] + r["own_returned"] for r in rows) and
                    c["explicit_render"] == sum(r["PNG"] for r in rows), "shard operation census")
    except BaseException as exc:
        errors.append(prior.base.error("execution", exc))
    status = "recorded_not_independently_consumed" if not errors and all(
        row["status"] in ("probe_completed", "native_stopped") for row in rows) else "partial_or_failed"
    store.emit("run_finished", status=status, counts=meter.counts, errors=errors)
    report = dict(schema=NAME + ".shard", status=status, anchor=anchor["index"], rows=rows,
                  counts=meter.counts, receipts=meter.receipts, errors=errors,
                  journal_chain_tail=store.previous, journal_sha256=physical.sha(store.path / "journal.jsonl"),
                  wall_seconds=time.monotonic() - meter.started, fixed_rows=21,
                  new_neural_calls=0, P1_admission=False)
    store.put("report.json", report, final=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepare-sha", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    require(args.authorize_physics, "explicit physics authorization")
    source_path = Path(args.prepared)
    require(physical.sha(source_path) == args.prepare_sha, "external prepare SHA")
    p = physical.native.read(source_path, cap=32 * 1024**2)
    require(p == prep.build() and physical.sha(source_path) == args.prepare_sha, "complete preparation replay")
    require(len(p["anchors"]) == 16 and len(p["probes"]) == 21 and len(p["schedule"]) == 336,
            "full fixed roster")
    source_report = physical.native.read(prep.DIRECTORY / "report.json", cap=64 * 1024**2)
    _, data = physical.verify(source_report["execution_binding"]["prepared"], prep.PREP_SHA)
    prior.base.hash_value(args.prepare_sha)
    physical.native.constructor_contract()
    directory = Path(args.output)
    require(directory.is_absolute() and directory.parent == physical.ROOT / "results" and
            directory.name.startswith(NAME + "_") and not directory.exists() and not directory.is_symlink() and
            shutil.disk_usage(directory.parent).free >= FLOOR + 16 * CAP, "exclusive full-run reservation")
    directory.mkdir()
    prior.base.paired.fsync_dir(directory.parent)
    bindings = deepcopy(p["bindings"])
    bindings[str(source_path)] = args.prepare_sha
    for file in (Path(__file__), Path(prior.__file__)):
        bindings[str(file.resolve())] = physical.sha(file)
    with (directory / "run_binding.json").open("xb") as stream:
        body = physical.native.raw(dict(prepare_path=str(source_path), prepare_sha256=args.prepare_sha,
                                        source_code_bindings=bindings, fixed_rows=336, caps_per_shard=CAPS))
        require(stream.write(body) == len(body), "complete run binding")
        stream.flush()
        os.fsync(stream.fileno())
    prior.base.paired.fsync_dir(directory)
    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
    print("Fixed 16 x 21 same-prefix physical probes, no neural inference, no P1 admission.", flush=True)
    for anchor in p["anchors"]:
        shard = directory / f"anchor_{anchor['index']:02d}"
        report = execute_shard(anchor, p["probes"], data, shard, factory, bindings)
        print(json.dumps(dict(anchor=anchor["index"], status=report["status"],
                              report_sha256=physical.sha(shard / "report.json"))), flush=True)
        if report["status"] != "recorded_not_independently_consumed":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
