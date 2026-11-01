"""Fixed 336-branch post-MAIN02 exploration; explicit physical authorization only."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from experiments.spectral_reachability import p1_matched_canvas_incident_audit_20261030_codex as proof
from experiments.spectral_reachability import p1_matched_suffix_scale_20261030 as old

NAME = "p1_matched_censored_runner_20261030_codex"
AUTH = "MAIN_APPROVED_CENSORED_FIXED336_PHYSICS"
runner, prior, require = old.runner, old.runner.prior, old.require
base, native = prior.base, runner.physical.native
ROOT = old.ROOT
CONSUMER = Path(__file__).with_name("p1_matched_censored_consumer_20261030_codex.py")
PROTOCOL = Path(__file__).with_name("p1_matched_censored_protocol_20261030_codex.md")


def source_files():
    return [Path(__file__).resolve(), CONSUMER.resolve(), PROTOCOL.resolve(), Path(proof.__file__).resolve()]


def build():
    """Rebuild fixed old input and pin the full bytes of each new source."""
    p = old.build()
    require(p["fixed_anchors"] == 16 and p["fixed_branches"] == 336
            and len(p["anchors"]) == len(p["probe_sets"]) == 16, "fixed roster")
    for i, (anchor, bank) in enumerate(zip(p["anchors"], p["probe_sets"], strict=True)):
        require(anchor["index"] == i and len(bank) == 21
                and [b["index"] for b in bank] == list(range(21)), "fixed ordered roster")
    bindings = copy.deepcopy(p["bindings"])
    for path in source_files():
        require(path.is_file() and not path.is_symlink(), "new source exists")
        bindings[str(path)] = runner.physical.sha(path)
    for path, digest in bindings.items():
        require(runner.physical.sha(path) == digest, "input/source drift")
    return dict(schema=NAME + ".prepare.v1", original=p, bindings=bindings,
                fixed_anchors=16, probes_per_anchor=21, fixed_branches=336,
                policy="probe guard censor only on proven finite canvas violation",
                independent_confirmation=False, P1_admission=False, physical_authorized=False)


def _write_new(path, value):
    path = Path(path)
    require(path.is_absolute() and path.parent.resolve() == path.parent and not path.is_symlink(),
            "canonical exclusive output")
    data = native.raw(value)
    with path.open("xb") as f:
        require(f.write(data) == len(data), "complete output write")
        f.flush()
        os.fsync(f.fileno())
    prior.base.paired.fsync_dir(path.parent)
    return runner.physical.sha(path)


def prepare(output):
    path = Path(output)
    require(path.parent == ROOT / "results" and path.name.startswith(NAME + "_prepare_"), "prepare namespace")
    return _write_new(path, build())


def verify_prepare(path, expected):
    native.actual_hash(expected)
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file()
            and runner.physical.sha(path) == expected, "external prepare SHA")
    p = native.read(path, cap=64 * 1024**2)
    require(p == build() and runner.physical.sha(path) == expected, "prepare/source rebuild")
    return p


class ObservedStore:
    """Mirror events already durably emitted by the existing store."""
    def __init__(self, store):
        self.store, self.events = store, []

    def __getattr__(self, name):
        return getattr(self.store, name)

    def emit(self, kind, **fields):
        index = self.store.index
        self.store.emit(kind, **fields)
        self.events.append(dict(index=index, kind=kind, **copy.deepcopy(fields)))


def _guard_origin(exc):
    tb = exc.__traceback__
    found = False
    while tb:
        found |= tb.tb_frame.f_code is base.moving.guard.__code__
        tb = tb.tb_next
    return found


def censored_proof(store, anchor, probe, row, exc):
    """Validate the exact just-returned transition and its original first frame."""
    require(type(exc) is base.moving.Rejected and str(exc) == "fresh RGB/canvas guard"
            and _guard_origin(exc) and not row["secondary_errors"], "original guard rejection required")
    rec = row["last_observation"]
    path = store.path / rec["png"]
    require(path.parent == store.path and path.resolve() == path and path.is_file() and not path.is_symlink()
            and runner.physical.sha(path) == rec["file_sha256"], "retained original frame")
    with Image.open(path) as image:
        pixels = np.asarray(image)
    trial = copy.deepcopy(row)
    trial["status"] = "engineering_failed"
    trial["error"] = dict(stage="branch", type="Rejected", message="fresh RGB/canvas guard")
    result = proof.incident(store.events, anchor, probe, trial, pixels)
    return {k: result[k] for k in ("anchor", "probe", "step", "action_float32_hex",
                                   "returned_state", "out_of_bounds", "frame_sha256", "blue_pixels")}


def episode(env, root, source_plan, plan, anchor, probe, row, meter, store):
    trace = prior.TraceStore(store)
    body = env.unwrapped
    packet = root["packets"][plan["condition"]]
    obs, native_goal = base.place(env, source_plan, root["slot"], meter, trace)
    frames = [base.capture(env, obs, "warmup", 0, row, meter, trace, root["warm"][0])]
    for step, command in enumerate(base.moving.warmup_commands(), 1):
        require(not body.eval_state(native_goal, base.source.vector(obs["state"]))[0], "warmup terminal")
        obs, info, t, tr, reward, actual = base.dispatch(env, command, "warmup", step, row, meter, trace, native_goal)
        image = base.capture(env, obs, "warmup", step, row, meter, trace, root["warm"][step])
        base.paired.check_return(env, command, actual, info, native_goal, reward, t)
        require(not t and not tr, "warmup stop")
        if step in (5, 10):
            frames.append(image)
    require(base.moving.exact(np.asarray(frames), packet["rgb_history"])
            and base.moving.exact(packet["raw_action_history"], base.moving.warmup_commands()), "history")
    goal = base.source.vector(root["goal"])
    require(base.source.root_hash(base.source.vector(obs["state"]), goal) == root["root_sha256"], "root")
    base.bind_goal(body, goal, packet["goal_rgb"], meter, trace)
    bound = base.invariant(body)
    base.capture(env, obs, "plan", 0, row, meter, trace)
    require(not body.eval_state(goal, base.source.vector(obs["state"]))[0], "root terminal")
    for step, command in enumerate(base.previous.commands(plan["commands"])[:5], 1):
        obs, info, t, tr, reward, actual = base.dispatch(env, command, "plan", step, row, meter, trace, goal)
        base.capture(env, obs, "plan", step, row, meter, trace)
        base.paired.check_return(env, command, actual, info, goal, reward, t)
        require(not t and not tr, "prefix stop")
    prior.compare_trace(row, anchor, trace, meter)
    def check_bound(info=None):
        now = base.invariant(body)
        require({k: v for k, v in now.items() if k != "snapshot"} ==
                {k: v for k, v in bound.items() if k != "snapshot"}, "goal/renderer drift")
        if info is not None:
            require(base.moving.exact(np.asarray(info["goal"]), packet["goal_rgb"]), "goal RGB drift")
    check_bound()
    start = base.source.vector(obs["state"]).copy()
    row["start_state"], row["start_verified"] = start.tolist(), True
    commands = np.frombuffer(bytes.fromhex(probe["raw5_float32_hex"]), dtype="<f4").reshape(5, 2)
    for step, command in enumerate(commands, 1):
        obs, info, t, tr, reward, actual = base.dispatch(env, command, "probe", step, row, meter, trace, goal)
        try:
            base.capture(env, obs, "probe", step, row, meter, trace)
        except base.moving.Rejected as exc:
            # Do not skip the checks that the old episode normally runs after capture.
            base.paired.check_return(env, command, actual, info, goal, reward, t)
            check_bound(info)
            evidence = censored_proof(store, anchor, probe, row, exc)
            row["status"], row["censor"] = "censored_out_of_domain", evidence
            store.emit("censored_out_of_domain", **meter.where, proof=evidence)
            return
        base.paired.check_return(env, command, actual, info, goal, reward, t)
        check_bound(info)
        state = base.source.vector(obs["state"]); delta = state - start
        row["responses"].append(dict(step=step, state=state.tolist(), delta_native7=delta.tolist(),
                                     delta_angle_wrapped=float(np.arctan2(np.sin(delta[4]), np.cos(delta[4]))),
                                     terminated=t, truncated=tr))
        if t or tr:
            row["status"] = "native_stopped"
            return
    row["status"] = "probe_completed"


def execute_shard(anchor, probes, data, output, factory, bindings):
    raw_store = runner.Store(output)
    store = ObservedStore(raw_store)
    rows = [runner.row_for(anchor, probe) for probe in probes]
    meter, errors, seen = runner.Meter(store), [], []
    prefix = np.frombuffer(bytes.fromhex(anchor["prefix5_float32_hex"]), dtype="<f4").reshape(5, 2)
    plan = dict(condition=anchor["condition"], commands=np.concatenate([prefix, np.zeros((5, 2), np.float32)]).tolist())
    try:
        store.emit("run_started", anchor=anchor["index"], fixed_rows=21, caps=runner.CAPS)
        for row, probe in zip(rows, probes, strict=True):
            meter.where = dict(branch=row["index"], anchor_index=anchor["index"], condition=row["condition"],
                               arm=row["arm"], candidate_role=row["candidate_role"], probe=row["probe"])
            env = None
            try:
                env = meter.call("constructor", factory)
                require(all(env is not e and env.unwrapped is not e.unwrapped for e in seen), "fresh env")
                seen.append(env)
                base.validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size == 224
                        and env.unwrapped.render_mode == "rgb_array", "renderer")
                with base.native_hooks(env, meter, row["secondary_errors"]):
                    episode(env, data["roots"][row["root_slot"]], data["source_plan"], plan,
                            anchor, probe, row, meter, store)
                require(not row["secondary_errors"], "hook cleanup")
            except BaseException as exc:
                row.update(status="engineering_failed", error=base.error("branch", exc))
                errors.append(dict(index=row["index"], **row["error"]))
            finally:
                if env is not None:
                    try:
                        meter.call("close", env.close, cleanup=True)
                    except BaseException as exc:
                        errors.append(base.error("close", exc))
            store.emit("branch_closed", branch=row["index"], status=row["status"])
            if errors or row["secondary_errors"]:
                break
        for path, digest in bindings.items():
            require(runner.physical.sha(path) == digest, "source drift")
        if not errors:
            c = {k: v["completed"] for k, v in meter.counts.items()}
            require(all(v["attempted"] == v["completed"] for v in meter.counts.values())
                    and c["constructor"] == c["close"] == 21 and c["reset"] == 42
                    and c["physics_step"] == 2*c["reset"] + 10*c["public_step"]
                    and c["set_state"] == 2*c["reset"] and c["goal_bind"] == 21
                    and c["goal_render"] == c["reset"] and c["goal_setter"] == c["reset"]+c["goal_bind"]
                    and c["public_step"] == sum(r["warmup_returned"]+r["own_returned"] for r in rows)
                    and c["explicit_render"] == sum(r["PNG"] for r in rows), "resource census")
    except BaseException as exc:
        errors.append(base.error("execution", exc))
    status = "recorded_not_consumed" if not errors and all(
        r["status"] in ("probe_completed", "native_stopped", "censored_out_of_domain") for r in rows) else "partial_or_failed"
    store.emit("run_finished", status=status, counts=meter.counts, errors=errors)
    report = dict(schema=NAME + ".shard.v1", status=status, anchor=anchor["index"], rows=rows,
                  counts=meter.counts, receipts=meter.receipts, errors=errors,
                  journal_chain_tail=store.previous, journal_sha256=runner.physical.sha(store.path / "journal.jsonl"),
                  wall_seconds=time.monotonic()-meter.started, fixed_rows=21, P1_admission=False)
    store.put("report.json", report, final=True)
    return report


def run(prepared, expected, output, *, authorization):
    require(authorization == AUTH, "explicit MAIN physical authorization")
    p = verify_prepare(prepared, expected)
    source_report = native.read(old.prep.DIRECTORY / "report.json", cap=64*1024**2)
    _, data = runner.physical.verify(source_report["execution_binding"]["prepared"], old.prep.PREP_SHA)
    native.constructor_contract()
    directory = Path(output)
    require(directory.is_absolute() and directory.parent == ROOT / "results"
            and directory.name.startswith(NAME + "_run_") and not directory.exists()
            and not directory.is_symlink() and shutil.disk_usage(directory.parent).free >= runner.FLOOR+16*runner.CAP,
            "exclusive full-run reservation")
    directory.mkdir()
    prior.base.paired.fsync_dir(directory.parent)
    bindings = copy.deepcopy(p["bindings"])
    bindings[str(Path(prepared))] = expected
    binding_sha = _write_new(directory / "run_binding.json",
                             dict(prepared=str(prepared), prepare_sha256=expected, bindings=bindings,
                                  fixed_branches=336, independent_confirmation=False, P1_admission=False))
    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
    shards = []
    for anchor, bank in zip(p["original"]["anchors"], p["original"]["probe_sets"], strict=True):
        path = directory / f"anchor_{anchor['index']:02d}"
        report = execute_shard(anchor, bank, data, path, factory, bindings)
        shards.append(dict(anchor=anchor["index"], report_sha256=runner.physical.sha(path / "report.json"),
                           status=report["status"]))
        if report["status"] != "recorded_not_consumed":
            break
    for path, digest in bindings.items():
        require(runner.physical.sha(path) == digest, "final source drift")
    result = dict(schema=NAME + ".run.v1", status="recorded_not_consumed" if len(shards) == 16
                  and all(s["status"] == "recorded_not_consumed" for s in shards) else "partial_or_failed",
                  shards=shards, run_binding_sha256=binding_sha, fixed_branches=336,
                  independent_confirmation=False, P1_admission=False)
    digest = _write_new(directory / "run_report.json", result)
    return dict(sha256=digest, status=result["status"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare", "run"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--prepared")
    ap.add_argument("--sha256")
    ap.add_argument("--authorization")
    a = ap.parse_args()
    if a.command == "prepare":
        require(a.prepared is None and a.sha256 is None and a.authorization is None, "prepare arguments")
        digest = prepare(a.output)
        result = dict(sha256=digest, status="prepared", P1_admission=False, independent_confirmation=False)
    else:
        require(a.prepared and a.sha256, "run arguments")
        result = dict(run(a.prepared, a.sha256, a.output, authorization=a.authorization),
                      P1_admission=False, independent_confirmation=False)
    print(json.dumps(result))
    return 0 if result["status"] in ("prepared", "recorded_not_consumed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
