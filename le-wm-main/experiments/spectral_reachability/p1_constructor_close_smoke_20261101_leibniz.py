"""Bounded CPU-only real PushT constructor-close development smoke."""
import argparse
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np

from . import p1_constructor_close_guard_20261101_leibniz as adapter

p, g = adapter.physical, adapter.physical.total
ROOT = p.legacy.kernel.moving.ROOT
NAME = "p1_constructor_close_smoke_20261101_leibniz"
DOC = ROOT.parent / "docs" / (NAME + ".md")
TEST = ROOT / "tests/test_p1_constructor_close_guard_20261101_leibniz.py"
CAPS = dict(constructor=1, reset=1, set_state=2, goal_setter=2, goal_bind=0,
            goal_render=2, explicit_render=1, public_step=1, physics_step=12, close=1)
TOTAL_CAPS = dict.fromkeys(g.AXES, 0)
TOTAL_CAPS.update(reset_calls=1, render_calls=3, physical_dispatch_calls=1, physics_steps=12)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs():
    env = p.legacy.kernel.moving.ENV
    g.require(sha(env) == p.legacy.kernel.moving.ENV_SHA, "native PushT source drift")
    files = [Path(m.__file__).resolve() for m in (adapter, p, g, p.legacy, p.moving)]
    files += [Path(__file__).resolve(), DOC, TEST, env]
    return {str(f): sha(f) for f in files}


def preparation():
    return {"schema": NAME + ".prepare.v1", "sources": inputs(), "physical_caps": CAPS,
            "total_caps": TOTAL_CAPS, "cases": ["normal", "injected_after_native_init"],
            "normal": {"seed": 0, "action": [0.0, 0.0], "public_steps": 1, "render": 1},
            "wall_seconds_per_case": 30, "neural": False, "P1_admission": False}


def write_new(path, obj):
    data = g.canonical(obj) + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


class Store:
    def __init__(self):
        self.started, self.events = time.monotonic(), []
    def check(self):
        g.require(time.monotonic() - self.started < 30, "case wall cap")
    def emit(self, kind, **kwargs):
        self.events.append({"kind": kind, **kwargs})


def contract(value, case):
    return {"schema": g.NAME + ".contract.v1", "run_id": NAME + "_" + case, "arm": "cem",
            "caps": value["total_caps"], "identity": {k: g.digest({"smoke": k}) for k in g.IDENTITY},
            "source_sha256": value["sources"], "shared_initialization": "full_standalone_charge_each_arm",
            "hook_modules": g.HOOKS, "scope": "per_root_per_arm_start_before_load"}


def run_case(output, value, case):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from stable_worldmodel.envs.pusht.env import PushT
    g.require(Path(inspect.getsourcefile(PushT)).resolve() == p.legacy.kernel.moving.ENV, "PushT source path")
    spec = gym.spec("swm/PushT-v1")
    g.require(spec.entry_point == "stable_worldmodel.envs.pusht.env:PushT"
              and spec.max_episode_steps is None and spec.additional_wrappers == (), "PushT lineage")
    c = contract(value, case)
    guard = g.Guard(output / (case + "_total.jsonl"), c, g.digest(c))
    pc = {"schema": p.NAME + ".contract.v1", "total_contract_sha256": g.digest(c), "caps": value["physical_caps"]}
    physical = p.PhysicalGuard(output / (case + "_physical.jsonl"), guard, value["physical_caps"], g.digest(pc))
    store, failure, info, env = Store(), None, {}, None
    meter = p.moving.Meter(store)
    class Injected(PushT):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            raise RuntimeError("planned injection after actual PushT constructor")
    try:
        with adapter.guarded_class_factory(physical, meter) as factory:
            cls = factory(PushT if case == "normal" else Injected)
            try:
                env = meter.call("constructor", lambda: gym.make(replace(spec, entry_point=cls),
                                  render_mode="rgb_array", resolution=224, with_target=False, render_action=False))
                obs, _ = env.reset(seed=0)
                image = env.render()
                meter.region = "warmup"
                obs, reward, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
                info = {"state_sha256": hashlib.sha256(np.asarray(obs["state"]).tobytes()).hexdigest(),
                        "rgb_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
                        "rgb_shape": list(image.shape), "reward": float(reward),
                        "terminated": bool(terminated), "truncated": bool(truncated)}
            except BaseException as exc:
                failure = {"type": type(exc).__name__, "message": str(exc)[:500]}
            finally:
                if env is not None:
                    try:
                        meter.call("close", env.close)
                    except BaseException as exc:
                        info["close_failure"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
    except BaseException as exc:
        failure = failure or {"type": type(exc).__name__, "message": str(exc)[:500]}
    pc_receipt = physical.close()
    t_receipt = guard.close()
    pv = p.verify_journal(pc_receipt["path"], pc_receipt["sha256"], g.digest(pc))
    tv = g.verify_journal(t_receipt["journal"], t_receipt["sha256"], g.digest(c))
    return {"case": case, "failure": failure, "info": info, "secondary": meter.secondary,
            "legacy_counts": meter.counts, "legacy_pending": list(meter.pending.values()),
            "legacy_events": store.events, "physical": pc_receipt, "total": t_receipt,
            "verified_physical": pv, "verified_total": tv}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--prepare", required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--output")
    args = parser.parse_args()
    prep = Path(args.prepare)
    g.require(prep.is_absolute() and prep.parent == ROOT / "results" and prep.name.startswith(NAME + "_PREP_"),
              "prepare path")
    if args.command == "prepare":
        print(json.dumps({"path": str(prep), "sha256": write_new(prep, preparation())}), flush=True)
        return
    output = Path(args.output or "")
    g.require(output.is_absolute() and output.parent == ROOT / "results" and output.name.startswith(NAME + "_RUN_")
              and not output.exists() and not output.is_symlink(), "exclusive output")
    g.require(args.sha256 and sha(prep) == args.sha256 and json.loads(prep.read_bytes()) == preparation(),
              "prepare/source drift")
    output.mkdir()
    rows = [run_case(output, preparation(), case) for case in ("normal", "injected_after_native_init")]
    normal, injected = rows
    ok = (normal["failure"] is None and normal["physical"]["status"] == "closed_unverified"
          and injected["failure"] == {"type": "RuntimeError", "message": "planned injection after actual PushT constructor"}
          and injected["verified_physical"]["charged"]["close"] == 1 and not injected["secondary"]
          and injected["physical"]["status"] == "failed_or_blocked")
    report = {"schema": NAME + ".report.v1", "status": "complete" if ok else "partial_or_failed",
              "prepare_sha256": args.sha256, "rows": rows, "neural": False, "P1_admission": False,
              "full_host_executed": False, "coverage_verified": False, "fairness_proven": False}
    report_sha = write_new(output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(output / "report.json"), "sha256": report_sha}),
          flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
