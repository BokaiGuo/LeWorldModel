"""Precommitted live PushT + real CUDA visual Handler development smoke."""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np

from . import p1_combined_guard_wiring_20261101_leibniz as combined
from . import p1_visual_guard_host_20261101_leibniz as source
from . import p1_total_guard_host_bridge_20261101_leibniz as bridge
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking

p, g, visual = combined.p, combined.g, combined.visual.visual
ROOT = source.ROOT
NAME = "p1_joint_live_visual_physical_smoke_20261101_leibniz"
DOC = ROOT.parent / "docs" / (NAME + ".md")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
AUTH = "AUTHORIZE_P1_JOINT_LIVE_SMOKE_HOST_CUDA"
PCAPS = dict(constructor=1, reset=1, set_state=2, goal_setter=2, goal_bind=0, goal_render=2,
             explicit_render=1, public_step=5, physics_step=52, close=1)
TCAPS = dict.fromkeys(g.AXES, 0)
TCAPS.update(model_load=1, source_encode_calls=1, source_encode_samples=1,
             reset_calls=1, render_calls=3, physical_dispatch_calls=5, physics_steps=52)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_value():
    value = source.prepare_value()
    paths = [Path(__file__).resolve(), DOC, TEST, Path(combined.__file__).resolve(),
             Path(combined.constructor.__file__).resolve(), Path(p.__file__).resolve(),
             Path(locking.__file__).resolve(), p.legacy.kernel.moving.ENV]
    g.require(sha(paths[-1]) == p.legacy.kernel.moving.ENV_SHA, "PushT source drift")
    sources = dict(value["sources"])
    for path in paths:
        h = sha(path)
        g.require(str(path) not in sources or sources[str(path)] == h, "source conflict")
        sources[str(path)] = h
    g.check_sources({"source_sha256": sources})
    return {"schema": NAME + ".prepare.v1", "sources": sources, "policy": value["policy"],
            "model_hash": value["model_hash"], "request_template": value["request_template"],
            "physical_caps": PCAPS, "total_caps": TCAPS, "seed": 0,
            "actions": [[0.0, 0.0] for _ in range(5)], "encoding_after_returned_steps": 5,
            "timeout_seconds": 120, "full_host_executed": False, "P1_admission": False}


def contract(value):
    identity = {k: g.digest({"local_smoke_only": k}) for k in g.IDENTITY}
    identity["model"] = value["model_hash"]["original_state_dict"]
    return {"schema": g.NAME + ".contract.v1", "run_id": NAME, "arm": "cem", "caps": value["total_caps"],
            "source_sha256": value["sources"], "identity": identity,
            "shared_initialization": "full_standalone_charge_each_arm", "hook_modules": g.HOOKS,
            "scope": "per_root_per_arm_start_before_load"}


class Store(visual.Store):
    def __init__(self, path):
        path = Path(path)
        g.require(path.is_absolute() and path.parent == ROOT / "results" and path.name.startswith(NAME + "_RUN_")
                  and not path.exists() and not path.is_symlink(), "exclusive output")
        super().__init__(path)


def validate_actions(rows):
    g.require(len(rows) == 5 and all(row["step"] == i + 1 and row["action"] == [0.0, 0.0]
              and row["actual_action"] == [0.0, 0.0] and not row["terminated"] and not row["truncated"]
              for i, row in enumerate(rows)), "five actual actions/returns required")


def run(value, output, prepared_sha):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from stable_worldmodel.envs.pusht.env import PushT
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments import checkpoint_compat, gpu_preflight

    g.require(Path(inspect.getsourcefile(PushT)).resolve() == p.legacy.kernel.moving.ENV
              and sha(p.legacy.kernel.moving.ENV) == p.legacy.kernel.moving.ENV_SHA, "native source")
    spec = gym.spec("swm/PushT-v1")
    g.require(spec.entry_point == "stable_worldmodel.envs.pusht.env:PushT"
              and spec.max_episode_steps is None and spec.additional_wrappers == (), "native lineage")
    with locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        store = Store(output)
        c = contract(value)
        guard = g.Guard(store.path / "total.jsonl", c, g.digest(c))
        pc = {"schema": p.NAME + ".contract.v1", "total_contract_sha256": g.digest(c), "caps": value["physical_caps"]}
        physical = p.PhysicalGuard(store.path / "physical.jsonl", guard, value["physical_caps"], g.digest(pc))
        meter = p.moving.Meter(store)
        meter.episode, meter.region = "joint_live", "setup"
        joint = combined.Combined(physical, meter, store)
        info = {"steps": [], "failure": None, "close_failure": None}
        host_info = before = after = response = observation = None
        started = time.monotonic()
        def check():
            g.require(time.monotonic() - started < value["timeout_seconds"], "smoke deadline")
            meter.check()
        try:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            host_info = gpu_preflight.require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                model = bridge.guarded_load(guard, lambda: checkpoint_compat.load_compatible_pretrained(
                    value["policy"], device=device).eval().requires_grad_(False))
                before = source.blind.host.signal.model_hash(model)
                g.require(before == value["model_hash"], "model hash")
                with joint.scope() as factory:
                    env = None
                    try:
                        cls = factory(PushT)
                        env = meter.call("constructor", lambda: gym.make(replace(spec, entry_point=cls),
                            render_mode="rgb_array", resolution=224, with_target=False, render_action=False))
                        obs, _ = env.reset(seed=value["seed"])
                        meter.region = "warmup"
                        for index, action in enumerate(value["actions"], 1):
                            check()
                            obs, reward, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
                            info["steps"].append({"step": index, "action": action,
                                "actual_action": np.asarray(env.unwrapped.latest_action).tolist(),
                                "state": np.asarray(obs["state"]).tolist(), "reward": float(reward),
                                "terminated": bool(terminated), "truncated": bool(truncated)})
                            if terminated or truncated:
                                break
                        validate_actions(info["steps"])
                        rgb = np.asarray(env.render()).copy()
                        g.require(rgb.dtype == np.uint8 and rgb.shape == (224, 224, 3), "fresh RGB")
                        name = "joint_live_plan_05.png"
                        png_sha = store.frame(name, rgb)
                        observation = {"step": 5, "path": str(store.path / name), "file_sha256": png_sha,
                                       "pixel_sha256": visual.array_sha(rgb),
                                       "state": np.asarray(obs["state"]).tolist()}
                        request = deepcopy(value["request_template"])
                        request.update(episode_id="joint_live", observed_step=5,
                                       png={k: observation[k] for k in ("path", "file_sha256", "pixel_sha256")})
                        request["request_sha256"] = visual.source.digest(
                            {k: v for k, v in request.items() if k != "request_sha256"})
                        store.put("joint_live_request.json", request)
                        meter.region = "encoding"
                        def encode():
                            sink = visual.Sink(store, meter, "joint_live", "encode")
                            result = visual.Handler(None).encode_observation(request, sink, model=model, device=device,
                                                                           check_limits=check)
                            sink.complete()
                            return result
                        response = joint._visual(encode, meter, store)
                        store.load_array(response["latent"])
                        after = source.blind.host.signal.model_hash(model)
                        g.require(before == after, "model drift")
                    finally:
                        if env is not None:
                            try:
                                meter.region = "closing"
                                meter.call("close", env.close)
                            except BaseException as exc:
                                info["close_failure"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
                                raise
        except BaseException as exc:
            guard.failed = True
            info["failure"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        pr = physical.close()
        tr = guard.close()
        pv = p.verify_journal(pr["path"], pr["sha256"], g.digest(pc))
        tv = g.verify_journal(tr["journal"], tr["sha256"], g.digest(c))
        ok = (info["failure"] is None and info["close_failure"] is None and response is not None
              and pv["status"] == tv["status"] == "closed_unverified")
        report = {"schema": NAME + ".report.v1", "status": "complete" if ok else "partial_or_failed",
                  "prepare_sha256": prepared_sha, "host": host_info, "result": info, "observation": observation,
                  "response": response, "model_before": before, "model_after": after,
                  "physical_receipt": pr, "physical_audit": pv, "total_receipt": tr, "total_audit": tv,
                  "legacy_pending": list(meter.pending.values()), "legacy_secondary": meter.secondary,
                  "full_host_executed": False, "coverage_verified": False, "full_cost_certified": False,
                  "fairness_proven": False, "P1_admission": False}
        report_sha = source.blind.write_new(store.path / "report.json", report)
        return {"status": report["status"], "report": str(store.path / "report.json"), "sha256": report_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--prepare", required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--output")
    parser.add_argument("--authorization")
    args = parser.parse_args()
    path = Path(args.prepare)
    g.require(path.is_absolute() and path.parent == ROOT / "results" and path.name.startswith(NAME + "_PREP_"),
              "prepare path")
    if args.command == "prepare":
        result = {"path": str(path), "sha256": source.blind.write_new(path, prepare_value())}
    else:
        g.require(args.authorization == AUTH, "explicit host CUDA authorization")
        g.require(args.sha256 and sha(path) == args.sha256, "prepare hash")
        value = json.loads(path.read_bytes())
        g.require(value == prepare_value(), "prepare/source drift")
        result = run(value, args.output, args.sha256)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result.get("status") == "partial_or_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
