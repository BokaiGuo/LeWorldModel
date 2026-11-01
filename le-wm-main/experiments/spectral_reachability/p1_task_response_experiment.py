"""Presealed development factorial: tail coverage x coordinate task metric."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import p1_dev303_dynamics_cp5_v13_cem_20261102_exclusive as source
from . import p1_task_response_revision as revision

ROOT = source.ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def prepare(output):
    previous = source.protocol()
    bindings = dict(previous["bindings"])
    bindings.update({str(Path(p).resolve()): sha(p) for p in (__file__, revision.__file__)})
    protocol = dict(schema="p1_task_response_factorial.v1", source=previous,
                    bindings=bindings, objectives=["latent", "object_tangent"],
                    searches=["original", "coherent"], variants=5, samples=64, iterations=5,
                    object_weight=1., pusher_weight=.05,
                    classification="posthoc_development_factorial_not_confirmation",
                    evaluation="report unchanged native success/reward and separate object/pusher errors",
                    stopping="execute all 16 predetermined arms; no sample additions",
                    P1_admission=False, full_cost_certified=False)
    output.mkdir()
    write(output / "protocol.json", protocol)
    print(sha(output / "protocol.json"), flush=True)


def load(output, expected):
    if sha(output / "protocol.json") != expected:
        raise ValueError("protocol SHA")
    p = json.loads((output / "protocol.json").read_text())
    if any(sha(path) != value for path, value in p["bindings"].items()):
        raise ValueError("source drift")
    return p


def plan(output, expected):
    protocol = load(output, expected)
    if (output / "plans.json").exists():
        raise FileExistsError("sealed plans already exist")
    from .e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from . import preflight_pusht_d2p_p0 as p0
    from . import p1_rgb_tangent_metric_20261101_main as tangent
    from PIL import Image

    with gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        model = load_compatible_pretrained(protocol["source"]["policy"], device=device).eval().requires_grad_(False)
        before = source.engine.base.blind.host.signal.model_hash(model)
        if before != protocol["source"]["model_hash"]:
            raise ValueError("frozen checkpoint mismatch")
        task = protocol["source"]["task"]
        tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
        mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
        counts, handles, rows = {}, [], []
        phase = ["goal"]
        for name, module in (("encoder", model.encoder), ("projector", model.projector),
                             ("action_encoder", model.action_encoder), ("predictor", model.predictor)):
            def pre(_module, args, name=name):
                record = counts.setdefault(phase[0] + ":" + name, dict(attempted=0, returned=0, samples=0))
                record["attempted"] += 1
                record["samples"] += int(args[0].shape[0])
            def post(_module, _args, _output, name=name):
                counts[phase[0] + ":" + name]["returned"] += 1
            handles.extend([module.register_forward_pre_hook(pre), module.register_forward_hook(post)])
        started = time.monotonic()
        try:
            with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                with np.load(source.blind.NPZ, allow_pickle=False) as archive:
                    goal_rgb = archive["goal_rgb"].copy()
                transform = p0.image_transform()
                def encode(image):
                    return model.encode({"pixels": p0._image_tensor(image, transform=transform, device=device)})["emb"][0, 0]
                goal = encode(goal_rgb)[None]
                jac = np.stack([(encode(tangent.perturb(goal_rgb, axis, .5))
                                 - encode(tangent.perturb(goal_rgb, axis, -.5))).cpu().numpy()
                                for axis in tangent.AXES], axis=1)
                q = tensor(revision.coordinate_metric(jac, protocol["object_weight"], protocol["pusher_weight"]))
                for role in protocol["source"]["roles"]:
                    phase[0] = f"cp5_{role['index']}"
                    with Image.open(role["cp5_png"]) as image:
                        latent = encode(np.asarray(image.convert("RGB")).copy())
                    states, actions = source.encode.contexts(task, role["actual_prefix"], latent.cpu().numpy())
                    z0, history = tensor(states)[None], tensor(actions)[None]
                    for objective in protocol["objectives"]:
                        for search in protocol["searches"]:
                            arm = f"{role['index']:02d}_{objective}_{search}"
                            phase[0] = arm
                            def score(variants):
                                flat = tensor(variants.reshape(320, 2, 5, 2))
                                z, past = z0.repeat(320, 1, 1), history.repeat(320, 1, 1)
                                for packet in range(2):
                                    incoming = torch.cat([past, ((flat[:, packet]-mean)/std).reshape(320, 1, 10)], dim=1)
                                    pred = model.predict(z, model.action_encoder(incoming))
                                    z, past = torch.cat([z[:, 1:], pred[:, -1:]], dim=1), incoming[:, 1:]
                                delta = z[:, -1] - goal
                                if objective == "object_tangent":
                                    delta = delta @ q.T
                                return delta.square().mean(dim=1).reshape(64, 5).cpu().numpy()
                            result = revision.plan(role["geometry"]["contact_direction_raw_xy"], score,
                                                   structured=search == "coherent",
                                                   common=source.engine.base.common,
                                                   seed=protocol["source"]["seed"])
                            rows.append(dict(arm=arm, index=role["index"], objective=objective,
                                             search=search, prefix=role["actual_prefix"], **result))
                            write(output / (arm + ".plan.json"), rows[-1])
                            for name in ("predictor", "action_encoder"):
                                if counts[arm+":"+name] != dict(attempted=10, returned=10, samples=3200):
                                    raise ValueError("per-arm model budget mismatch")
                            print(arm, result["selected"]["robust_score"], flush=True)
            after = source.engine.base.blind.host.signal.model_hash(model)
            if before != after or len(rows) != 16:
                raise ValueError("model/panel integrity")
            load(output, expected)
            write(output / "plans.json", dict(protocol_sha256=expected, rows=rows,
                  host=host, before=before, after=after, counts=counts,
                  wall_seconds=time.monotonic()-started, P1_admission=False,
                  full_cost_certified=False))
        finally:
            for handle in handles:
                handle.remove()


def physics(output, expected):
    """All sealed arms, exact old-prefix/RGB replay; evaluator state stays here."""
    protocol = load(output, expected)
    from . import p1_dev303_dynamics_cp5_tail_physical_20261102_exclusive as physical
    base = physical.base
    import importlib
    import inspect
    plans_path = output / "plans.json"
    plans = json.loads(plans_path.read_text())
    if plans["protocol_sha256"] != expected or len(plans["rows"]) != 16:
        raise ValueError("full frozen model panel required")
    _, parent, roles, _ = physical.source_data()
    originals = {row["index"]: row for row in roles}
    store = base.Store(output / "physics")
    store.json("binding.json", dict(plans_sha256=sha(plans_path), protocol_sha256=expected))
    rows = []
    for selected in plans["rows"]:
        original = originals[selected["index"]]
        arm = selected["arm"]
        row = dict(arm=arm, transitions=[], error=None)
        cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
        if sha(inspect.getsourcefile(cls)) != base.native.ENV_SHA:
            raise ValueError("simulator changed")
        env = cls(render_mode="rgb_array", resolution=224, with_target=False)
        try:
            obs, info = env.reset(seed=physical.SEED)
            if (not np.array_equal(obs["state"], original["reset_checkpoint"]["state"])
                    or not np.array_equal(info["goal_state"], parent["goal_state_evaluation_only"])):
                raise ValueError("reset/goal mismatch")
            batches = [("warmup", np.zeros((10, 2), np.float32)),
                       ("prefix", np.asarray(selected["prefix"], np.float32)),
                       ("tail", np.asarray(selected["selected"]["actions"], np.float32).reshape(10, 2))]
            for phase, actions in batches:
                for step, action in enumerate(actions, 1):
                    obs, reward, terminated, truncated, info = env.step(action.copy())
                    if not np.array_equal(env.latest_action, action):
                        raise ValueError("actual action mismatch")
                    row["transitions"].append(dict(phase=phase, step=step, action=action.tolist(),
                        state=np.asarray(obs["state"]).tolist(), reward=float(reward),
                        terminated=bool(terminated), truncated=bool(truncated)))
                    if phase == "prefix":
                        base.capture(store, env, arm, phase, step, original["prefix_checkpoints"][step-1])
                    if phase == "tail":
                        base.capture(store, env, arm, phase, step)
                    if terminated or truncated:
                        if phase != "tail":
                            raise ValueError("pre-tail unexpected native stop")
                        break
            row["status"] = "recorded"
        except Exception as exc:
            row["error"] = str(exc)
            row["status"] = "failed"
        finally:
            env.close()
        store.json(arm + ".json", row)
        rows.append(row)
        print(arm, row["status"], row["transitions"][-1]["reward"], flush=True)
    store.json("report.json", dict(rows=rows, protocol_sha256=expected,
               plans_sha256=sha(plans_path), P1_admission=False,
               independent_confirmation=False, goal_state_evaluation_only=parent["goal_state_evaluation_only"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "plan", "physics"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output.resolve())
    else:
        {"plan": plan, "physics": physics}[args.command](args.output.resolve(), args.sha256)
