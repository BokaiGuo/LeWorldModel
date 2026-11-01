"""Sealed matched development comparison for mode-preserving tail search."""
import argparse
import importlib
import inspect
import json
from pathlib import Path
import time

import numpy as np

from . import p1_dev303_dynamics_cp5_v13_cem_20261102_exclusive as source
from . import p1_multimodal_tail_search as multimodal
from . import p1_task_response_revision as baseline
from .p1_task_response_experiment import sha, write

ROOT = source.ROOT


def prepare(output):
    previous = source.protocol()
    bindings = dict(previous["bindings"])
    bindings.update({str(Path(p).resolve()): sha(p) for p in
                     (__file__, multimodal.__file__, baseline.__file__)})
    output.mkdir()
    protocol = dict(schema="p1_multimodal_tail_development.v1", source=previous,
                    bindings=bindings, objectives=["latent"],
                    searches=["coherent", "multimodal"],
                    samples=64, variants=5, iterations=5, horizon_packets=2,
                    comparison="four fixed cp5 prefixes; same RGB/history, latent goal and score",
                    classification="posthoc_development_not_confirmation",
                    no_simulator_state_or_outcome_for_planning=True,
                    P1_admission=False, full_cost_certified=False)
    write(output / "protocol.json", protocol)
    print(sha(output / "protocol.json"), flush=True)


def load(output, expected):
    path = output / "protocol.json"
    if sha(path) != expected:
        raise ValueError("protocol SHA")
    protocol = json.loads(path.read_text())
    if any(sha(path) != digest for path, digest in protocol["bindings"].items()):
        raise ValueError("source binding drift")
    return protocol


def plan(output, expected):
    protocol = load(output, expected)
    if (output / "plans.json").exists():
        raise FileExistsError("plans already exist")
    from .e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0
    from PIL import Image
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend

    with gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(protocol["source"]["policy"], device=device).eval().requires_grad_(False)
        before = source.engine.base.blind.host.signal.model_hash(model)
        if before != protocol["source"]["model_hash"]:
            raise ValueError("frozen checkpoint mismatch")
        task = protocol["source"]["task"]
        tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
        mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
        goal = tensor(task["goal"])[None]
        counts, handles, rows = {}, [], []
        phase = ["startup"]
        for name, module in (("encoder", model.encoder), ("projector", model.projector),
                             ("action_encoder", model.action_encoder), ("predictor", model.predictor)):
            def pre(_module, args, name=name):
                entry = counts.setdefault(phase[0]+":"+name, dict(attempted=0, returned=0, samples=0))
                entry["attempted"] += 1
                entry["samples"] += int(args[0].shape[0])
            def post(_module, _args, _output, name=name):
                counts[phase[0]+":"+name]["returned"] += 1
            handles.extend((module.register_forward_pre_hook(pre), module.register_forward_hook(post)))
        started = time.monotonic()
        try:
            with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                transform = p0.image_transform()
                for role in protocol["source"]["roles"]:
                    phase[0] = f"cp5_{role['index']:02d}"
                    with Image.open(role["cp5_png"]) as image:
                        pixels = np.asarray(image.convert("RGB")).copy()
                    latent = model.encode({"pixels": p0._image_tensor(
                        pixels, transform=transform, device=device)})["emb"][0, 0]
                    states, actions = source.encode.contexts(task, role["actual_prefix"], latent.cpu().numpy())
                    z0, history = tensor(states)[None], tensor(actions)[None]
                    for search in protocol["searches"]:
                        arm = f"{role['index']:02d}_latent_{search}"
                        phase[0] = arm
                        def score(variants):
                            flat = tensor(variants.reshape(320, 2, 5, 2))
                            z, past = z0.repeat(320, 1, 1), history.repeat(320, 1, 1)
                            for packet in range(2):
                                outgoing = torch.cat((past, ((flat[:, packet]-mean)/std).reshape(320, 1, 10)), dim=1)
                                pred = model.predict(z, model.action_encoder(outgoing))
                                if pred.shape != z.shape or not bool(torch.isfinite(pred).all()):
                                    raise ValueError("model prediction invalid")
                                z, past = torch.cat((z[:, 1:], pred[:, -1:]), dim=1), outgoing[:, 1:]
                            return (z[:, -1]-goal).square().mean(dim=1).reshape(64, 5).cpu().numpy()
                        direction = role["geometry"]["contact_direction_raw_xy"]
                        args = dict(common=source.engine.base.common, seed=protocol["source"]["seed"])
                        result = (baseline.plan(direction, score, structured=True, **args)
                                  if search == "coherent" else multimodal.plan(direction, score, **args))
                        row = dict(arm=arm, index=role["index"], objective="latent", search=search,
                                   prefix=role["actual_prefix"], **result)
                        rows.append(row)
                        write(output / (arm + ".plan.json"), row)
                        for name in ("predictor", "action_encoder"):
                            if counts[arm+":"+name] != dict(attempted=10, returned=10, samples=3200):
                                raise ValueError("model budget mismatch")
                        print(arm, result["selected"]["robust_score"], flush=True)
            torch.cuda.synchronize()
            after = source.engine.base.blind.host.signal.model_hash(model)
            if before != after or len(rows) != 8:
                raise ValueError("model or panel integrity")
            load(output, expected)
            write(output / "plans.json", dict(protocol_sha256=expected, rows=rows, host=host,
                  before=before, after=after, counts=counts, wall_seconds=time.monotonic()-started,
                  P1_admission=False, full_cost_certified=False))
        finally:
            for handle in handles:
                handle.remove()


def physics(output, expected):
    load(output, expected)
    from . import p1_dev303_dynamics_cp5_tail_physical_20261102_exclusive as physical
    base = physical.base
    plans_path = output / "plans.json"
    plans = json.loads(plans_path.read_text())
    if plans["protocol_sha256"] != expected or len(plans["rows"]) != 8:
        raise ValueError("complete sealed panel required")
    _, parent, original_rows, _ = physical.source_data()
    originals = {row["index"]: row for row in original_rows}
    store = base.Store(output / "physics")
    store.json("binding.json", dict(plans_sha256=sha(plans_path), protocol_sha256=expected))
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    if sha(inspect.getsourcefile(cls)) != base.native.ENV_SHA:
        raise ValueError("simulator changed")
    rows = []
    for selected in plans["rows"]:
        original, arm = originals[selected["index"]], selected["arm"]
        row = dict(arm=arm, transitions=[], error=None)
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
                    obs, reward, terminated, truncated, _ = env.step(action.copy())
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
                            raise ValueError("pre-tail native stop")
                        break
            row["status"] = "recorded"
        except Exception as exc:
            row.update(status="failed", error=str(exc))
        finally:
            env.close()
        store.json(arm + ".json", row)
        rows.append(row)
        print(arm, row["status"], flush=True)
    store.json("report.json", dict(rows=rows, protocol_sha256=expected, plans_sha256=sha(plans_path),
               P1_admission=False, independent_confirmation=False,
               goal_state_evaluation_only=parent["goal_state_evaluation_only"]))


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
