"""Six-episode cp5 q1-vs-q2 development contrast with own RGB feedback."""
import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from . import codex_p1_cp5_q1_feedback_20261101_exclusive as spec
from . import codex_p1_diag32_cp5_hold_20261101_exclusive as old
from . import codex_p1_diag32_fixed_physical_20261101_exclusive as physical

ROOT, engine = spec.ROOT, physical.engine
NAME = "codex_p1_cp5_q1_feedback_runner_20261101_exclusive"
require, sha = engine.require, engine.sha


def activation():
    return spec.contract()


class Meter:
    def __init__(self, store, role, check):
        self.store, self.role, self.check = store, role, check
        self.caps = {"actual_rgb_encode": 2, "action_encoder": 6, "predict": 6}
        self.attempts = dict.fromkeys(self.caps, 0)
        self.returns = dict.fromkeys(self.caps, 0)

    def call(self, kind, context, fn):
        self.check()
        require(self.attempts[kind] < self.caps[kind], "neural cap")
        self.attempts[kind] += 1
        self.store.emit("neural_attempt", role=self.role, resource=kind, context=context)
        result = fn()
        self.returns[kind] += 1
        self.store.emit("neural_return", role=self.role, resource=kind, context=context)
        return result


def check_hooks(after, before, calls):
    require(all(after[k] - before[k] == calls[k] for k in ("predict", "action_encoder")),
            "module hook/accounting mismatch")


def encode(model, device, pixels, pixel_sha, meter, boundary):
    import torch
    require(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
            and hashlib.sha256(pixels.tobytes()).hexdigest() == pixel_sha, "fresh RGB")
    image = old.p0._image_tensor(pixels, transform=old.p0.image_transform(), device=device)
    with torch.inference_mode():
        result = meter.call("actual_rgb_encode", {"boundary": boundary, "pixel_sha256": pixel_sha},
                            lambda: model.encode({"pixels": image}))
    z = result["emb"][0, 0].detach().cpu().numpy().astype(np.float32)
    require(z.shape == (192,) and np.isfinite(z).all(), "encoded RGB")
    return z


def score(model, device, task, raw, meter, boundary, label):
    import torch
    packets = spec.actions(raw, (2 if boundary == 5 else 1, 5, 2))
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    z = tensor(task["root_context"]["latent_history"])[None]
    past = tensor(task["root_context"]["action_history"])[None]
    goal = tensor(task["goal"])[None]
    mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
    require(bool((std > 0).all()), "normalization")
    values = []
    with torch.inference_mode():
        for index, packet in enumerate(packets):
            context = torch.cat((past, ((tensor(packet) - mean) / std).reshape(1, 1, 10)), 1)
            loc = {"boundary": boundary, "candidate": label, "packet": index}
            encoded = meter.call("action_encoder", loc, lambda: model.action_encoder(context))
            prediction = meter.call("predict", loc, lambda: model.predict(z, encoded))
            require(prediction.shape == z.shape and bool(torch.isfinite(prediction).all()), "prediction")
            z = torch.cat((z, prediction[:, -1:]), 1)[:, -3:]
            past = context[:, -2:]
            values.append(float((z[:, -1] - goal).square().mean().item()))
    require(np.isfinite(values).all(), "score")
    return values


def run(activation_path, activation_sha, output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    require(sha(activation_path) == activation_sha and json.loads(activation_path.read_bytes()) == activation(),
            "activation/source drift")
    plan = json.loads(activation_path.read_bytes())
    previous = json.loads(old.PRESEAL.read_bytes())
    source_prepared, data, root_expected, _, _, _, _ = physical.inputs()
    task_base, _, _, _ = old.fixed.blind.load(0)
    engine.NAME, engine.CAPS = NAME, plan["physical_caps_total"]
    store = engine.Store(output)
    store.put("activation_binding.json", {"path": str(activation_path), "sha256": activation_sha})
    start = time.monotonic()
    device = torch.device("cuda:0")
    report = {"schema": NAME + ".report.v1", "status": "partial_or_failed", "rows": [], "errors": [],
              "activation_sha256": activation_sha, "plan_sha256": activation_sha, "development_only": True,
              "independent_confirmation": False, "P1_admission": False, "fair_total_cost": False}

    def check():
        require(time.monotonic() - start < plan["wall_seconds"], "wall cap")
        store.check()
        require(torch.cuda.memory_allocated(device) <= plan["cuda_allocated_cap_bytes"]
                and torch.cuda.memory_reserved(device) <= plan["cuda_reserved_cap_bytes"], "CUDA memory cap")

    physical_meter = engine.Meter(store, check)
    env = None
    try:
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        report["host"] = require_cuda_allocation("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(previous["model_policy"], device=device).eval().requires_grad_(False)
        report["model_before"] = engine.signal.model_hash(model)
        require(report["model_before"]["original_state_dict"] == previous["model_sha256"], "model identity")
        for module in model.modules():
            path = str(Path(inspect.getfile(type(module))).resolve())
            require(path in source_prepared["sources"] and sha(path) == source_prepared["sources"][path],
                    "model source")
        hooks = {"predict": 0, "action_encoder": 0}
        def hook(kind):
            def count(_module, _args):
                hooks[kind] += 1
            return count
        handles = [model.predictor.register_forward_pre_hook(hook("predict")),
                   model.action_encoder.register_forward_pre_hook(hook("action_encoder"))]
        store.emit("run_started", plan_sha256=activation_sha, caps=engine.CAPS)
        with torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            for role in spec.ROLES:
                for policy in spec.POLICIES:
                    ident = role + "_" + policy
                    row = deepcopy(engine.p3.roster()[0])
                    row.update(episode_id=ident, role=role, policy=policy, status="not_started",
                               transitions=[], observations=[], decisions=[], neural=None)
                    report["rows"].append(row)
                    physical_meter.episode, physical_meter.region = ident, "setup"
                    meter = Meter(store, ident, check)
                    hooks_before = deepcopy(hooks)
                    env = None
                    try:
                        env = physical_meter.call("constructor", lambda: engine.old.native_factory(physical_meter))
                        body, ref = env.unwrapped, data["references"][0]
                        require(body._live_meter is physical_meter and body._live_constructor_complete
                                and body.relative and body.action_scale == 100 and body.control_hz == 10
                                and body.dt == .01 and body.with_target is False and not body.render_action
                                and body.render_size == 224 and np.array_equal(env.action_space.low, [-1, -1])
                                and np.array_equal(env.action_space.high, [1, 1]), "native action ABI")
                        row["status"] = "running"
                        if not engine.moving.setup(env, ref, data["plan"], row, physical_meter, store):
                            row["status"] = "native_stopped_warmup"
                            continue
                        root = row["last_observation"]
                        require(all(root[k] == root_expected[k] for k in spec.IDENTITY), "root")
                        row["observations"].append(deepcopy(root))
                        require(not body.eval_state(ref["goal"], np.asarray(root["state"]))[0], "root stopped")
                        candidates = plan["candidates"][role]
                        prefix = spec.actions(candidates["prefix"], (5, 2))
                        commands = prefix.copy()
                        task5 = choice5 = None
                        step = 0
                        while step < len(commands):
                            command = commands[step]
                            physical_meter.region = "prefix" if step < 5 else "suffix"
                            store.emit("branch_dispatch", role=ident, step=step + 1, command=command.tolist())
                            obs, reward, terminated, truncated, info = env.step(command.copy())
                            step += 1
                            row["control_returned"] += 1
                            actual = np.asarray(body.latest_action)
                            require(engine.old.kernel.moving.exact(actual, command)
                                    and np.array_equal(info["goal_state"], ref["goal"])
                                    and np.array_equal(info["goal"], ref["goal_rgb"]), "actual action/goal")
                            state = np.asarray(obs["state"])
                            require(state.shape == (7,) and np.isfinite(state).all()
                                    and engine.old.kernel.moving.exact(state, np.asarray(body._get_obs()))
                                    and type(info.get("n_contacts")) is int, "physical observation")
                            engine.old.native_transition(body, ref["goal"], obs, reward, terminated, truncated, row)
                            event = {"role": ident, "step": step, "command": actual.tolist(),
                                     "state": state.tolist(), "reward": float(reward),
                                     "terminated": bool(terminated), "truncated": bool(truncated),
                                     "n_contacts": info["n_contacts"], "metrics": deepcopy(row["final_metrics"])}
                            row["executed_raw"].append(actual.tolist())
                            row["transitions"].append(event)
                            store.emit("branch_returned", **event)
                            try:
                                pixels, receipt = engine.old.capture(env, obs, row, store, "plan", step)
                            except engine.old.kernel.moving.Rejected as exc:
                                require(str(exc) == "fresh RGB/canvas guard" and engine.guard_origin(exc), "guard")
                                row.update(status="censored_out_of_domain", censor_step=step)
                                break
                            row["observations"].append(deepcopy(receipt))
                            if terminated or truncated:
                                row.update(status="native_stopped", stop_step=step)
                                break
                            if step not in (5, 10):
                                continue
                            saved = Path(receipt["path"])
                            require(saved.parent == store.path and sha(saved) == receipt["file_sha256"], "frame")
                            with Image.open(saved) as image:
                                saved_pixels = np.asarray(image).copy()
                            require(np.array_equal(saved_pixels, pixels), "fresh/saved RGB")
                            if step == 5:
                                require(all(receipt[k] == candidates["required_cp5"][k] for k in spec.IDENTITY),
                                        "cached candidate cp5 identity")
                            before = engine.old.state_seal(body)
                            latent = encode(model, device, saved_pixels, receipt["pixel_sha256"], meter, step)
                            if step == 5:
                                identity = {"role": ident, "boundary": 5, "pixel_sha256": receipt["pixel_sha256"],
                                            "prefix": prefix.tolist(), "preseal_sha256": old.PRESEAL_SHA}
                                task5 = old.fixed.feedback_task(
                                    task_base, prefix, np.asarray(row["executed_raw"], np.float32),
                                    np.asarray(previous["raw_warmup10"], np.float32)[5:], latent, identity)
                                require(task5["root_context"]["action_history"] ==
                                        previous["cp5_action_histories_expected"][role], "cp5 history")
                                both = [score(model, device, task5, candidates[k], meter, 5, k)
                                        for k in ("hold", "optimized")]
                                q1, q2 = [s[0] for s in both], [s[1] for s in both]
                                choice5 = spec.select_cp5(policy, q1, q2)
                                scores = {"q1": q1, "q2": q2}
                                packet = spec.actions(candidates[choice5], (2, 5, 2))[0]
                            else:
                                context = spec.cp10_context(task5["root_context"], prefix,
                                                            np.asarray(row["executed_raw"], np.float32)[5:10],
                                                            latent, task5["action_mean"], task5["action_std"])
                                task10 = deepcopy(task5)
                                task10["root_context"] = context
                                scores = [score(model, device, task10, [candidates[k][1]], meter, 10, k)[0]
                                          for k in ("hold", "optimized")]
                                choice = spec.select_cp10(scores)
                                packet = spec.actions(candidates[choice], (2, 5, 2))[1]
                            torch.cuda.synchronize(device)
                            require(engine.old.state_seal(body) == before, "planning changed physical state")
                            decision = {"boundary": step, "pixel_sha256": receipt["pixel_sha256"], "scores": scores,
                                        "choice": choice5 if step == 5 else choice,
                                        "packet_sha256": spec.digest_actions(packet)}
                            row["decisions"].append(decision)
                            store.put(ident + f"_cp{step}_decision.json", decision)
                            store.emit("decision_sealed", role=ident, **decision)
                            commands = np.concatenate((commands, packet))
                        else:
                            row["status"] = "completed15"
                    finally:
                        row["neural"] = {"attempts": meter.attempts, "returns": meter.returns,
                                         "module_hooks": {k: hooks[k] - hooks_before[k] for k in hooks}}
                        check_hooks(hooks, hooks_before, meter.attempts)
                        physical_meter.region = "closing"
                        if env is not None:
                            physical_meter.call("close", env.close)
                            env = None
                        store.put(ident + "_row.json", row)
                    print(json.dumps({"episode": ident, "status": row["status"],
                                      "returned": row["control_returned"]}), flush=True)
        torch.cuda.synchronize(device)
        for handle in handles:
            handle.remove()
        report["model_after"] = engine.signal.model_hash(model)
        require(report["model_before"] == report["model_after"] and not physical_meter.pending
                and not physical_meter.secondary, "model/accounting")
        require(activation() == json.loads(activation_path.read_bytes()), "source drift")
        report["status"] = "recorded_development"
    except BaseException as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1000]})
    finally:
        if env is not None:
            physical_meter.region = "closing"
            try:
                physical_meter.call("close", env.close)
            except BaseException as exc:
                report["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1000]})
        store.finalizing = True
        report.update(counts=physical_meter.counts, pending=physical_meter.pending,
                      secondary_errors=physical_meter.secondary, wall_seconds=time.monotonic()-start)
        store.emit("run_finished", status=report["status"], counts=physical_meter.counts)
        report["journal_sha256"] = sha(store.path / "journal.jsonl")
        report["journal_chain_tail"] = store.previous
        store.put("report.json", report, final=True)
    return report


def output_path(tag):
    require(tag.isalnum(), "tag")
    return ROOT / "results" / (NAME + "_run_" + tag)


def prepare(tag):
    require(tag.isalnum(), "tag")
    prepared = ROOT / "results" / (NAME + "_ACTIVATION_" + tag + ".json")
    output = output_path(tag)
    require(not os.path.lexists(prepared) and not os.path.lexists(output), "exclusive output")
    digest = old.fixed.blind.write_new(prepared, activation())
    return prepared, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.phase == "prepare":
        require(args.sha256 is None, "prepare takes no digest")
        prepared, digest = prepare(args.tag)
        print(json.dumps({"activation": str(prepared), "sha256": digest}), flush=True)
        return 0
    require(args.tag.isalnum() and args.sha256 is not None and len(args.sha256) == 64, "tag/digest")
    prepared = ROOT / "results" / (NAME + "_ACTIVATION_" + args.tag + ".json")
    output = output_path(args.tag)
    require(not os.path.lexists(output), "exclusive output")
    require(sha(prepared) == args.sha256 and json.loads(prepared.read_bytes()) == activation(),
            "activation/source drift")
    with physical.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        result = run(prepared, args.sha256, output)
    print(json.dumps({"output": str(output), "status": result["status"], "errors": result["errors"],
                      "report_sha256": sha(output / "report.json")}), flush=True)
    return int(result["status"] != "recorded_development")


if __name__ == "__main__":
    raise SystemExit(main())
