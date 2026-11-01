"""Known-root post-hoc cp5 whole-tail model vs actual RGB analysis."""
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

from . import codex_p1_diag32_fixed_physical_verify_20261101_exclusive as old_verify
from . import codex_p1_diag32_cp5_hold_verify_20261101_exclusive as new_verify
from . import codex_p1_cp10_rgb_refresh_control_20261101_exclusive as refresh
from . import preflight_pusht_d2p_p0 as p0

ROOT = old_verify.ROOT
physical_engine = old_verify.physical
engine = physical_engine.engine
NAME = "codex_p1_cp5_tail_rgb_diagnosis_20261101_exclusive"
IDENTITY = ("state", "snapshot", "pixel_sha256", "goal", "goal_rgb_sha256")
CAPS = {"encode": 21, "action_encoder": 6, "predict": 6}
require, sha = engine.require, engine.sha


def check_cp5(a, b):
    require(all(a[k] == b[k] for k in IDENTITY), "cp5 identity mismatch")


def physical(state, goal):
    state, goal = np.asarray(state, np.float64), np.asarray(goal, np.float64)
    require(state.shape == goal.shape == (7,) and np.isfinite(state).all() and np.isfinite(goal).all(), "state")
    delta = state - goal
    return {"pusher_xy": float(np.linalg.norm(delta[:2])), "object_xy": float(np.linalg.norm(delta[2:4])),
            "joint_xy": float(np.linalg.norm(delta[:4])),
            "wrapped_angle": float(abs(np.arctan2(np.sin(delta[4]), np.cos(delta[4])))),
            "native_success": bool(old_verify.analysis.native_metrics(state, goal)["native_success"]),
            "strict_success": bool(old_verify.analysis.native_metrics(state, goal)["wrapped_allxy_success"])}


def prediction_label(step):
    return {5: "observed", 10: "model_one_packet", 15: "model_two_packets"}.get(step)


def build():
    old_verify.verify()
    new_verify.verify()
    old = json.loads((old_verify.DIRECTORY / "report.json").read_bytes())["row"]
    report = json.loads((new_verify.DIRECTORY / "report.json").read_bytes())
    new = next(r for r in report["rows"] if r["role"] == "selected")
    contract = json.loads(new_verify.CONTRACT.read_bytes())
    require(old["control_returned"] == new["control_returned"] == 15
            and len(old["observations"]) == len(new["observations"]) == 16, "complete physical histories")
    require(np.array_equal(np.asarray(old["executed_raw"][:5], np.float32),
                           np.asarray(new["executed_raw"][:5], np.float32)), "cp5 same actual actions")
    check_cp5(old["observations"][5], new["observations"][5])
    old_tail = np.asarray(old["executed_raw"][5:], dtype="<f4").reshape(2, 5, 2)
    new_tail = np.asarray(new["executed_raw"][5:], dtype="<f4").reshape(2, 5, 2)
    require(np.array_equal(old_tail, np.asarray(contract["hold_tails"]["selected"]["raw"], np.float32))
            and np.array_equal(new_tail, np.asarray(new["suffix"]["actions"], np.float32)), "sealed full tails")
    source_prepared, _, _, _, _, _, _ = physical_engine.inputs()
    task, _, task_sources, policy = physical_engine.blind.load(0)
    preseal = json.loads(new_verify.live.PRESEAL.read_bytes())
    paths = [Path(__file__).resolve(), ROOT / "tests" / ("test_" + NAME + ".py"),
             old_verify.PREP, old_verify.DIRECTORY / "report.json", old_verify.DIRECTORY / "journal.jsonl",
             new_verify.CONTRACT, new_verify.DIRECTORY / "report.json", new_verify.DIRECTORY / "journal.jsonl"]
    points = {"hold": old["observations"][5:16], "optimized": new["observations"][5:16]}
    paths += [Path(point["path"]) for branch in points.values() for point in branch]
    bindings = dict(source_prepared["sources"])
    bindings.update(task_sources)
    for path in paths:
        path = path.resolve()
        h = sha(path)
        require(str(path) not in bindings or bindings[str(path)] == h, "source collision")
        bindings[str(path)] = h
    require(all(sha(p) == h for p, h in bindings.items()), "source drift")
    return {"schema": NAME + ".protocol.v1", "task": task, "policy": policy,
            "model_hash": report["model_before"], "prefix": old["executed_raw"][:5],
            "warmup_last5": preseal["raw_warmup10"][5:],
            "arms": {"hold": {"tail": old_tail.tolist(), "frames": points["hold"]},
                     "optimized": {"tail": new_tail.tolist(), "frames": points["optimized"]}},
            "recorded_cp5_q2": {"hold": new["comparison"]["hold_q2"],
                                "optimized": new["comparison"]["optimized_q2"]},
            "bindings": bindings, "caps": CAPS, "wall_seconds": 600, "output_cap_bytes": 32*1024**2,
            "encode_schedule": "one shared cp5 and each arm's physical steps 6 through 15, total 21",
            "prediction_schedule": "per arm cp5->10->15 open loop and own actual cp10->15 refresh",
            "intermediate_prediction": "unavailable; no temporal interpolation",
            "physical_calls": 0, "training": False, "development_only": True,
            "outcomes_known_before_audit": True, "independent_confirmation": False, "P1_admission": False}


def run(prepared, expected, output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    require(sha(prepared) == expected and json.loads(prepared.read_bytes()) == build(), "prepared drift")
    value = json.loads(prepared.read_bytes())
    require(output.parent == ROOT / "results" and output.name.startswith(NAME + "_run_")
            and not os.path.lexists(output), "exclusive output")
    with physical_engine.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        output.mkdir()
        ledger = output / "events.jsonl"
        counts, returns = dict.fromkeys(CAPS, 0), dict.fromkeys(CAPS, 0)
        started = time.monotonic()
        status, failure, result, before, after = "failed", None, None, None, None

        def emit(item):
            payload = (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode()
            require(time.monotonic() - started < value["wall_seconds"]
                    and (ledger.stat().st_size if ledger.exists() else 0) + len(payload)
                    <= value["output_cap_bytes"], "wall/log cap")
            with ledger.open("ab") as out:
                require(out.write(payload) == len(payload), "ledger write")
                out.flush()
                os.fsync(out.fileno())

        def call(kind, context, fn):
            require(counts[kind] < CAPS[kind], "neural budget")
            counts[kind] += 1
            emit({"event": "attempt", "resource": kind, "context": context})
            v = fn()
            returns[kind] += 1
            emit({"event": "return", "resource": kind, "context": context})
            return v

        try:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            host = require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            model = load_compatible_pretrained(value["policy"], device=device).eval().requires_grad_(False)
            before = engine.signal.model_hash(model)
            require(before == value["model_hash"], "model identity")
            for module in model.modules():
                path = str(Path(inspect.getfile(type(module))).resolve())
                require(path in value["bindings"] and sha(path) == value["bindings"][path], "model module source")
            counts_hooks = {"predict": 0, "action_encoder": 0}
            def hook(kind):
                def inc(_module, _args):
                    counts_hooks[kind] += 1
                return inc
            handles = [model.predictor.register_forward_pre_hook(hook("predict")),
                       model.action_encoder.register_forward_pre_hook(hook("action_encoder"))]
            tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
            task = value["task"]
            mean, std, goal = tensor(task["action_mean"]), tensor(task["action_std"]), tensor(task["goal"])
            require(bool((std > 0).all()), "normalizer")
            def encode(point, label):
                path = Path(point["path"])
                require(not path.is_symlink() and sha(path) == point["file_sha256"], "frame file")
                with Image.open(path) as image:
                    pixels = np.asarray(image).copy()
                require(pixels.shape == (224, 224, 3) and hashlib.sha256(pixels.tobytes()).hexdigest() ==
                        point["pixel_sha256"], "frame pixels")
                image = p0._image_tensor(pixels, transform=p0.image_transform(), device=device)
                z = call("encode", label, lambda: model.encode({"pixels": image}))["emb"][0, 0]
                require(z.shape == (192,) and bool(torch.isfinite(z).all()), "encode")
                return z
            def predict(z, past, packet, label):
                context = torch.cat((past, ((tensor(packet) - mean) / std).reshape(1, 1, 10)), 1)
                action_z = call("action_encoder", label, lambda: model.action_encoder(context))
                pred = call("predict", label, lambda: model.predict(z, action_z))
                require(pred.shape == z.shape and bool(torch.isfinite(pred).all()), "predict")
                return torch.cat((z, pred[:, -1:]), 1)[:, -3:], context[:, -2:]
            with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                cp5 = encode(value["arms"]["hold"]["frames"][0], {"arm": "shared", "step": 5})
                z5 = cp5.cpu().numpy().astype(np.float32)
                source = new_verify.live.fixed
                identity = {"purpose": NAME, "boundary": 5, "pixel_sha256":
                            value["arms"]["hold"]["frames"][0]["pixel_sha256"]}
                task5 = source.feedback_task(task, value["prefix"], value["prefix"], value["warmup_last5"],
                                             z5, identity)
                rows = {}
                for arm in ("hold", "optimized"):
                    slot = value["arms"][arm]
                    real = {5: cp5}
                    for point in slot["frames"][1:]:
                        real[point["step"]] = encode(point, {"arm": arm, "step": point["step"]})
                    z, history = tensor(task5["root_context"]["latent_history"])[None], \
                                 tensor(task5["root_context"]["action_history"])[None]
                    z, history = predict(z, history, slot["tail"][0], {"arm": arm, "phase": "open", "step": 10})
                    p10 = z[0, -1].clone()
                    z, history = predict(z, history, slot["tail"][1], {"arm": arm, "phase": "open", "step": 15})
                    p15 = z[0, -1].clone()
                    score15 = float((p15 - goal).square().mean().item())
                    require(abs(score15 - value["recorded_cp5_q2"][arm]) < 1e-5, "recorded cp5 score mismatch")
                    context10 = refresh.cp10_context(task5["root_context"], value["prefix"], slot["tail"][0],
                                                     real[10].cpu().numpy(), task["action_mean"], task["action_std"])
                    rz, rh = tensor(context10["latent_history"])[None], tensor(context10["action_history"])[None]
                    rz, _ = predict(rz, rh, slot["tail"][1], {"arm": arm, "phase": "own_cp10_rgb", "step": 15})
                    fresh15 = rz[0, -1].clone()
                    stage = []
                    for point in slot["frames"]:
                        step = point["step"]
                        actual = real[step]
                        row = {"step": step, "pixel_sha256": point["pixel_sha256"],
                               "physical": physical(point["state"], point["goal"]),
                               "actual_rgb_goal_mse": float((actual-goal).square().mean().item()),
                               "model_prediction": None}
                        predicted = {10: p10, 15: p15}.get(step)
                        if predicted is not None:
                            row["model_prediction"] = {"mode": "cp5_open_loop", "target_step": step,
                                "goal_mse": float((predicted-goal).square().mean().item()),
                                "actual_latent_rmse": float((predicted-actual).square().mean().sqrt().item())}
                        if step == 15:
                            row["own_cp10_refresh_prediction"] = {
                                "goal_mse": float((fresh15-goal).square().mean().item()),
                                "actual_latent_rmse": float((fresh15-actual).square().mean().sqrt().item())}
                        stage.append(row)
                    rows[arm] = {"stages": stage, "actual_latents": {str(k): v.cpu().tolist() for k, v in real.items()},
                                 "predicted_cp10": p10.cpu().tolist(), "predicted_cp15": p15.cpu().tolist(),
                                 "own_cp10_predicted_cp15": fresh15.cpu().tolist()}
                torch.cuda.synchronize()
            for handle in handles:
                handle.remove()
            after = engine.signal.model_hash(model)
            require(before == after and counts == returns == CAPS
                    and counts_hooks == {"predict": 6, "action_encoder": 6}
                    and all(sha(p) == h for p, h in value["bindings"].items()), "completion/source/accounting")
            result, status = {"arms": rows, "goal_latent": task["goal"],
                              "cp10_rgb_same": value["arms"]["hold"]["frames"][5]["pixel_sha256"] ==
                                               value["arms"]["optimized"]["frames"][5]["pixel_sha256"]}, "complete"
        except BaseException as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        report = {"schema": NAME + ".report.v1", "status": status, "failure": failure,
                  "prepare_sha256": expected, "counts": counts, "returns": returns,
                  "model_before": before, "model_after": after, "elapsed_seconds": time.monotonic()-started,
                  "events_sha256": sha(ledger) if ledger.exists() else None, "result": result,
                  "physical_calls": 0, "training": False, "development_only": True,
                  "independent_confirmation": False, "P1_admission": False}
        digest = physical_engine.blind.write_new(output / "report.json", report)
        return report, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    require(args.tag.isalnum(), "tag")
    prep = ROOT / "results" / (NAME + "_PREP_" + args.tag + ".json")
    output = ROOT / "results" / (NAME + "_run_" + args.tag)
    if args.mode == "prepare":
        print(json.dumps({"prepared": str(prep), "sha256": physical_engine.blind.write_new(prep, build())}))
        return 0
    require(args.sha256, "prepared SHA required")
    report, digest = run(prep, args.sha256, output)
    print(json.dumps({"output": str(output), "status": report["status"], "failure": report["failure"],
                      "report_sha256": digest}))
    return int(report["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
