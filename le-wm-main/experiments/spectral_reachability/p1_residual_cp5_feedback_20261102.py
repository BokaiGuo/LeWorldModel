"""Post-development cp5 RGB/action-only, eight-way V13 model planner."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev303_v3_cuda_executor_20261102_exclusive as origin
from . import p1_dev303_rgb_boundary_static_20261102_exclusive as blind
from . import p1_dev303_cp5_rgb_action_export_20261102_exclusive as rgb
from . import p1_dev303_cp5_dual_encode_20261102_exclusive as encode
from . import p1_dev303_cp5_encoded_v13_replan_20261102_exclusive as planner
from . import p1_dev303_goal_tangent_builder_20261102_exclusive as metric

ROOT = origin.ROOT
NAME = "p1_residual_cp5_feedback_20261102"
MODEL_REPORT = ROOT / "results/p1_dev303_residual_spectral_20261102_main_run01/report.json"
PHYS_DIR = ROOT / "results/p1_dev303_residual_spectral_physical_20261102_main_run01"
ANALYSIS = ROOT / "results/p1_dev303_residual_spectral_saved_analysis_20261102_main_run01.json"
MODEL_SHA = "555eb599aa26de8d524849405e059a2366f10b40a5300acf14c57f5ffaeb3443"
PHYS_SHA = "e4de331a64adec269daf1e6efd75f940068800520a66a3870f3e0d389f8b9283"
ANALYSIS_SHA = "b1430e10764e0ff419d54fc260ce64f20ec48973f454aff29705ce4fa22f508b"
METRIC_SHA = "3b6c229fa6eb811f219b1968f8ee3496b0988af2a5d4c182e35b50eeffcbaf31"
ROLES = ((4, "spectral_0_mode2_+1"), (18, "zero"), (19, "task_greedy_direct"),
         (20, "random_control"))
METRICS = ("latent", "goal_tangent")
AUTHORITY = "DEV303_RESIDUAL_CP5_EIGHT_WAY_MODEL_ONLY_CUDA"
engine = planner.engine
require = origin.need


def file_sha(path):
    return rgb.sha(rgb.read_regular(path))


def fixed(path, digest):
    raw = rgb.read_regular(path)
    require(rgb.sha(raw) == digest, f"SHA mismatch: {path}")
    return rgb.decode(raw)


def prefix_rows(model_report, physical_report, directory):
    """Only prefix actions and cp5 image metadata are read from physical rows."""
    require(model_report["schema"] == "dev303_residual_spectral_candidates.report.v1"
            and model_report["status"] == "model_only_complete"
            and model_report["selections"]["latent"] == 4
            and model_report["P1_admission"] is False, "model report contract")
    require(physical_report["schema"] == "dev303_residual_spectral_physical.report.v1"
            and physical_report["model_report_sha256"] == MODEL_SHA
            and physical_report["P1_admission"] is False, "physical report binding")
    result = []
    for index, label in ROLES:
        model_rows = [row for row in model_report["rows"] if row["index"] == index]
        phys_rows = [row for row in physical_report["rows"]
                     if row["index"] == index and row["metric"] == "goal_tangent"]
        require(len(model_rows) == len(phys_rows) == 1, "unique fixed arm")
        m, p = model_rows[0], phys_rows[0]
        require(m["label"] == p["label"] == label
                and p["arm"] == f"candidate_{index:02d}_goal_tangent", "arm label")
        actions = [t for t in p["transitions"] if t["phase"] == "prefix"]
        warmup = [t for t in p["transitions"] if t["phase"] == "warmup"]
        require([t["step"] for t in warmup] == list(range(1, 11))
                and np.array_equal(np.asarray([t["action"] for t in warmup], dtype=np.float64),
                                   np.zeros((10, 2))), "actual warmup action history")
        require([t["step"] for t in actions] == [1, 2, 3, 4, 5], "actual prefix order")
        actual = np.asarray([t["action"] for t in actions], dtype=np.float64)
        require(actual.shape == (5, 2) and np.isfinite(actual).all()
                and (np.abs(actual) <= 1).all()
                and np.array_equal(actual, actual.astype("<f4").astype(np.float64))
                and actual.tolist() == m["prefix"], "actual action identity")
        frames = [o for o in p["observations"] if o["phase"] == "prefix" and o["step"] == 5]
        require(len(frames) == 1, "unique cp5 image")
        frame = {k: frames[0][k] for k in ("file", "file_sha256", "pixel_sha256")}
        image = rgb._frame(directory, frame)
        geometry = engine._geometry(image)
        result.append({"index": index, "label": label, "arm": p["arm"],
                       "actual_prefix": actual.tolist(),
                       "actual_prefix_sha256": rgb.sha(actual.astype("<f4").tobytes()),
                       "cp5_png": str((directory / frame["file"]).resolve()),
                       "cp5_file_sha256": frame["file_sha256"],
                       "cp5_pixel_sha256": frame["pixel_sha256"], "geometry": geometry})
    return result


def source_inputs():
    model = fixed(MODEL_REPORT, MODEL_SHA)
    physical = fixed(PHYS_DIR / "report.json", PHYS_SHA)
    analysis = fixed(ANALYSIS, ANALYSIS_SHA)
    require(analysis["schema"] == "dev303_residual_spectral_saved_analysis.v1"
            and analysis["physical_report_sha256"] == PHYS_SHA
            and analysis["frame_references_verified"] == 756
            and analysis["P1_admission"] is False, "saved analysis binding")
    model_protocol = fixed(MODEL_REPORT.parent / "protocol.json", model["protocol_sha256"])
    require(model_protocol["source_sha256"][str(Path(metric.__file__).resolve())] == METRIC_SHA
            and file_sha(metric.__file__) == METRIC_SHA, "sealed goal tangent helper")
    origin_prep = origin.read_prepared()
    origin.verify_bindings(origin_prep["model_source_bindings_declared_not_reverified"])
    origin.verify_bindings(origin_prep["producer_sources"])
    task_path = Path(origin_prep["task"]["task_export"])
    task_raw = rgb.read_regular(task_path)
    require(rgb.sha(task_raw) == origin_prep["task"]["task_sha256"], "task SHA")
    envelope = rgb.decode(gzip.decompress(task_raw))
    task = envelope["task"]
    require(envelope["task_id"] == origin_prep["task"]["task_id"]
            and envelope["P1_admission"] is False and task["latent_dim"] == 192,
            "task identity")
    require(model["model_before"] == model["model_after"]
            and model["model_before"]["original_state_dict"] == origin_prep["model_sha256_expected_from_prepare"],
            "frozen model identity")
    require(file_sha(blind.NPZ) == blind.NPZ_SHA, "blind goal source")
    with np.load(blind.NPZ, allow_pickle=False) as archive:
        require(set(archive.files) == {"rgb_history", "raw_action_history", "goal_rgb"},
                "blind source layout")
        goal = archive["goal_rgb"].copy()
        require(np.array_equal(archive["raw_action_history"], np.zeros((10, 2)))
                and task["root_raw_action_history"] == np.zeros((10, 2)).tolist(),
                "task warmup history identity")
    require(goal.shape == (224, 224, 3) and goal.dtype == np.uint8
            and rgb.sha(goal.tobytes()) == model["metric"]["goal_pixel_sha256"], "goal identity")
    return origin_prep, task, goal, model, prefix_rows(model, physical, PHYS_DIR)


def protocol():
    prep, task, goal, model, roles = source_inputs()
    modules = (Path(__file__), Path(origin.__file__), Path(blind.__file__), Path(rgb.__file__),
               Path(encode.__file__), Path(planner.__file__), Path(planner.parent.__file__),
               Path(engine.__file__), Path(engine.base.__file__), Path(engine.base.common.__file__),
               Path(engine.contact.__file__), Path(engine.base.p0.__file__), Path(metric.__file__),
               Path(metric.tangent.__file__), Path(metric.visibility.__file__))
    bindings = {str(path.resolve()): file_sha(path) for path in modules}
    bindings.update({str(path.resolve()): file_sha(path) for path in
                     (MODEL_REPORT, PHYS_DIR / "report.json", ANALYSIS, MODEL_REPORT.parent / "protocol.json",
                      Path(prep["task"]["task_export"]), blind.NPZ)})
    bindings.update({row["cp5_png"]: row["cp5_file_sha256"] for row in roles})
    return {"schema": NAME + ".protocol.v1", "classification": "posthoc_development_not_holdout",
            "roles": roles, "metrics": list(METRICS), "task": task,
            "policy": prep["model_policy"], "model_hash": model["model_before"],
            "goal_pixel_sha256": rgb.sha(goal.tobytes()),
            "source_report_sha256": {"model": MODEL_SHA, "physical": PHYS_SHA, "analysis": ANALYSIS_SHA},
            "helper_sha256": METRIC_SHA, "seed": planner.parent.SEED,
            "initialization": "V13 common zero mean, zero seeds, same seed and std for all eight",
            "per_plan": {"iterations": 5, "candidates_per_iteration": 64, "variants": 5,
                         "horizon_packets": 2, "action_encoder_calls": 10,
                         "predict_calls": 10, "predict_samples": 3200},
            "shared_preparation": {"goal_encode_calls": 11, "cp5_encode_calls": 4},
            "projection": "V13 sum of first three actions projected on RGB nearest-object direction >= 0.15; not per-step contact",
            "physical_fields_used": "cp5 RGB and executed prefix actions only; no reward or state or later observations",
            "physical_actions_authorized": False, "P1_admission": False,
            "independent_confirmation": False, "full_cost_certified": False, "bindings": bindings}


def write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return rgb.sha(raw)


def run_cuda(contract_path, contract_sha, output, authority):
    require(authority == AUTHORITY, "model-only authority")
    expected = fixed(contract_path, contract_sha)
    require(expected == protocol(), "presealed protocol drift")
    require(not os.path.lexists(output), "exclusive output")
    output = Path(output)
    output.mkdir()
    counts, rows, hooks = {}, [], []
    phase = {"name": "startup"}
    with (output / "events.jsonl").open("x", buffering=1) as journal:
        def event(kind, **data):
            journal.write(json.dumps({"kind": kind, "phase": phase["name"], **data},
                                     sort_keys=True, allow_nan=False) + "\n")
            journal.flush()
        def pre(name):
            def invoke(_module, args):
                key = phase["name"] + ":" + name
                counts.setdefault(key, {"attempted": 0, "returned": 0, "batch_samples": 0})
                counts[key]["attempted"] += 1
                counts[key]["batch_samples"] += int(args[0].shape[0])
                event("module_attempt", module=name, count=counts[key]["attempted"])
            return invoke
        def post(name):
            def invoke(_module, _args, _output):
                key = phase["name"] + ":" + name
                counts[key]["returned"] += 1
                event("module_return", module=name, count=counts[key]["returned"])
            return invoke
        try:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            import torch
            from torch.nn.attention import SDPBackend, sdpa_kernel
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            device = torch.device("cuda:0")
            host = require_cuda_allocation("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            model = load_compatible_pretrained(expected["policy"], device=device).eval().requires_grad_(False)
            before = engine.base.blind.host.signal.model_hash(model)
            require(before == expected["model_hash"], "model state and buffer SHA")
            for name, module in (("encoder", model.encoder), ("projector", model.projector),
                                 ("action_encoder", model.action_encoder), ("predictor", model.predictor)):
                hooks.extend((module.register_forward_pre_hook(pre(name)), module.register_forward_hook(post(name))))
            _, task, goal_rgb, _, roles = source_inputs()
            tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
            mean, std, goal = tensor(task["action_mean"]), tensor(task["action_std"]), tensor(task["goal"])[None]
            require(bool((std > 0).all()), "action normalization")
            with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                phase["name"] = "shared_goal"
                q, metric_meta = metric.build(model, goal_rgb, device)
                require(metric_meta["encoder_calls"] == 11
                        and np.allclose(metric_meta["goal_latent"], task["goal"], rtol=1e-5, atol=1e-6),
                        "goal encoding and helper budget")
                q_tensor = tensor(q)
                for role in roles:
                    phase["name"] = f"cp5_{role['index']:02d}"
                    path = Path(role["cp5_png"])
                    frame = {"file": path.name, "file_sha256": role["cp5_file_sha256"],
                             "pixel_sha256": role["cp5_pixel_sha256"]}
                    pixels = rgb._frame(path.parent, frame)
                    image = engine.base.p0._image_tensor(pixels, transform=engine.base.p0.image_transform(), device=device)
                    latent = model.encode({"pixels": image})["emb"]
                    require(tuple(latent.shape) == (1, 1, 192) and latent.device == device
                            and bool(torch.isfinite(latent).all()), "cp5 CUDA latent")
                    states, actions = encode.contexts(task, role["actual_prefix"], latent[0, 0].cpu().numpy())
                    z0, history = tensor(states)[None], tensor(actions)[None]
                    for metric_name in METRICS:
                        phase["name"] = f"plan_{role['index']:02d}_{metric_name}"
                        def score(variants):
                            batch = engine.SAMPLES * engine.VARIANTS
                            flat = tensor(variants.reshape(batch, 2, 5, 2))
                            z, past = z0.repeat(batch, 1, 1), history.repeat(batch, 1, 1)
                            for packet in range(engine.HORIZON):
                                context = torch.cat((past, ((flat[:, packet] - mean) / std).reshape(batch, 1, 10)), dim=1)
                                prediction = model.predict(z, model.action_encoder(context))
                                require(prediction.shape == z.shape and bool(torch.isfinite(prediction).all()),
                                        "finite model prediction")
                                z = torch.cat((z, prediction[:, -1:]), dim=1)[:, -3:]
                                past = context[:, -2:]
                            delta = z[:, -1] - goal
                            if metric_name == "goal_tangent":
                                delta = delta @ q_tensor.T
                            return delta.square().mean(dim=1).reshape(engine.SAMPLES, engine.VARIANTS).cpu().numpy()
                        result = planner.plan_one(role["geometry"]["contact_direction_raw_xy"], score)
                        actions_out = np.asarray(result["selected"]["actions"], dtype="<f4").reshape(10, 2)
                        require(np.isfinite(actions_out).all() and (np.abs(actions_out) <= 1).all(), "bounded tail")
                        row = {"index": role["index"], "label": role["label"], "metric": metric_name,
                               "cp5_pixel_sha256": role["cp5_pixel_sha256"],
                               "actual_prefix_sha256": role["actual_prefix_sha256"],
                               "tail_actions_sha256": rgb.sha(actions_out.tobytes()), **result}
                        rows.append(row)
                        write_new(output / f"candidate_{role['index']:02d}_{metric_name}.json", row)
                        event("plan_complete", index=role["index"], metric=metric_name)
            torch.cuda.synchronize(device)
            after = engine.base.blind.host.signal.model_hash(model)
            require(before == after and len(rows) == 8
                    and len({r["initial_bank_sha256"] for r in rows}) == 1, "model or initialization drift")
            for name, n, samples in (("encoder", 11, 11), ("projector", 11, 11)):
                require(counts[f"shared_goal:{name}"] == {"attempted": n, "returned": n, "batch_samples": samples},
                        "goal hook ledger")
            for role in roles:
                for name in ("encoder", "projector"):
                    require(counts[f"cp5_{role['index']:02d}:{name}"]["attempted"] == 1
                            and counts[f"cp5_{role['index']:02d}:{name}"]["returned"] == 1, "cp5 hook ledger")
                for m in METRICS:
                    for name in ("action_encoder", "predictor"):
                        require(counts[f"plan_{role['index']:02d}_{m}:{name}"]
                                == {"attempted": 10, "returned": 10, "batch_samples": 3200}, "plan hook ledger")
            require(expected == protocol(), "postrun protocol closure")
            phase["name"] = "finished"
            event("finished", rows=len(rows))
            report = {"schema": NAME + ".report.v1", "status": "model_only_complete",
                      "protocol_sha256": contract_sha, "model_before": before, "model_after": after,
                      "host": host, "metric": metric_meta, "rows": rows, "hook_ledger": counts,
                      "journal_sha256": file_sha(output / "events.jsonl"),
                      "full_cost_certified": False, "physical_actions_authorized": False,
                      "P1_admission": False, "independent_confirmation": False}
            return write_new(output / "report.json", report)
        except Exception as exc:
            event("failed", error_type=type(exc).__name__, message=str(exc))
            write_new(output / "failure.json", {"error_type": type(exc).__name__, "message": str(exc),
                                                "phase": phase["name"], "hook_ledger": counts,
                                                "completed_rows": len(rows), "P1_admission": False})
            raise
        finally:
            for handle in hooks:
                handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--authority")
    args = parser.parse_args()
    require(args.tag.isascii() and args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    if args.command == "prepare":
        require(args.sha256 is None and args.authority is None and not os.path.lexists(output), "prepare only")
        print(json.dumps({"contract": str(prepared), "sha256": write_new(prepared, protocol())}))
    else:
        require(args.sha256 and args.authority, "explicit SHA and authorization")
        with engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            digest = run_cuda(prepared, args.sha256, output, args.authority)
        print(json.dumps({"report_sha256": digest, "output": str(output)}))


if __name__ == "__main__":
    main()
