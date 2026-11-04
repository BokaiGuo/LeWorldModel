"""Development-only real-RGB CEM using an analytic affine-inverse score."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_failure_real_rgb_metric_20261103 import ORIGINAL, ORIGINAL_SHA
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA, estimate_state
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import OLD, OLD_SHA, OUTPUT as SOURCE, cem_four
from experiments.a1_pointmaze_proprio_score_feedback_cem_20261103 import OUTPUT as COMPARISON
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state
from experiments.a1_pointmaze_rgb_prefix_causal_tail_20261103 import replay_legacy
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import cem
from experiments.pc_pointmaze_affine_inverse_20261103 import MEAN, STD, get_encoder
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE_SHA = "9c8e00d5a5d1f1401108bbcfcd5dfda12658a9f7d49468189131a7db461bcb8c"
COMPARISON_SHA = "a07f3c73c6cbcc215899d7bbaa6ab1b30ecadf43188b7a882e926c66abdac3df"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_AFFINE_FEEDBACK_CEM_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_affine_feedback_cem_20261103_DEV02.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def projected_cost(prediction, goal, pinv, position_std):
    """Predict position-distance proxy without reading simulator outcomes."""
    if (prediction.ndim != 3 or prediction.shape[1:] != (256, 16)
            or goal.shape != (256, 16) or pinv.shape != (4, 16)
            or position_std.shape != (2,)):
        raise ValueError("invalid prediction or encoder shape")
    if not all(bool(torch.isfinite(x).all()) for x in (prediction, goal, pinv, position_std)):
        raise ValueError("nonfinite projection input")
    if bool((position_std <= 0).any()):
        raise ValueError("invalid position normalizer")
    delta = (prediction.mean(dim=1) - goal.mean(dim=0)) @ pinv.T
    result = torch.linalg.vector_norm(delta[:, :2] * position_std, dim=1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite projection score")
    return result


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (COMPARISON, COMPARISON_SHA),
                           (ORIGINAL, ORIGINAL_SHA), (CAMERA, CAMERA_SHA), (OLD, OLD_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed source: {path}")
    source, comparison, original, camera, old = (
        json.loads(path.read_text()) for path in (SOURCE, COMPARISON, ORIGINAL, CAMERA, OLD))
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (DEFAULT_CHECKPOINT, original["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", original["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", original["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed input: {path}")
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera["xml_sha256"]:
        raise RuntimeError("scene changed")
    weight, _, singular = get_encoder()
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, arrays, starts, sequences, terminals = [], {}, [], [], []
    started = time.monotonic()
    print("Running eight exposed roots with real RGB feedback and affine-inverse CEM.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise RuntimeError("model identity mismatch")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), STD, rtol=0, atol=1e-7)):
            raise RuntimeError("proprio normalizer mismatch")
        action_mean, action_std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(action_std).all()) or bool((action_std <= 0).any()):
            raise ValueError("invalid action normalizer")
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(STD[:2], dtype=torch.float32, device=device)
        if [r["root"] for r in source["rows"]] != list(range(1888, 1896)):
            raise RuntimeError("unexpected root roster")
        for prior in source["rows"]:
            root = prior["root"]
            if sha256_file(DEFAULT_DATASET / "obses" / f"episode_{root:03d}.pth") != prior["archive_sha256"]:
                raise RuntimeError("archive changed")
            initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            rgb0, goal_rgb = saved[f"r{root}_rgb0"], saved[f"r{root}_goal_rgb"]
            goal_state, _ = estimate_state(goal_rgb, camera["metadata"], old["body_offset_from_static_scene"])
            goal = encode_with_state(model, goal_rgb, goal_state, device)
            goal_token = goal["proprio"].reshape(256, 16)
            state0, _ = estimate_state(rgb0, camera["metadata"], old["body_offset_from_static_scene"])
            counts = {"initial_samples": 0, "tail_samples": 0, "initial_batches": 0, "tail_batches": 0}
            def scorer(context, phase):
                def evaluate(proposals):
                    counts[f"{phase}_samples"] += len(proposals)
                    counts[f"{phase}_batches"] += 1
                    plans = torch.as_tensor(proposals, dtype=torch.float32, device=device)
                    suffix = ((plans - action_mean) / action_std).repeat(1, 1, 5).permute(1, 0, 2)
                    with torch.no_grad():
                        predicted = model.unroll(context, act_suffix=suffix)["proprio"][-1]
                        result = projected_cost(predicted.reshape(len(plans), 256, 16),
                                                goal_token, pinv, pos_std)
                    return result.cpu().numpy()
                return evaluate
            with NativeRGBBridge(initial) as env:
                if not np.array_equal(env.initial_rgb, rgb0):
                    raise RuntimeError("reset RGB mismatch")
                torch.cuda.synchronize()
                t0 = time.monotonic()
                context = encode_with_state(model, rgb0, state0, device)
                plan, cost0, trace0 = cem_four(saved[f"r{root}_initial_noise"], scorer(context, "initial"))
                torch.cuda.synchronize()
                initial_seconds = time.monotonic() - t0
                history = [env.initial_rgb]
                for action in np.repeat(plan[0][None], 5, axis=0):
                    history.append(env.step(action[None]))
                torch.cuda.synchronize()
                t0 = time.monotonic()
                state5 = causal_state(history[-2], history[-1], camera["metadata"],
                                      old["body_offset_from_static_scene"], .05)
                feedback = encode_with_state(model, history[-1], state5, device)
                tail, cost5, trace5 = cem(saved[f"r{root}_tail_noise"], scorer(feedback, "tail"))
                torch.cuda.synchronize()
                tail_seconds = time.monotonic() - t0
                rgb10 = env.step(np.repeat(tail[0][None], 5, axis=0))
                if env.step_count != 10:
                    raise RuntimeError("action clock mismatch")
            if counts != {"initial_samples": 64, "tail_samples": 48, "initial_batches": 4, "tail_batches": 3}:
                raise RuntimeError("prediction count mismatch")
            sequence = np.concatenate((np.repeat(plan[0][None], 5, axis=0),
                                       np.repeat(tail[0][None], 5, axis=0)))
            starts.append(initial)
            sequences.append(sequence)
            terminals.append(rgb10)
            arrays[f"r{root}_history"] = np.stack(history)
            arrays[f"r{root}_rgb10"] = rgb10
            rows.append({"root": root, "goal_rgb_state": goal_state.tolist(), "cp5_input": state5.tolist(),
                         "initial_plan": plan.tolist(), "tail": tail.tolist(), "initial_cost": cost0,
                         "tail_cost": cost5, "initial_trace": trace0, "tail_trace": trace5,
                         "prediction_counts": counts, "current_rgb_encodes": 2, "physical_steps": 10,
                         "initial_encode_and_plan_seconds": initial_seconds,
                         "feedback_encode_and_plan_seconds": tail_seconds})
            print(f"root {root}: 10 physical steps", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    replay_rgb, endpoints = replay_legacy(starts, sequences, xml)
    endpoints2 = physical.replay_position_endpoints(np.asarray(starts), np.asarray(sequences))
    if not np.array_equal(replay_rgb, np.stack(terminals)) or not np.array_equal(endpoints[:, :2], endpoints2[:, :2]):
        raise RuntimeError("physical replay mismatch")
    for index, row in enumerate(rows):
        goal_eval = dataset.states[row["root"], 10, :2].cpu().numpy().astype(np.float64)
        distance = float(np.linalg.norm(endpoints[index, :2] - goal_eval))
        previous = next(r for r in comparison["rows"] if r["root"] == row["root"])
        row.update({"endpoint_eval_only": endpoints[index].tolist(), "goal_position_eval_only": goal_eval.tolist(),
                    "distance_eval_only": distance,
                    "previous_distances_eval_only": {b["arm"]: b["distance_eval_only"] for b in previous["branches"]}})
    arrays["sequences"], arrays["endpoints_eval_only"] = np.asarray(sequences), endpoints
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "pc_pointmaze_affine_feedback_cem.v1", "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "source_sha256": SOURCE_SHA,
              "comparison_sha256": COMPARISON_SHA, "arrays_sha256": sha256_file(ARRAYS),
              "model_before": before, "model_after": after, "encoder_singular_values": singular.tolist(),
              "device": torch.cuda.get_device_name(device), "physical_executions": len(rows),
              "physical_steps": 10 * len(rows), "wall_seconds": time.monotonic() - started,
              "rows": rows, "scope": "exposed development roots; not independent P1 or certified equal cost"}
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {OUTPUT} sha256={sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
