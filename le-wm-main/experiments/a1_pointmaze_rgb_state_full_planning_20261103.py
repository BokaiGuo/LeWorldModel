"""Develop initial and feedback CEM with RGB-derived position inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from experiments.a0_pointmaze_native_feedback_time_20261102 import encode
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import estimate_state
from experiments.a1_pointmaze_native_spectral_20261102 import packet
from experiments.a1_pointmaze_rgb_task_spectral_dev_20261102 import task_mask
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import cem, score_plans
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE = ROOT / "results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json"
SOURCE_SHA = "ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade"
CAMERA = ROOT / "results/a0_pointmaze_camera_metadata_20261103_DEV01.json"
CAMERA_SHA = "1b7e13ee87608cbc2fd5e98f3c42b098d0416c9425ec879e2cdd0bc963fc5ecf"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_RGB_STATE_FULL_PLANNING_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a1_pointmaze_rgb_state_full_planning_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def state_from_rgb(image, policy, camera, offset):
    if policy == "zero_state":
        return np.zeros(4, dtype=np.float32)
    if policy == "rgb_position":
        return estimate_state(image, camera, offset)[0]
    raise ValueError(f"unknown state policy: {policy}")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing to overwrite existing result")
    if sha256_file(SOURCE) != SOURCE_SHA or sha256_file(CAMERA) != CAMERA_SHA:
        raise RuntimeError("sealed source or camera changed")
    source, camera_doc = json.loads(SOURCE.read_text()), json.loads(CAMERA.read_text())
    for path, expected in (
        (SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
        (DEFAULT_CHECKPOINT, source["checkpoint_sha256"]),
        (DEFAULT_DATASET / "states.pth", source["states_sha256"]),
        (DEFAULT_DATASET / "actions.pth", source["actions_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera_doc["xml_sha256"]:
        raise RuntimeError("scene changed")
    scene = mujoco.MjModel.from_xml_string(xml)
    body = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_BODY, "particle")
    if body < 0:
        raise RuntimeError("particle body missing")
    offset = scene.body_pos[body].copy()
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    arrays, rows, start = {}, [], time.monotonic()
    print("Initial plus feedback RGB-position planning; four exposed development roots.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved:
        with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            device = require_cuda()
            model, dataset = load_model_and_dataset(args, device)
            before = parameter_hash(model)
            if before != source["model_before"]:
                raise RuntimeError("model identity changed")
            mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
            if not torch.isfinite(std).all() or torch.any(std <= 0):
                raise ValueError("invalid action normalizer")
            for prior in source["rows"]:
                root = prior["root"]
                archive_path = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
                if sha256_file(archive_path) != prior["archive_sha256"]:
                    raise RuntimeError("archive changed")
                initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
                rgb0, goal_rgb = saved[f"r{root}_rgb0"], saved[f"r{root}_goal_rgb"]
                goal = encode(model, goal_rgb, device)
                mask, _ = task_mask(rgb0, goal_rgb)
                initial_noise, tail_noise = saved[f"r{root}_initial_noise"], saved[f"r{root}_tail_noise"]
                branches, sequences = [], []
                for policy in ("zero_state", "rgb_position"):
                    for method in ("random_cem", "spectral_cem"):
                        with NativeRGBBridge(initial) as env:
                            if not np.array_equal(env.initial_rgb, rgb0):
                                raise RuntimeError("reset RGB mismatch")
                            torch.cuda.synchronize()
                            t0 = time.monotonic()
                            input0 = state_from_rgb(rgb0, policy, camera_doc["metadata"], offset)
                            context = encode_with_state(model, rgb0, input0, device)
                            torch.cuda.synchronize()
                            encode0_seconds = time.monotonic() - t0
                            injected, singular_values, jac = None, None, None
                            structure_seconds = 0.
                            if method == "spectral_cem":
                                def predicted_visual(u):
                                    return model.unroll(context, act_suffix=packet(u, mean, std))["visual"][-1][0]
                                torch.cuda.synchronize()
                                t0 = time.monotonic()
                                zero = torch.zeros(2, device=device)
                                jac = torch.stack([torch.autograd.functional.jvp(predicted_visual, zero, d)[1].detach()
                                                   for d in torch.eye(2, device=device)], dim=-1)
                                selected = jac[0, torch.as_tensor(mask, device=device), :, :].reshape(-1, 2)
                                _, singular, vh = torch.linalg.svd(selected, full_matrices=False)
                                torch.cuda.synchronize()
                                structure_seconds = time.monotonic() - t0
                                direction = vh[0].cpu().numpy()
                                injected = np.zeros((4, 2, 2), np.float32)
                                injected[:, 0, :] = np.asarray([.3, -.3, .6, -.6], np.float32)[:, None] * direction
                                singular_values = singular.cpu().tolist()
                            counts = {"initial_batches": 0, "initial_samples": 0,
                                      "tail_batches": 0, "tail_samples": 0}
                            def score_initial(proposals):
                                counts["initial_batches"] += 1
                                counts["initial_samples"] += len(proposals)
                                return score_plans(model, context, goal, mask, proposals, mean, std)
                            torch.cuda.synchronize()
                            t0 = time.monotonic()
                            plan, cost, trace = cem(initial_noise, score_initial, injected)
                            torch.cuda.synchronize()
                            initial_seconds = time.monotonic() - t0
                            rgb5 = env.step(np.repeat(plan[0][None], 5, axis=0))
                            torch.cuda.synchronize()
                            t0 = time.monotonic()
                            input5 = state_from_rgb(rgb5, policy, camera_doc["metadata"], offset)
                            feedback = encode_with_state(model, rgb5, input5, device)
                            feedback_mask, _ = task_mask(rgb5, goal_rgb)
                            torch.cuda.synchronize()
                            feedback_seconds = time.monotonic() - t0
                            def score_tail(proposals):
                                counts["tail_batches"] += 1
                                counts["tail_samples"] += len(proposals)
                                return score_plans(model, feedback, goal, feedback_mask, proposals, mean, std)
                            torch.cuda.synchronize()
                            t0 = time.monotonic()
                            tail, tail_cost, tail_trace = cem(tail_noise, score_tail)
                            torch.cuda.synchronize()
                            tail_seconds = time.monotonic() - t0
                            rgb10 = env.step(np.repeat(tail[0][None], 5, axis=0))
                        if policy == "zero_state":
                            old = next(b for b in prior["branches"] if b["method"] == method)
                            if (not np.array_equal(plan, np.asarray(old["initial_plan"], np.float32))
                                    or not np.array_equal(tail, np.asarray(old["tail_plan"], np.float32))
                                    or image_hash(rgb5) != old["rgb5_sha256"]
                                    or image_hash(rgb10) != old["rgb10_sha256"]):
                                raise RuntimeError("zero-state source replay mismatch")
                        key = f"r{root}_{policy}_{method}"
                        arrays[f"{key}_rgb5"], arrays[f"{key}_rgb10"] = rgb5, rgb10
                        arrays[f"{key}_mask5"] = feedback_mask
                        if jac is not None:
                            arrays[f"{key}_jacobian"] = jac.cpu().numpy()
                        sequences.append(np.concatenate((np.repeat(plan[0][None], 5, axis=0),
                                                         np.repeat(tail[0][None], 5, axis=0))))
                        branches.append({"policy": policy, "method": method, "input0": input0.tolist(),
                                         "input5": input5.tolist(), "initial_plan": plan.tolist(),
                                         "initial_cost": cost, "initial_trace": trace,
                                         "tail": tail.tolist(), "tail_cost": tail_cost, "tail_trace": tail_trace,
                                         "injected": None if injected is None else injected.tolist(),
                                         "singular_values": singular_values, "prediction_counts": counts,
                                         "initial_encoding_seconds": encode0_seconds,
                                         "structure_seconds": structure_seconds,
                                         "initial_cem_seconds": initial_seconds,
                                         "feedback_encoding_seconds": feedback_seconds,
                                         "tail_cem_seconds": tail_seconds,
                                         "jvp_calls": 2 if injected is not None else 0,
                                         "svd_calls": 1 if injected is not None else 0,
                                         "physical_steps": 10, "feedback_encodes": 1,
                                         "rgb5_sha256": image_hash(rgb5), "rgb10_sha256": image_hash(rgb10)})
                        print(f"root {root}: {policy}/{method} executed", flush=True)
                target = dataset.states[root, 10, :2].cpu().numpy().astype(np.float64)
                endpoints = physical.replay_position_endpoints(np.repeat(initial[None], len(sequences), axis=0),
                                                                np.stack(sequences))
                for branch, end in zip(branches, endpoints, strict=True):
                    branch["endpoint_eval_only"] = end.tolist()
                    branch["distance_eval_only"] = float(np.linalg.norm(end[:2] - target))
                rows.append({"root": root, "goal_position_eval_only": target.tolist(), "branches": branches})
            after = parameter_hash(model)
            if before != after:
                raise RuntimeError("frozen model changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a1_pointmaze_rgb_state_full_planning.v1", "source_sha256": SOURCE_SHA,
              "camera_sha256": CAMERA_SHA, "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "body_offset_from_static_scene": offset.tolist(), "model_before": before, "model_after": after,
              "device": torch.cuda.get_device_name(device), "physical_backend": physical.position_backend_receipt(),
              "scope": "exposed roots; unequal total compute; not independent P1",
              "wall_seconds": time.monotonic() - start, "rows": rows}
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
