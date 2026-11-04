"""Causal frame-difference velocity in same-prefix visual feedback CEM."""
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
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_RGB_CAUSAL_VELOCITY_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a1_pointmaze_rgb_causal_velocity_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")
POLICIES = ("zero_state", "rgb_position_zero_velocity", "rgb_position_causal_velocity")


def causal_state(rgb_previous, rgb_current, camera, offset, dt):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("positive finite observation interval required")
    prev, _ = estimate_state(rgb_previous, camera, offset)
    curr, _ = estimate_state(rgb_current, camera, offset)
    curr[2:] = (curr[:2]-prev[:2])/dt
    if not np.isfinite(curr).all():
        raise ValueError("invalid state estimate")
    return curr


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    if sha256_file(SOURCE) != SOURCE_SHA or sha256_file(CAMERA) != CAMERA_SHA:
        raise RuntimeError("source changed")
    source, camera_doc = json.loads(SOURCE.read_text()), json.loads(CAMERA.read_text())
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (DEFAULT_CHECKPOINT, source["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", source["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", source["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed input: {path}")
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera_doc["xml_sha256"]:
        raise RuntimeError("scene changed")
    scene = mujoco.MjModel.from_xml_string(xml)
    body = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_BODY, "particle")
    if body < 0:
        raise RuntimeError("particle body missing")
    offset = scene.body_pos[body].copy()
    dt = float(scene.opt.timestep)*physical.FRAME_SKIP
    if not np.isclose(dt, .05, atol=1e-12, rtol=0):
        raise RuntimeError("high-level action interval differs from fixed protocol")
    saved = np.load(SOURCE.with_suffix(".npz"), allow_pickle=False)
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    arrays, rows, started = {}, [], time.monotonic()
    print("Real RGB causal velocity versus zero-state feedback CEM; exposed roots.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise RuntimeError("checkpoint identity mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        for row in source["rows"]:
            root = row["root"]
            image_path = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
            if sha256_file(image_path) != row["archive_sha256"]:
                raise RuntimeError("archive image changed")
            initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            goal_image = saved[f"r{root}_goal_rgb"]
            goal = encode(model, goal_image, device)
            branches, sequences = [], []
            for parent in row["branches"]:
                initial_method = parent["method"]
                plan = np.asarray(parent["initial_plan"], np.float32)
                packet = np.repeat(plan[0][None], 5, axis=0)
                shared_hashes = None
                for policy in POLICIES:
                    with NativeRGBBridge(initial) as env:
                        if not np.array_equal(env.initial_rgb, saved[f"r{root}_rgb0"]):
                            raise RuntimeError("initial RGB mismatch")
                        frames = [env.initial_rgb]
                        for action in packet:
                            frames.append(env.step(action[None]))
                        hashes = [image_hash(image) for image in frames]
                        if hashes[-1] != parent["rgb5_sha256"]:
                            raise RuntimeError("sealed source cp5 mismatch")
                        if shared_hashes is not None and hashes != shared_hashes:
                            raise RuntimeError("same prefix RGB history mismatch")
                        shared_hashes = hashes
                        position_only, _ = estimate_state(frames[-1], camera_doc["metadata"], offset)
                        estimated = causal_state(frames[-2], frames[-1], camera_doc["metadata"], offset, dt)
                        inputs = {"zero_state": np.zeros(4, np.float32),
                                  "rgb_position_zero_velocity": position_only,
                                  "rgb_position_causal_velocity": estimated}
                        torch.cuda.synchronize()
                        timer = time.monotonic()
                        feedback = encode_with_state(model, frames[-1], inputs[policy], device)
                        mask, _ = task_mask(frames[-1], goal_image)
                        counts = {"batches": 0, "samples": 0}
                        def scorer(proposals):
                            counts["batches"] += 1
                            counts["samples"] += len(proposals)
                            return score_plans(model, feedback, goal, mask, proposals, mean, std)
                        tail, cost, trace = cem(saved[f"r{root}_tail_noise"], scorer)
                        torch.cuda.synchronize()
                        planning_seconds = time.monotonic()-timer
                        rgb10 = env.step(np.repeat(tail[0][None], 5, axis=0))
                    if policy == "zero_state":
                        if not np.array_equal(tail, np.asarray(parent["tail_plan"], np.float32)):
                            raise RuntimeError("zero-state source selected tail mismatch")
                        if abs(cost-parent["tail_cost"]) > 1e-5 or image_hash(rgb10) != parent["rgb10_sha256"]:
                            raise RuntimeError("zero-state source score or RGB mismatch")
                    key = f"r{root}_{initial_method}_{policy}"
                    arrays[f"{key}_history"] = np.stack(frames)
                    arrays[f"{key}_rgb10"] = rgb10
                    sequences.append(np.concatenate((packet, np.repeat(tail[0][None], 5, axis=0))))
                    branches.append({"initial_method": initial_method, "policy": policy,
                                     "input_state": inputs[policy].tolist(), "estimated_causal_state": estimated.tolist(),
                                     "tail": tail.tolist(), "tail_cost": cost, "trace": trace,
                                     "history_sha256": hashes, "rgb10_sha256": image_hash(rgb10),
                                     "tail_prediction": counts, "feedback_observation_count": 5,
                                     "feedback_encoder_calls": 1, "physical_steps": 10,
                                     "feedback_planning_seconds": planning_seconds})
                print(f"root {root}, {initial_method}: 3 feedback controls executed", flush=True)
            # Only now inspect evaluator state; it is excluded from planner inputs.
            goal_pos = dataset.states[root, 10, :2].cpu().numpy().astype(np.float64)
            endpoints = physical.replay_position_endpoints(np.repeat(initial[None], len(sequences), axis=0),
                                                            np.stack(sequences))
            prefix_sequences = np.stack(sequences)[:, :5]
            cp5 = physical.replay_position_endpoints(np.repeat(initial[None], len(sequences), axis=0),
                                                     prefix_sequences)
            for b, end, mid in zip(branches, endpoints, cp5):
                b["endpoint_eval_only"] = end.tolist()
                b["cp5_backend_state_eval_only"] = mid.tolist()
                b["distance_eval_only"] = float(np.linalg.norm(end[:2]-goal_pos))
                estimate = np.asarray(b["estimated_causal_state"])
                b["cp5_position_error_eval_only"] = float(np.linalg.norm(estimate[:2]-mid[:2]))
                b["cp5_velocity_error_eval_only"] = float(np.linalg.norm(estimate[2:]-mid[2:]))
            rows.append({"root": root, "goal_position_eval_only": goal_pos.tolist(), "branches": branches})
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a1_pointmaze_rgb_causal_velocity.v1", "source_sha256": SOURCE_SHA,
              "camera_sha256": CAMERA_SHA, "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "body_offset_from_static_scene": offset.tolist(), "observation_interval": dt,
              "model_before": before, "model_after": after, "device": torch.cuda.get_device_name(device),
              "physical_backend": physical.position_backend_receipt(), "wall_seconds": time.monotonic()-started,
              "scope": "exposed roots; same saved prefixes; causal RGB feedback; not independent P1", "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
