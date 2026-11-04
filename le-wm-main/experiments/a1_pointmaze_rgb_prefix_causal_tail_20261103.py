"""Real-RGB tail CEM after previously selected RGB-position prefixes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import mujoco
import numpy as np
import torch

from experiments.a0_pointmaze_native_feedback_time_20261102 import encode
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import estimate_state
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import cem, score_plans
from experiments.pointmaze_native_rgb_bridge_20261102 import LEGACY, RUNTIME, NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE = ROOT / "results/a1_pointmaze_rgb_state_full_planning_20261103_DEV01.json"
SOURCE_SHA = "4288b73e675e45526db46698a7abdcd68b619fc235e0a73bc20f91bead6d9b99"
ORIGINAL = ROOT / "results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json"
ORIGINAL_SHA = "ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade"
CAMERA = ROOT / "results/a0_pointmaze_camera_metadata_20261103_DEV01.json"
CAMERA_SHA = "1b7e13ee87608cbc2fd5e98f3c42b098d0416c9425ec879e2cdd0bc963fc5ecf"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_RGB_POSITION_PREFIX_CAUSAL_TAIL_PROTOCOL_20261103.md"
WORKER = ROOT / "experiments/pointmaze_candidate_rgb_replay_worker_20261103.py"
OUTPUT = ROOT / "results/a1_pointmaze_rgb_prefix_causal_tail_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def replay_legacy(starts, sequences, xml):
    with tempfile.TemporaryDirectory(prefix="pc_causal_tail_check_") as tmp:
        request, scene, response = (Path(tmp) / n for n in ("request.npz", "scene.xml", "response.npz"))
        np.savez_compressed(request, sources=np.asarray(starts, np.float64),
                            sequences=np.asarray(sequences, np.float64))
        scene.write_text(xml)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        proc = subprocess.run([str(LEGACY / "bin/python"), str(WORKER), str(request), str(scene), str(response)],
                              capture_output=True, text=True, env=env, timeout=300)
        if proc.returncode or not response.is_file():
            raise RuntimeError(f"legacy replay failed: {proc.stderr[-2000:]}")
        with np.load(response, allow_pickle=False) as data:
            return data["rgb10"].copy(), data["endpoints"].copy()


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (ORIGINAL, ORIGINAL_SHA), (CAMERA, CAMERA_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    source, original, camera = (json.loads(p.read_text()) for p in (SOURCE, ORIGINAL, CAMERA))
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (ORIGINAL.with_suffix(".npz"), original["arrays_sha256"]),
                           (DEFAULT_CHECKPOINT, original["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", original["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", original["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"sealed input changed: {path}")
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera["xml_sha256"]:
        raise RuntimeError("scene differs")
    scene = mujoco.MjModel.from_xml_string(xml)
    if not np.isclose(scene.opt.timestep * 5, .05, rtol=0, atol=1e-12):
        raise RuntimeError("unexpected time step")
    offset = source["body_offset_from_static_scene"]
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    arrays, rows, starts, sequences, terminal_images = {}, [], [], [], []
    begun = time.monotonic()
    print("Executing matched RGB-position-prefix tail CEM: zero versus causal velocity.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as old, \
            np.load(ORIGINAL.with_suffix(".npz"), allow_pickle=False) as saved, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise RuntimeError("model changed")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid action normalizer")
        for row in source["rows"]:
            root = row["root"]
            initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            goal_rgb = saved[f"r{root}_goal_rgb"]
            goal = encode(model, goal_rgb, device)
            mask = saved[f"r{root}_initial_mask"]
            noise = saved[f"r{root}_tail_noise"]
            branches = []
            for parent in (x for x in row["branches"] if x["policy"] == "rgb_position"):
                method = parent["method"]
                plan = np.asarray(parent["initial_plan"], np.float32)
                packet = np.repeat(plan[0][None], 5, axis=0)
                previous_frames = None
                for policy in ("zero_velocity", "causal_velocity"):
                    with NativeRGBBridge(initial) as env:
                        if not np.array_equal(env.initial_rgb, saved[f"r{root}_rgb0"]):
                            raise RuntimeError("initial RGB mismatch")
                        frames = [env.initial_rgb]
                        for action in packet:
                            frames.append(env.step(action[None]))
                        if not np.array_equal(frames[-1], old[f"r{root}_rgb_position_{method}_rgb5"]):
                            raise RuntimeError("sealed prefix image mismatch")
                        frame_hashes = [image_hash(x) for x in frames]
                        if previous_frames is not None and frame_hashes != previous_frames:
                            raise RuntimeError("paired prefix history mismatch")
                        previous_frames = frame_hashes
                        torch.cuda.synchronize()
                        t0 = time.monotonic()
                        zero_state, _ = estimate_state(frames[-1], camera["metadata"], offset)
                        causal = causal_state(frames[-2], frames[-1], camera["metadata"], offset, .05)
                        state = zero_state if policy == "zero_velocity" else causal
                        feedback = encode_with_state(model, frames[-1], state, device)
                        torch.cuda.synchronize()
                        feedback_seconds = time.monotonic() - t0
                        counts = {"batches": 0, "samples": 0}
                        def scorer(proposals):
                            counts["batches"] += 1
                            counts["samples"] += len(proposals)
                            return score_plans(model, feedback, goal, mask, proposals, mean, std)
                        torch.cuda.synchronize()
                        t0 = time.monotonic()
                        tail, cost, trace = cem(noise, scorer)
                        torch.cuda.synchronize()
                        plan_seconds = time.monotonic() - t0
                        rgb10 = env.step(np.repeat(tail[0][None], 5, axis=0))
                        if env.step_count != 10:
                            raise RuntimeError("incorrect action clock")
                    if counts != {"batches": 3, "samples": 48}:
                        raise RuntimeError("incorrect CEM count")
                    key = f"r{root}_{method}_{policy}"
                    arrays[f"{key}_history"] = np.stack(frames)
                    arrays[f"{key}_rgb10"] = rgb10
                    starts.append(initial)
                    sequences.append(np.concatenate((packet, np.repeat(tail[0][None], 5, axis=0))))
                    terminal_images.append(rgb10)
                    branches.append({"method": method, "policy": policy, "prefix": plan[0].tolist(),
                                     "input_state": state.tolist(), "tail": tail.tolist(),
                                     "tail_cost": cost, "trace": trace, "counts": counts,
                                     "feedback_seconds": feedback_seconds, "tail_seconds": plan_seconds,
                                     "physical_steps": 10, "rgb_observations": 6,
                                     "history_sha256": frame_hashes, "rgb10_sha256": image_hash(rgb10)})
                print(f"root {root}, {method}: both tails executed", flush=True)
            rows.append({"root": root, "branches": branches})
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    # Evaluate only after every arm's action sequence was chosen and executed.
    legacy_rgb, legacy_endpoints = replay_legacy(starts, sequences, xml)
    if not np.array_equal(legacy_rgb, np.stack(terminal_images)):
        raise RuntimeError("legacy terminal image mismatch")
    endpoints = physical.replay_position_endpoints(np.asarray(starts), np.asarray(sequences))
    if not np.array_equal(endpoints[:, :2], legacy_endpoints[:, :2]):
        raise RuntimeError("position backend and legacy endpoint mismatch")
    idx = 0
    for row in rows:
        target = dataset.states[row["root"], 10, :2].cpu().numpy().astype(np.float64)
        row["goal_position_eval_only"] = target.tolist()
        for branch in row["branches"]:
            distance = float(np.linalg.norm(legacy_endpoints[idx, :2] - target))
            branch["endpoint_eval_only"] = legacy_endpoints[idx].tolist()
            branch["distance_eval_only"] = distance
            branch["strict_endpoint_success_eval_only"] = bool(distance < .5)
            idx += 1
    arrays["legacy_endpoints_eval_only"] = legacy_endpoints
    arrays["sequences"] = np.asarray(sequences)
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a1_pointmaze_rgb_prefix_causal_tail.v1",
              "source_sha256": SOURCE_SHA, "original_sha256": ORIGINAL_SHA, "camera_sha256": CAMERA_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "worker_sha256": sha256_file(WORKER),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "device": torch.cuda.get_device_name(device), "model_before": before, "model_after": after,
              "physical_executions": len(sequences), "action_steps": len(sequences)*10,
              "prediction_samples": len(sequences)*48, "prediction_packets": len(sequences)*48,
              "replay_terminal_images_exact": len(sequences), "replay_position_exact": len(sequences),
              "wall_seconds": time.monotonic()-begun,
              "scope": "exposed roots and fixed prefixes; endpoint-only success; no P1", "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
