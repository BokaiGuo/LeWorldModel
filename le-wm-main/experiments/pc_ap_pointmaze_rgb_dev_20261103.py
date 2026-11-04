"""Observation-memory versus final-target AP-compatible PointMaze control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_native_feedback_time_20261102 import encode, goal_latent
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA, estimate_state
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import OLD, OLD_SHA, OUTPUT as SOURCE
from experiments.a1_pointmaze_proprio_score_feedback_cem_20261103 import SOURCE_SHA
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state
from experiments.a1_pointmaze_rgb_prefix_causal_tail_20261103 import replay_legacy
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.pc_ap_observation_memory import ObservationMemory
from experiments.spectral_reachability.pc_ap_official_cem_kernel import official_cem_kernel
from experiments.spectral_reachability.pc_ap_pointmaze_callback_20261103 import PointMazeAPCallback
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

PROTOCOL = ROOT.parent / "docs/PC_AP_POINTMAZE_RGB_DEV_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_ap_pointmaze_rgb_dev_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")
ROOTS = (1888, 1889)
MEMORY = (1800, 1801, 1802)


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (CAMERA, CAMERA_SHA), (OLD, OLD_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    source, camera, old = (json.loads(p.read_text()) for p in (SOURCE, CAMERA, OLD))
    if sha256_file(SOURCE.with_suffix(".npz")) != source["arrays_sha256"]:
        raise RuntimeError("saved inputs changed")
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera["xml_sha256"]:
        raise RuntimeError("scene changed")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, arrays, starts, sequences, finals = [], {}, [], [], []
    started = time.monotonic()
    print("Run fixed AP-compatible memory versus final-target RGB feedback on two exposed roots.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise RuntimeError("frozen model mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("invalid normalization")
        memory_images, memory_features, memory_hashes = {}, {}, {}
        torch.cuda.synchronize()
        memory_started = time.monotonic()
        for episode in MEMORY:
            path = DEFAULT_DATASET / "obses" / f"episode_{episode:03d}.pth"
            memory_hashes[str(episode)] = sha256_file(path)
            images = torch.load(path, map_location="cpu", weights_only=True).numpy()[:21]
            if images.shape != (21, 224, 224, 3) or images.dtype != np.uint8:
                raise ValueError("invalid memory images")
            memory_images[str(episode)] = images
            memory_features[str(episode)] = np.stack([
                goal_latent(encode(model, image, device)).cpu().numpy() for image in images
            ])
        memory = ObservationMemory(memory_features, [str(root) for root in ROOTS], block=5)
        torch.cuda.synchronize()
        memory_seconds = time.monotonic() - memory_started
        lower = ((torch.full_like(mean, -1.) - mean) / std).repeat(5)
        upper = ((torch.ones_like(mean) - mean) / std).repeat(5)
        prior = ((torch.zeros_like(mean) - mean) / std).repeat(5).reshape(1, 1, 10)
        for root in ROOTS:
            initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            goal_rgb = saved[f"r{root}_goal_rgb"].copy()
            goal_state, _ = estimate_state(goal_rgb, camera["metadata"], old["body_offset_from_static_scene"])
            goal = encode_with_state(model, goal_rgb, goal_state, device)
            goal_feature = goal_latent(goal).cpu().numpy()
            branches = []
            for arm in ("memory", "final_goal"):
                decisions, actions_all = [], []
                with NativeRGBBridge(initial) as env:
                    if not np.array_equal(env.initial_rgb, saved[f"r{root}_rgb0"]):
                        raise RuntimeError("reset RGB mismatch")
                    frames = [env.initial_rgb]
                    for decision in range(2):
                        torch.cuda.synchronize()
                        decision_start = time.monotonic()
                        state = (estimate_state(frames[-1], camera["metadata"],
                                                old["body_offset_from_static_scene"])[0]
                                 if decision == 0 else causal_state(
                                     frames[-2], frames[-1], camera["metadata"],
                                     old["body_offset_from_static_scene"], .05))
                        context = encode_with_state(model, frames[-1], state, device)
                        retrieval = None
                        if arm == "memory":
                            query = goal_latent(context).cpu().numpy()
                            tic = time.monotonic()
                            hit = memory.retrieve(query, goal_feature, 10, 5 * decision)
                            retrieval_seconds = time.monotonic() - tic
                            target_rgb = memory_images[hit.episode_id][hit.start + 5]
                            target_state, _ = estimate_state(
                                target_rgb, camera["metadata"], old["body_offset_from_static_scene"])
                            target = encode_with_state(model, target_rgb, target_state, device)
                            retrieval = {"episode": hit.episode_id, "start": hit.start,
                                         "horizon": hit.horizon, "target_index": hit.start + 5,
                                         "target_rgb_sha256": image_hash(target_rgb),
                                         "squared_distance": hit.squared_distance,
                                         "seconds": retrieval_seconds}
                        else:
                            target = goal
                        callback = PointMazeAPCallback(model, context, target["proprio"][0, 0], device)
                        generator = torch.Generator(device=device).manual_seed(2026110300 + root * 10 + decision)
                        result = official_cem_kernel(callback, prior, lower, upper, generator,
                                                     population=16, iterations=3, elites=4)
                        if (callback.batches, callback.samples) != (4, 50):
                            raise RuntimeError("prediction accounting mismatch")
                        normalized = result.normalized_block.reshape(5, 2)
                        raw = (normalized * std + mean).cpu().numpy()
                        raw = np.clip(raw, -1, 1).astype(np.float32)
                        torch.cuda.synchronize()
                        planning_seconds = time.monotonic() - decision_start
                        input_hash = image_hash(frames[-1])
                        for action in raw:
                            frames.append(env.step(action[None]))
                        actions_all.extend(raw)
                        decisions.append({"input_rgb_sha256": input_hash,
                                          "output_rgb_sha256": image_hash(frames[-1]),
                                          "retrieval": retrieval, "planning_seconds": planning_seconds,
                                          "prediction_batches": callback.batches,
                                          "prediction_samples": callback.samples,
                                          "initial_cost": result.initial_cost,
                                          "refined_cost": result.refined_cost,
                                          "refined_selected": result.refined_selected,
                                          "actions": raw.tolist()})
                    if env.step_count != 10:
                        raise RuntimeError("action clock mismatch")
                sequence = np.asarray(actions_all, np.float32)
                key = f"r{root}_{arm}"
                arrays[f"{key}_frames"] = np.stack(frames)
                arrays[f"{key}_actions"] = sequence
                starts.append(initial)
                sequences.append(sequence)
                finals.append(frames[-1])
                branches.append({"arm": arm, "decisions": decisions, "physical_steps": 10})
                print(f"{root} {arm}: ten actions executed", flush=True)
            rows.append({"root": root, "branches": branches})
        after = parameter_hash(model)
        if after != before:
            raise RuntimeError("model parameters changed")
    replay_rgb, endpoints = replay_legacy(starts, sequences, xml)
    position_endpoints = physical.replay_position_endpoints(np.asarray(starts), np.asarray(sequences))
    if not np.array_equal(replay_rgb, np.stack(finals)) or not np.array_equal(
            endpoints[:, :2], position_endpoints[:, :2]):
        raise RuntimeError("independent execution replay mismatch")
    arrays["endpoints_eval_only"] = endpoints
    for i, row in enumerate(rows):
        goal_eval = dataset.states[row["root"], 10, :2].cpu().numpy().astype(np.float64)
        row["goal_position_eval_only"] = goal_eval.tolist()
        for j, branch in enumerate(row["branches"]):
            endpoint = endpoints[2*i+j]
            distance = float(np.linalg.norm(endpoint[:2] - goal_eval))
            branch.update({"endpoint_eval_only": endpoint.tolist(), "distance_eval_only": distance,
                           "strict_endpoint_success_eval_only": bool(distance < .5)})
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "pc_ap_pointmaze_rgb_dev.v1", "rows": rows,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "source_sha256": sha256_file(SOURCE), "arrays_sha256": sha256_file(ARRAYS),
              "memory_archive_sha256": memory_hashes, "memory_frames_encoded": 63,
              "memory_seconds": memory_seconds, "model_before": before, "model_after": after,
              "physical_executions": len(sequences), "verification_replays": len(sequences),
              "wall_seconds": time.monotonic() - started,
              "scope": "exposed roots, adaptation, no matched total budget or independent confirmation"}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
