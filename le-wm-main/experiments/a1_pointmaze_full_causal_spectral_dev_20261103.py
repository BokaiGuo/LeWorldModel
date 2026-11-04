"""Recompile initial spectral/random CEM and causal-RGB tail end to end."""
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
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state
from experiments.a1_pointmaze_rgb_prefix_causal_tail_20261103 import replay_legacy
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import cem, score_plans
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

OLD = ROOT / "results/a1_pointmaze_rgb_state_full_planning_20261103_DEV01.json"
OLD_SHA = "4288b73e675e45526db46698a7abdcd68b619fc235e0a73bc20f91bead6d9b99"
ORIGINAL = ROOT / "results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json"
ORIGINAL_SHA = "ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade"
PRIOR = ROOT / "results/a1_pointmaze_rgb_prefix_causal_tail_20261103_DEV01.json"
PRIOR_SHA = "fd20c6f83df7dd6e2c0a21cfb05f25ab9d2627dfe2958266eb5b36e1381ecb2f"
CAMERA = ROOT / "results/a0_pointmaze_camera_metadata_20261103_DEV01.json"
CAMERA_SHA = "1b7e13ee87608cbc2fd5e98f3c42b098d0416c9425ec879e2cdd0bc963fc5ecf"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_FULL_CAUSAL_SPECTRAL_DEVELOPMENT_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a1_pointmaze_full_causal_spectral_dev_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((OLD, OLD_SHA), (ORIGINAL, ORIGINAL_SHA), (PRIOR, PRIOR_SHA), (CAMERA, CAMERA_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    old, original, prior, camera = (json.loads(p.read_text()) for p in (OLD, ORIGINAL, PRIOR, CAMERA))
    for path, expected in ((OLD.with_suffix(".npz"), old["arrays_sha256"]),
                           (ORIGINAL.with_suffix(".npz"), original["arrays_sha256"]),
                           (PRIOR.with_suffix(".npz"), prior["arrays_sha256"]),
                           (DEFAULT_CHECKPOINT, original["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", original["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", original["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"input changed: {path}")
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera["xml_sha256"]:
        raise RuntimeError("scene changed")
    if not np.isclose(mujoco.MjModel.from_xml_string(xml).opt.timestep * 5, .05, rtol=0, atol=1e-12):
        raise RuntimeError("step interval changed")
    offset = old["body_offset_from_static_scene"]
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    arrays, rows, starts, sequences, ends = {}, [], [], [], []
    started = time.monotonic()
    print("Recompile initial random/spectral CEM and execute causal RGB tail on development roots.", flush=True)
    with np.load(ORIGINAL.with_suffix(".npz"), allow_pickle=False) as saved, \
            np.load(PRIOR.with_suffix(".npz"), allow_pickle=False) as saved_prior, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != old["model_before"]:
            raise RuntimeError("model mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid normalization")
        for oldrow, priorrow in zip(old["rows"], prior["rows"], strict=True):
            root = oldrow["root"]
            if root != priorrow["root"]:
                raise RuntimeError("root order mismatch")
            initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            rgb0, rgb_goal = saved[f"r{root}_rgb0"], saved[f"r{root}_goal_rgb"]
            mask = saved[f"r{root}_initial_mask"]
            goal = encode(model, rgb_goal, device)
            branches = []
            for method in ("random_cem", "spectral_cem"):
                parent = next(x for x in oldrow["branches"] if x["policy"] == "rgb_position" and x["method"] == method)
                expected = next(x for x in priorrow["branches"] if x["policy"] == "causal_velocity" and x["method"] == method)
                with NativeRGBBridge(initial) as env:
                    if not np.array_equal(env.initial_rgb, rgb0):
                        raise RuntimeError("initial observation mismatch")
                    torch.cuda.synchronize()
                    t0 = time.monotonic()
                    input0, _ = estimate_state(rgb0, camera["metadata"], offset)
                    context = encode_with_state(model, rgb0, input0, device)
                    torch.cuda.synchronize()
                    initial_encode_seconds = time.monotonic()-t0
                    injected = None
                    structure_seconds = 0.
                    if method == "spectral_cem":
                        def predicted_visual(u):
                            return model.unroll(context, act_suffix=packet(u, mean, std))["visual"][-1][0]
                        torch.cuda.synchronize()
                        t0 = time.monotonic()
                        jac = torch.stack([torch.autograd.functional.jvp(predicted_visual, torch.zeros(2, device=device), d)[1].detach()
                                           for d in torch.eye(2, device=device)], dim=-1)
                        selected = jac[0, torch.as_tensor(mask, device=device), :, :].reshape(-1, 2)
                        _, singular, vh = torch.linalg.svd(selected, full_matrices=False)
                        torch.cuda.synchronize()
                        structure_seconds = time.monotonic()-t0
                        direction = vh[0].cpu().numpy()
                        injected = np.zeros((4, 2, 2), np.float32)
                        injected[:, 0, :] = np.asarray([.3, -.3, .6, -.6], np.float32)[:, None] * direction
                        arrays[f"r{root}_{method}_jac"] = jac.cpu().numpy()
                    counts = {"initial_batches": 0, "initial_samples": 0, "tail_batches": 0, "tail_samples": 0}
                    def initial_scorer(proposals):
                        counts["initial_batches"] += 1
                        counts["initial_samples"] += len(proposals)
                        return score_plans(model, context, goal, mask, proposals, mean, std)
                    torch.cuda.synchronize()
                    t0 = time.monotonic()
                    plan, cost0, trace0 = cem(saved[f"r{root}_initial_noise"], initial_scorer, injected)
                    torch.cuda.synchronize()
                    initial_seconds = time.monotonic()-t0
                    if not np.array_equal(plan, np.asarray(parent["initial_plan"], np.float32)):
                        raise RuntimeError("initial plan does not reproduce sealed RGB-position planner")
                    frames = [env.initial_rgb]
                    for action in np.repeat(plan[0][None], 5, axis=0):
                        frames.append(env.step(action[None]))
                    if image_hash(frames[-1]) != parent["rgb5_sha256"]:
                        raise RuntimeError("cp5 mismatch")
                    torch.cuda.synchronize()
                    t0 = time.monotonic()
                    state5 = causal_state(frames[-2], frames[-1], camera["metadata"], offset, .05)
                    feedback = encode_with_state(model, frames[-1], state5, device)
                    torch.cuda.synchronize()
                    feedback_seconds = time.monotonic()-t0
                    def tail_scorer(proposals):
                        counts["tail_batches"] += 1
                        counts["tail_samples"] += len(proposals)
                        return score_plans(model, feedback, goal, mask, proposals, mean, std)
                    torch.cuda.synchronize()
                    t0 = time.monotonic()
                    tail, cost5, trace5 = cem(saved[f"r{root}_tail_noise"], tail_scorer)
                    torch.cuda.synchronize()
                    tail_seconds = time.monotonic()-t0
                    rgb10 = env.step(np.repeat(tail[0][None], 5, axis=0))
                    if env.step_count != 10:
                        raise RuntimeError("action clock mismatch")
                if not (np.array_equal(tail, np.asarray(expected["tail"], np.float32))
                        and np.array_equal(np.stack(frames), saved_prior[f"r{root}_{method}_causal_velocity_history"])
                        and np.array_equal(rgb10, saved_prior[f"r{root}_{method}_causal_velocity_rgb10"])):
                    raise RuntimeError("prior causal fixed-prefix execution did not reproduce")
                if counts != {"initial_batches": 3, "initial_samples": 48, "tail_batches": 3, "tail_samples": 48}:
                    raise RuntimeError("budget counting mismatch")
                key = f"r{root}_{method}"
                arrays[f"{key}_history"], arrays[f"{key}_rgb10"] = np.stack(frames), rgb10
                starts.append(initial)
                sequences.append(np.concatenate((np.repeat(plan[0][None], 5, axis=0),
                                                  np.repeat(tail[0][None], 5, axis=0))))
                ends.append(rgb10)
                branches.append({"method": method, "initial_plan": plan.tolist(), "initial_cost": cost0,
                                 "initial_trace": trace0, "input5": state5.tolist(), "tail": tail.tolist(),
                                 "tail_cost": cost5, "tail_trace": trace5, "counts": counts,
                                 "jvp_calls": 2 if injected is not None else 0,
                                 "svd_calls": 1 if injected is not None else 0,
                                 "initial_encode_seconds": initial_encode_seconds,
                                 "structure_seconds": structure_seconds, "initial_cem_seconds": initial_seconds,
                                 "feedback_seconds": feedback_seconds, "tail_cem_seconds": tail_seconds,
                                 "rgb10_sha256": image_hash(rgb10)})
                print(f"root {root}, {method}: recompiled and executed", flush=True)
            rows.append({"root": root, "branches": branches})
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model modified")
    legacy_rgb, legacy_endpoints = replay_legacy(starts, sequences, xml)
    endpoints = physical.replay_position_endpoints(np.asarray(starts), np.asarray(sequences))
    if not np.array_equal(legacy_rgb, np.stack(ends)) or not np.array_equal(endpoints[:, :2], legacy_endpoints[:, :2]):
        raise RuntimeError("physical replay mismatch")
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
    arrays["sequences"] = np.asarray(sequences)
    arrays["endpoints_eval_only"] = legacy_endpoints
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a1_pointmaze_full_causal_spectral_dev.v1", "old_sha256": OLD_SHA,
              "original_sha256": ORIGINAL_SHA, "prior_sha256": PRIOR_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "device": torch.cuda.get_device_name(device),
              "model_before": before, "model_after": after, "physical_executions": 8,
              "physical_steps": 80, "initial_prediction_samples": 384, "tail_prediction_samples": 384,
              "predicted_packet_transitions": 1152, "jvp_calls": 8, "svd_calls": 4,
              "separate_replay_trajectories": 8, "wall_seconds": time.monotonic()-started,
              "scope": "known development roots; unequal structure cost; no P1", "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
