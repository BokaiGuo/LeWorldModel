"""Development RGB planner: spectral CEM versus random CEM with extra batch."""
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

ORIGINAL = ROOT / "results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json"
ORIGINAL_SHA = "ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade"
CAMERA = ROOT / "results/a0_pointmaze_camera_metadata_20261103_DEV01.json"
CAMERA_SHA = "1b7e13ee87608cbc2fd5e98f3c42b098d0416c9425ec879e2cdd0bc963fc5ecf"
OLD = ROOT / "results/a1_pointmaze_rgb_state_full_planning_20261103_DEV01.json"
OLD_SHA = "4288b73e675e45526db46698a7abdcd68b619fc235e0a73bc20f91bead6d9b99"
ROOTS = tuple(range(1888, 1896))
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CAUSAL_EXTRA_CEM_BUDGET_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a1_pointmaze_causal_extra_cem_budget_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def cem_four(innovations, scoring):
    """Four-generation random CEM, matching the original update and tie rules."""
    noise = np.asarray(innovations, np.float32)
    if noise.shape != (4, 16, 2, 2) or not np.isfinite(noise).all():
        raise ValueError("invalid innovations")
    mean, std = np.zeros((2, 2), np.float32), np.full((2, 2), .5, np.float32)
    best_cost, best_plan, trace = float("inf"), None, []
    for batch in noise:
        proposals = np.clip(mean + std * batch, -1, 1)
        scores = np.asarray(scoring(proposals), np.float64)
        if scores.shape != (16,) or not np.isfinite(scores).all():
            raise ValueError("invalid CEM scores")
        order = np.argsort(scores, kind="stable")
        if scores[order[0]] < best_cost:
            best_cost, best_plan = float(scores[order[0]]), proposals[order[0]].copy()
        elite = proposals[order[:4]]
        mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), .1)
        trace.append({"proposals": proposals.tolist(), "scores": scores.tolist(),
                      "elite_indices": order[:4].tolist(), "mean": mean.tolist(), "std": std.tolist()})
    return best_plan, best_cost, trace


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((ORIGINAL, ORIGINAL_SHA), (CAMERA, CAMERA_SHA), (OLD, OLD_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    original, camera, old = (json.loads(p.read_text()) for p in (ORIGINAL, CAMERA, OLD))
    for path, expected in ((DEFAULT_CHECKPOINT, original["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", original["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", original["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera["xml_sha256"]:
        raise RuntimeError("scene changed")
    if not np.isclose(mujoco.MjModel.from_xml_string(xml).opt.timestep * 5, .05, rtol=0, atol=1e-12):
        raise RuntimeError("incorrect interval")
    offset = old["body_offset_from_static_scene"]
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    arrays, rows, starts, sequences, terminal_images = {}, [], [], [], []
    started = time.monotonic()
    print("Executing fixed eight development roots: extra random CEM versus spectral CEM.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != old["model_before"]:
            raise RuntimeError("model identity mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid action normalizer")
        for root in ROOTS:
            archive_path = DEFAULT_DATASET / "obses" / f"episode_{root:03d}.pth"
            archive = torch.load(archive_path, map_location="cpu", weights_only=True).numpy()
            goal_rgb = archive[10]
            initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            with NativeRGBBridge(initial) as env:
                rgb0 = env.initial_rgb
            mask, centers = task_mask(rgb0, goal_rgb)
            goal = encode(model, goal_rgb, device)
            rng = np.random.default_rng(20261103 + root)
            initial_noise = rng.standard_normal((4, 16, 2, 2)).astype(np.float32)
            tail_noise = rng.standard_normal((3, 16, 1, 2)).astype(np.float32)
            arrays[f"r{root}_initial_noise"], arrays[f"r{root}_tail_noise"] = initial_noise, tail_noise
            arrays[f"r{root}_rgb0"], arrays[f"r{root}_goal_rgb"], arrays[f"r{root}_mask"] = rgb0, goal_rgb, mask
            branches = []
            for method in ("random_extra", "spectral"):
                with NativeRGBBridge(initial) as env:
                    if not np.array_equal(env.initial_rgb, rgb0):
                        raise RuntimeError("reset RGB mismatch")
                    torch.cuda.synchronize()
                    t0 = time.monotonic()
                    state0, _ = estimate_state(rgb0, camera["metadata"], offset)
                    context = encode_with_state(model, rgb0, state0, device)
                    torch.cuda.synchronize()
                    encode_seconds = time.monotonic()-t0
                    injected, structure_seconds = None, 0.
                    if method == "spectral":
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
                        injected = np.zeros((4, 2, 2), np.float32)
                        injected[:, 0, :] = np.asarray([.3, -.3, .6, -.6], np.float32)[:, None] * vh[0].cpu().numpy()
                        arrays[f"r{root}_{method}_jac"] = jac.cpu().numpy()
                    counts = {"initial_batches": 0, "initial_samples": 0, "tail_batches": 0, "tail_samples": 0}
                    def initial_scorer(proposals):
                        counts["initial_batches"] += 1
                        counts["initial_samples"] += len(proposals)
                        return score_plans(model, context, goal, mask, proposals, mean, std)
                    torch.cuda.synchronize()
                    t0 = time.monotonic()
                    if method == "random_extra":
                        plan, cost0, trace0 = cem_four(initial_noise, initial_scorer)
                    else:
                        plan, cost0, trace0 = cem(initial_noise[:3], initial_scorer, injected)
                    torch.cuda.synchronize()
                    initial_seconds = time.monotonic()-t0
                    frames = [env.initial_rgb]
                    for action in np.repeat(plan[0][None], 5, axis=0):
                        frames.append(env.step(action[None]))
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
                    tail, cost5, trace5 = cem(tail_noise, tail_scorer)
                    torch.cuda.synchronize()
                    tail_seconds = time.monotonic()-t0
                    rgb10 = env.step(np.repeat(tail[0][None], 5, axis=0))
                    if env.step_count != 10:
                        raise RuntimeError("action clock mismatch")
                expected_count = 64 if method == "random_extra" else 48
                if counts != {"initial_batches": expected_count//16, "initial_samples": expected_count,
                              "tail_batches": 3, "tail_samples": 48}:
                    raise RuntimeError("wrong sample counts")
                key = f"r{root}_{method}"
                arrays[f"{key}_history"], arrays[f"{key}_rgb10"] = np.stack(frames), rgb10
                starts.append(initial)
                sequences.append(np.concatenate((np.repeat(plan[0][None], 5, axis=0),
                                                  np.repeat(tail[0][None], 5, axis=0))))
                terminal_images.append(rgb10)
                branches.append({"method": method, "initial_plan": plan.tolist(), "initial_cost": cost0,
                                 "initial_trace": trace0, "injected": None if injected is None else injected.tolist(),
                                 "input5": state5.tolist(), "tail": tail.tolist(), "tail_cost": cost5,
                                 "tail_trace": trace5, "counts": counts,
                                 "jvp_calls": 2 if injected is not None else 0, "svd_calls": 1 if injected is not None else 0,
                                 "initial_encode_seconds": encode_seconds, "structure_seconds": structure_seconds,
                                 "initial_cem_seconds": initial_seconds, "feedback_seconds": feedback_seconds,
                                 "tail_cem_seconds": tail_seconds, "rgb10_sha256": image_hash(rgb10)})
                print(f"root {root} {method}: real RGB closed-loop execution complete", flush=True)
            rows.append({"root": root, "archive_sha256": sha256_file(archive_path), "task_centers": centers,
                         "branches": branches})
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    replay_rgb, legacy_endpoints = replay_legacy(starts, sequences, xml)
    endpoints = physical.replay_position_endpoints(np.asarray(starts), np.asarray(sequences))
    if not np.array_equal(replay_rgb, np.stack(terminal_images)) or not np.array_equal(endpoints[:, :2], legacy_endpoints[:, :2]):
        raise RuntimeError("terminal replay mismatch")
    index = 0
    for row in rows:
        target = dataset.states[row["root"], 10, :2].cpu().numpy().astype(np.float64)
        row["goal_position_eval_only"] = target.tolist()
        for branch in row["branches"]:
            distance = float(np.linalg.norm(legacy_endpoints[index, :2] - target))
            branch["endpoint_eval_only"] = legacy_endpoints[index].tolist()
            branch["distance_eval_only"] = distance
            branch["strict_endpoint_success_eval_only"] = bool(distance < .5)
            index += 1
    arrays["sequences"], arrays["endpoints_eval_only"] = np.asarray(sequences), legacy_endpoints
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a1_pointmaze_causal_extra_cem_budget.v1",
              "source_sha256": ORIGINAL_SHA, "camera_sha256": CAMERA_SHA, "old_sha256": OLD_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "device": torch.cuda.get_device_name(device),
              "model_before": before, "model_after": after, "physical_executions": 16,
              "physical_steps": 160, "separate_replay_trajectories": 16,
              "wall_seconds": time.monotonic()-started,
              "scope": "previously referenced development roots; compute unequal; no P1", "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
