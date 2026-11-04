"""Fixed exposed obstacle roots: real-RGB random versus spectral CEM."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA, estimate_state
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import OLD, OLD_SHA
from experiments.a1_pointmaze_native_spectral_20261102 import packet
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.pc_pointmaze_obstacle_v16_replay_20261103 import DATA, PANEL
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.pc_ap_pointmaze_callback_20261103 import PointMazeAPCallback
from experiments.spectral_reachability.pc_ap_spectral_cem_20261103 import spectral_cem
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

ROOTS = (8343, 8345, 8349, 8351)
GOAL = np.array([1.8, 2.8, 0., 0.], dtype=np.float64)
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_OBSTACLE_RGB_SPECTRAL_DEV_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_obstacle_rgb_spectral_dev_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing to overwrite experiment")
    for path, expected in ((CAMERA, CAMERA_SHA), (OLD, OLD_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"input changed: {path}")
    camera, old = (json.loads(p.read_text()) for p in (CAMERA, OLD))
    manifest = json.loads((DATA / "challenge_manifest.json").read_text())
    panel = json.loads(PANEL.read_text())
    if not set(ROOTS).issubset(panel["selected_roots"]):
        raise RuntimeError("fixed roots absent from source panel")
    if sha256_file(DATA / "states.pth") != manifest["states_sha256"]:
        raise RuntimeError("challenge state hash mismatch")
    if not np.array_equal(np.asarray(manifest["challenge_goal_qpos"]), GOAL[:2]):
        raise RuntimeError("goal mismatch")
    if hashlib.sha256(official_xml().encode()).hexdigest() != camera["xml_sha256"]:
        raise RuntimeError("scene mismatch")
    states = torch.load(DATA / "states.pth", weights_only=True, map_location="cpu")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, arrays, starts, sequences, terminal = [], {}, [], [], []
    started = time.monotonic()
    print("Compare spectral and random CEM on four exposed obstacle roots.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("invalid action normalization")
        lower = ((torch.full_like(mean, -1.) - mean) / std).repeat(5)
        upper = ((torch.ones_like(mean) - mean) / std).repeat(5)
        prior = ((torch.zeros_like(mean) - mean) / std).repeat(5).reshape(1, 1, 10)
        with NativeRGBBridge(GOAL) as env:
            goal_rgb = env.initial_rgb.copy()
        goal_state, _ = estimate_state(goal_rgb, camera["metadata"], old["body_offset_from_static_scene"])
        goal = encode_with_state(model, goal_rgb, goal_state, device)
        arrays["goal_rgb"] = goal_rgb
        for root in ROOTS:
            initial = states[root, 40].numpy().astype(np.float64)
            branches = []
            for arm in ("random", "spectral"):
                decisions, actions = [], []
                with NativeRGBBridge(initial) as env:
                    frames = [env.initial_rgb]
                    for decision in range(5):
                        torch.cuda.synchronize()
                        tic = time.monotonic()
                        state = (estimate_state(frames[-1], camera["metadata"],
                                                old["body_offset_from_static_scene"])[0]
                                 if decision == 0 else causal_state(
                                     frames[-2], frames[-1], camera["metadata"],
                                     old["body_offset_from_static_scene"], .05))
                        context = encode_with_state(model, frames[-1], state, device)
                        injected, structure_seconds = None, 0.
                        if arm == "spectral":
                            def predicted(action):
                                return model.unroll(context, act_suffix=packet(action, mean, std))["proprio"][-1][0]
                            torch.cuda.synchronize()
                            struct_start = time.monotonic()
                            jac = torch.stack([
                                torch.autograd.functional.jvp(predicted, torch.zeros(2, device=device), d)[1].detach()
                                for d in torch.eye(2, device=device)
                            ], dim=-1).reshape(-1, 2)
                            if not torch.isfinite(jac).all():
                                raise ValueError("invalid Jacobian")
                            _, singular, vh = torch.linalg.svd(jac, full_matrices=False)
                            raw_injection = torch.tensor([.3, -.3, .6, -.6], device=device)[:, None] * vh[0]
                            injected = ((raw_injection - mean) / std).repeat(1, 5)
                            torch.cuda.synchronize()
                            structure_seconds = time.monotonic() - struct_start
                            arrays[f"r{root}_d{decision}_jacobian"] = jac.cpu().numpy()
                            arrays[f"r{root}_d{decision}_singular"] = singular.cpu().numpy()
                        callback = PointMazeAPCallback(model, context, goal["proprio"][0, 0], device)
                        generator = torch.Generator(device=device).manual_seed(2026110300 + root * 10 + decision)
                        chosen, prior_cost, refined_cost, refined = spectral_cem(
                            callback, prior, lower, upper, generator, injected)
                        if (callback.batches, callback.samples) != (4, 50):
                            raise RuntimeError("candidate budget mismatch")
                        raw = np.clip((chosen.reshape(5, 2) * std + mean).cpu().numpy(), -1, 1).astype(np.float32)
                        torch.cuda.synchronize()
                        elapsed = time.monotonic() - tic
                        input_hash = image_hash(frames[-1])
                        for action in raw:
                            frames.append(env.step(action[None]))
                        actions.extend(raw)
                        decisions.append({"input_rgb_sha256": input_hash, "output_rgb_sha256": image_hash(frames[-1]),
                                          "initial_cost": prior_cost, "refined_cost": refined_cost,
                                          "refined_selected": refined, "prediction_batches": callback.batches,
                                          "prediction_samples": callback.samples, "jvp_calls": 2 if injected is not None else 0,
                                          "svd_calls": 1 if injected is not None else 0,
                                          "structure_seconds": structure_seconds, "planning_seconds": elapsed})
                    if env.step_count != 25:
                        raise RuntimeError("action clock mismatch")
                seq = np.asarray(actions, np.float32)
                arrays[f"r{root}_{arm}_frames"] = np.stack(frames)
                arrays[f"r{root}_{arm}_actions"] = seq
                starts.append(initial)
                sequences.append(seq)
                terminal.append(frames[-1])
                branches.append({"arm": arm, "physical_steps": 25, "decisions": decisions})
                print(f"root {root} {arm}: 25 actions completed", flush=True)
            rows.append({"root": root, "branches": branches})
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("frozen model changed")
    # New bridge instances replay the sealed actions; the second backend evaluates positions.
    endpoints = physical.replay_position_endpoints(np.asarray(starts), np.asarray(sequences))
    for i, (initial, seq, expected) in enumerate(zip(starts, sequences, terminal)):
        with NativeRGBBridge(initial) as env:
            for action in seq:
                frame = env.step(action[None])
            if not np.array_equal(frame, expected):
                raise RuntimeError(f"terminal RGB replay mismatch at {i}")
    arrays["endpoints_eval_only"] = endpoints
    for i, row in enumerate(rows):
        for j, branch in enumerate(row["branches"]):
            end = endpoints[2*i+j]
            distance = float(np.linalg.norm(end[:2] - GOAL[:2]))
            branch.update({"endpoint_eval_only": end.tolist(), "distance_eval_only": distance,
                           "offline_endpoint_le_025": bool(distance <= .25)})
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "pc_pointmaze_obstacle_rgb_spectral_dev.v1", "rows": rows,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "manifest_sha256": sha256_file(DATA / "challenge_manifest.json"),
              "states_sha256": sha256_file(DATA / "states.pth"), "panel_sha256": sha256_file(PANEL),
              "camera_sha256": CAMERA_SHA, "checkpoint_sha256": sha256_file(DEFAULT_CHECKPOINT),
              "arrays_sha256": sha256_file(ARRAYS), "model_before": before, "model_after": after,
              "physical_executions": len(sequences), "rgb_replays": len(sequences),
              "position_evaluations": len(sequences), "wall_seconds": time.monotonic() - started,
              "scope": "exposed development roots; unequal total computation; no P1"}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
