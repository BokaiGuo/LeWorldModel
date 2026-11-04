"""Spectral injection and matched CEM on exposed PointMaze RGB episodes."""
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
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import OLD, OLD_SHA, OUTPUT as SOURCE
from experiments.a1_pointmaze_native_spectral_20261102 import packet
from experiments.a1_pointmaze_proprio_score_feedback_cem_20261103 import SOURCE_SHA
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state
from experiments.a1_pointmaze_rgb_prefix_causal_tail_20261103 import replay_legacy
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.pc_ap_pointmaze_rgb_dev_20261103 import OUTPUT as AP_SOURCE
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.pc_ap_pointmaze_callback_20261103 import PointMazeAPCallback
from experiments.spectral_reachability.pc_ap_spectral_cem_20261103 import spectral_cem
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_AP_SPECTRAL_CEM_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_ap_spectral_cem_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")
AP_SHA = "48b42f358cde910d4a66758f0c6b57a098e17ad95c87aac982083fafa7cede91"


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (CAMERA, CAMERA_SHA), (OLD, OLD_SHA), (AP_SOURCE, AP_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    source, camera, old, ap = (json.loads(p.read_text()) for p in (SOURCE, CAMERA, OLD, AP_SOURCE))
    if sha256_file(SOURCE.with_suffix(".npz")) != source["arrays_sha256"]:
        raise RuntimeError("source arrays changed")
    if sha256_file(AP_SOURCE.with_suffix(".npz")) != ap["arrays_sha256"]:
        raise RuntimeError("AP arrays changed")
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera["xml_sha256"]:
        raise RuntimeError("scene changed")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, arrays, starts, sequences, finals = [], {}, [], [], []
    started = time.monotonic()
    print("Execute same-CEM random versus spectral injection on eight exposed roots.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved, \
            np.load(AP_SOURCE.with_suffix(".npz"), allow_pickle=False) as prior_ap, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"] or before != ap["model_before"]:
            raise RuntimeError("model mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("invalid normalizer")
        lower = ((torch.full_like(mean, -1.) - mean) / std).repeat(5)
        upper = ((torch.ones_like(mean) - mean) / std).repeat(5)
        prior = ((torch.zeros_like(mean) - mean) / std).repeat(5).reshape(1, 1, 10)
        for root in range(1888, 1896):
            initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            goal_rgb = saved[f"r{root}_goal_rgb"].copy()
            goal_state, _ = estimate_state(goal_rgb, camera["metadata"], old["body_offset_from_static_scene"])
            goal = encode_with_state(model, goal_rgb, goal_state, device)
            branches = []
            for arm in ("random", "spectral"):
                decisions, actions_all = [], []
                with NativeRGBBridge(initial) as env:
                    if not np.array_equal(env.initial_rgb, saved[f"r{root}_rgb0"]):
                        raise RuntimeError("reset image mismatch")
                    frames = [env.initial_rgb]
                    for decision in range(2):
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
                                torch.autograd.functional.jvp(predicted, torch.zeros(2, device=device), direction)[1].detach()
                                for direction in torch.eye(2, device=device)
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
                            arrays[f"r{root}_d{decision}_injection_raw"] = raw_injection.cpu().numpy()
                        callback = PointMazeAPCallback(model, context, goal["proprio"][0, 0], device)
                        generator = torch.Generator(device=device).manual_seed(2026110300 + root * 10 + decision)
                        chosen, prior_cost, refined_cost, refined = spectral_cem(
                            callback, prior, lower, upper, generator, injected)
                        if (callback.batches, callback.samples) != (4, 50):
                            raise RuntimeError("prediction budget mismatch")
                        raw = np.clip((chosen.reshape(5, 2) * std + mean).cpu().numpy(), -1, 1).astype(np.float32)
                        torch.cuda.synchronize()
                        elapsed = time.monotonic() - tic
                        input_hash = image_hash(frames[-1])
                        for action in raw:
                            frames.append(env.step(action[None]))
                        actions_all.extend(raw)
                        decisions.append({"input_rgb_sha256": input_hash,
                                          "output_rgb_sha256": image_hash(frames[-1]),
                                          "actions": raw.tolist(), "initial_cost": prior_cost,
                                          "refined_cost": refined_cost, "refined_selected": refined,
                                          "prediction_batches": callback.batches,
                                          "prediction_samples": callback.samples,
                                          "jvp_calls": 2 if injected is not None else 0,
                                          "svd_calls": 1 if injected is not None else 0,
                                          "structure_seconds": structure_seconds, "planning_seconds": elapsed})
                seq = np.asarray(actions_all, np.float32)
                if root in (1888, 1889) and arm == "random":
                    np.testing.assert_array_equal(seq, prior_ap[f"r{root}_final_goal_actions"])
                    np.testing.assert_array_equal(np.stack(frames), prior_ap[f"r{root}_final_goal_frames"])
                arrays[f"r{root}_{arm}_frames"], arrays[f"r{root}_{arm}_actions"] = np.stack(frames), seq
                starts.append(initial)
                sequences.append(seq)
                finals.append(frames[-1])
                branches.append({"arm": arm, "decisions": decisions, "physical_steps": 10})
                print(f"root {root} {arm}: ten actions executed", flush=True)
            rows.append({"root": root, "branches": branches})
        after = parameter_hash(model)
        if after != before:
            raise RuntimeError("model changed")
    replay_rgb, endpoints = replay_legacy(starts, sequences, xml)
    endpoints2 = physical.replay_position_endpoints(np.asarray(starts), np.asarray(sequences))
    if not np.array_equal(replay_rgb, np.stack(finals)) or not np.array_equal(endpoints[:, :2], endpoints2[:, :2]):
        raise RuntimeError("physical replay mismatch")
    arrays["endpoints_eval_only"] = endpoints
    for i, row in enumerate(rows):
        target = dataset.states[row["root"], 10, :2].cpu().numpy().astype(np.float64)
        row["goal_position_eval_only"] = target.tolist()
        for j, branch in enumerate(row["branches"]):
            endpoint = endpoints[2*i+j]
            distance = float(np.linalg.norm(endpoint[:2] - target))
            branch.update({"endpoint_eval_only": endpoint.tolist(), "distance_eval_only": distance,
                           "strict_endpoint_success_eval_only": bool(distance < .5)})
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "pc_pointmaze_ap_spectral_cem.v1", "rows": rows,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "source_sha256": sha256_file(SOURCE), "prior_ap_sha256": sha256_file(AP_SOURCE),
              "arrays_sha256": sha256_file(ARRAYS), "model_before": before, "model_after": after,
              "physical_executions": len(sequences), "verification_replays": len(sequences),
              "wall_seconds": time.monotonic() - started,
              "scope": "exposed dev roots; equal candidate counts not equal compute; no P1"}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
