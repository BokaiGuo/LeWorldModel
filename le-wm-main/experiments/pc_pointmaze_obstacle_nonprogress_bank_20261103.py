"""Fixed spectral bank with model-only non-progress selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA, estimate_state
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import OLD, OLD_SHA
from experiments.a1_pointmaze_native_spectral_20261102 import packet
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import DATA, GOAL, ROOTS, OUTPUT as SOURCE
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.pc_ap_pointmaze_callback_20261103 import PointMazeAPCallback
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE_SHA = "e5955123b5b88797a66e4c703a1c0f5b321a23c8fbead371dc029c8d44a5ff91"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_OBSTACLE_NONPROGRESS_BANK_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_obstacle_nonprogress_bank_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def choose_nonprogress(scores, current):
    scores = np.asarray(scores, np.float64)
    if scores.shape != (9,) or not np.isfinite(scores).all() or not np.isfinite(current):
        raise ValueError("invalid scores")
    eligible = np.flatnonzero(scores[1:] >= current) + 1
    selected = int(eligible[np.argmin(scores[eligible])]) if len(eligible) else None
    return selected, eligible.tolist(), int(np.argmin(scores))


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (CAMERA, CAMERA_SHA), (OLD, OLD_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"input changed: {path}")
    source = json.loads(SOURCE.read_text())
    if (sha256_file(SOURCE.with_suffix(".npz")) != source["arrays_sha256"]
            or sha256_file(DATA / "states.pth") != source["states_sha256"]
            or sha256_file(DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]):
        raise RuntimeError("source inputs changed")
    camera, old = (json.loads(p.read_text()) for p in (CAMERA, OLD))
    states = torch.load(DATA / "states.pth", weights_only=True, map_location="cpu")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, arrays, starts, sequences, ends = [], {}, [], [], []
    started = time.monotonic()
    print("Select model-only non-progress directions and execute complete fixed bank.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise RuntimeError("model identity differs")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or not (std > 0).all():
            raise RuntimeError("invalid action normalization")
        goal_rgb = saved["goal_rgb"]
        goal_state, _ = estimate_state(goal_rgb, camera["metadata"], old["body_offset_from_static_scene"])
        goal = encode_with_state(model, goal_rgb, goal_state, device)
        for root in ROOTS:
            initial = states[root, 40].numpy().astype(np.float64)
            rgb = saved[f"r{root}_random_frames"][0]
            state, _ = estimate_state(rgb, camera["metadata"], old["body_offset_from_static_scene"])
            context = encode_with_state(model, rgb, state, device)
            target = goal["proprio"][0, 0]
            current = float((context["proprio"][0, 0] - target).square().mean())
            def predicted(action):
                return model.unroll(context, act_suffix=packet(action, mean, std))["proprio"][-1][0]
            torch.cuda.synchronize()
            tic = time.monotonic()
            jac = torch.stack([torch.autograd.functional.jvp(
                predicted, torch.zeros(2, device=device), direction)[1].detach()
                for direction in torch.eye(2, device=device)], dim=-1).reshape(-1, 2)
            if not torch.isfinite(jac).all():
                raise RuntimeError("invalid Jacobian")
            _, singular, vh = torch.linalg.svd(jac, full_matrices=False)
            bank = torch.stack([torch.zeros(2, device=device)] +
                               [m * vh[i] for i in range(2) for m in (.3, -.3, .6, -.6)])
            normalized = ((bank-mean)/std).repeat(1, 5).unsqueeze(0).unsqueeze(2)
            callback = PointMazeAPCallback(model, context, target, device)
            scores = callback(normalized)[0].cpu().numpy()
            selected, eligible, greedy = choose_nonprogress(scores, current)
            torch.cuda.synchronize()
            plan_seconds = time.monotonic()-tic
            raw_bank = bank.detach().cpu().numpy().astype(np.float32)
            arrays[f"r{root}_bank"] = raw_bank
            arrays[f"r{root}_jacobian"] = jac.cpu().numpy()
            arrays[f"r{root}_singular"] = singular.cpu().numpy()
            arrays[f"r{root}_scores"] = scores
            images = []
            for action in raw_bank:
                with NativeRGBBridge(initial) as env:
                    if not np.array_equal(env.initial_rgb, rgb):
                        raise RuntimeError("initial RGB mismatch")
                    for _ in range(5):
                        image = env.step(action[None])
                images.append(image)
                starts.append(initial)
                sequences.append(np.repeat(action[None], 5, axis=0))
                ends.append(image)
            arrays[f"r{root}_rgb5"] = np.stack(images)
            rows.append({"root": root, "current_cost": current, "scores": scores.tolist(),
                         "selected": selected, "eligible": eligible, "greedy": greedy,
                         "jvp_calls": 2, "svd_calls": 1, "candidate_predictions": callback.samples,
                         "planning_seconds": plan_seconds})
            print(f"root {root}: selected={selected}, eligible={len(eligible)}", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    starts, sequences = np.asarray(starts), np.asarray(sequences)
    cp5 = physical.replay_position_endpoints(starts, sequences)
    arrays["cp5_eval_only"] = cp5
    for i, row in enumerate(rows):
        start = starts[9*i]
        row["initial_distance_eval_only"] = float(np.linalg.norm(start[:2]-GOAL[:2]))
        row["distances_eval_only"] = [
            float(np.linalg.norm(cp5[9*i+j, :2]-GOAL[:2])) for j in range(9)]
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "pc_pointmaze_obstacle_nonprogress_bank.v1", "rows": rows,
              "source_sha256": SOURCE_SHA, "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "model_before": before, "model_after": after, "physical_executions": len(sequences),
              "wall_seconds": time.monotonic()-started, "scope": "exposed development roots; no P1"}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
