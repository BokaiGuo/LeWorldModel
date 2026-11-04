"""Common CEM optimizer with and without task-Jacobian initialization."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_native_feedback_time_20261102 import encode
from experiments.a1_pointmaze_native_spectral_20261102 import packet
from experiments.a1_pointmaze_rgb_task_spectral_dev_20261102 import task_mask
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

ROOTS = (1884, 1885, 1886, 1887)
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_TASK_SPECTRAL_CEM_DEV_PROTOCOL_20261102.md"
OUTPUT = ROOT / "results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json"
ARRAYS = ROOT / "results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.npz"


def cem(innovations, scoring, injected=None):
    innovations = np.asarray(innovations, dtype=np.float32)
    if innovations.ndim != 4 or innovations.shape[:2] != (3, 16) or innovations.shape[-1] != 2:
        raise ValueError("expected [3,16,horizon,2] innovations")
    if not np.isfinite(innovations).all():
        raise ValueError("non-finite innovations")
    horizon = innovations.shape[2]
    if horizon not in (1, 2):
        raise ValueError("invalid horizon")
    mean, std = np.zeros((horizon, 2), np.float32), np.full((horizon, 2), .5, np.float32)
    best_cost, best_plan, trace = float("inf"), None, []
    for iteration in range(3):
        proposals = np.clip(mean + std * innovations[iteration], -1, 1)
        if iteration == 0 and injected is not None:
            injected = np.asarray(injected, dtype=np.float32)
            if horizon != 2 or injected.shape != (4, 2, 2) or not np.isfinite(injected).all() or np.any(np.abs(injected) > 1):
                raise ValueError("invalid injected proposals")
            proposals[:4] = injected
        values = np.asarray(scoring(proposals), dtype=np.float64)
        if values.shape != (16,) or not np.isfinite(values).all():
            raise ValueError("invalid CEM scores")
        order = np.argsort(values, kind="stable")
        if values[order[0]] < best_cost:
            best_cost, best_plan = float(values[order[0]]), proposals[order[0]].copy()
        elite = proposals[order[:4]]
        mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), .1)
        trace.append({"proposals": proposals.tolist(), "scores": values.tolist(),
                      "elite_indices": order[:4].tolist(), "mean": mean.tolist(), "std": std.tolist()})
    return best_plan, best_cost, trace


def score_plans(model, context, goal, mask, proposals, mean, std):
    plans = torch.as_tensor(proposals, dtype=torch.float32, device=mean.device)
    normalized = (plans - mean) / std
    suffix = normalized.repeat(1, 1, 5).permute(1, 0, 2)
    with torch.no_grad():
        output = model.unroll(context, act_suffix=suffix)["visual"][-1]
        target = goal["visual"][0, 0]
        if output.shape[1:] != (1, 16, 16, 384):
            raise ValueError("unexpected model visual tensor")
        error = ((output - target) ** 2).mean(dim=(1, 4))
        result = error[:, torch.as_tensor(mask, device=mean.device)].mean(dim=1)
    return result.cpu().numpy()


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, arrays, started = [], {}, time.monotonic()
    print("Executing common CEM with and without spectral initialization; real RGB feedback.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid normalizer")
        for root in ROOTS:
            path = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
            archive = torch.load(path, map_location="cpu", weights_only=True).numpy()
            state = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            with NativeRGBBridge(state) as env:
                rgb0 = env.initial_rgb
            mask, centers = task_mask(rgb0, archive[10])
            context, goal = encode(model, rgb0, device), encode(model, archive[10], device)
            def predicted_visual(u):
                result = model.unroll(context, act_suffix=packet(u, mean, std))["visual"][-1]
                if result.shape != (1, 1, 16, 16, 384):
                    raise ValueError("bad predicted tensor")
                return result[0]
            torch.cuda.synchronize()
            structure_start = time.monotonic()
            zero = torch.zeros(2, device=device)
            jac = torch.stack([torch.autograd.functional.jvp(predicted_visual, zero, d)[1].detach()
                               for d in torch.eye(2, device=device)], dim=-1)
            selected = jac[0, torch.as_tensor(mask, device=device), :, :].reshape(-1, 2)
            _, singular, vh = torch.linalg.svd(selected, full_matrices=False)
            torch.cuda.synchronize()
            structure_seconds = time.monotonic() - structure_start
            direction = vh[0].cpu().numpy()
            injected = np.zeros((4, 2, 2), dtype=np.float32)
            injected[:, 0, :] = np.asarray([.3, -.3, .6, -.6], dtype=np.float32)[:, None] * direction
            rng = np.random.default_rng(20261103 + root)
            initial_noise = rng.standard_normal((3, 16, 2, 2)).astype(np.float32)
            tail_noise = rng.standard_normal((3, 16, 1, 2)).astype(np.float32)
            arrays[f"r{root}_jacobian"] = jac.cpu().numpy()
            arrays[f"r{root}_rgb0"], arrays[f"r{root}_goal_rgb"] = rgb0, archive[10]
            arrays[f"r{root}_initial_mask"] = mask
            arrays[f"r{root}_initial_noise"], arrays[f"r{root}_tail_noise"] = initial_noise, tail_noise
            branches, sequences = [], []
            for method in ("random_cem", "spectral_cem"):
                torch.cuda.synchronize()
                plan_start = time.monotonic()
                plan, cost, trace = cem(initial_noise,
                                        lambda x: score_plans(model, context, goal, mask, x, mean, std),
                                        injected if method == "spectral_cem" else None)
                torch.cuda.synchronize()
                initial_seconds = time.monotonic() - plan_start
                with NativeRGBBridge(state) as env:
                    if not np.array_equal(env.initial_rgb, rgb0):
                        raise RuntimeError("reset RGB mismatch")
                    rgb5 = env.step(np.repeat(plan[0][None], 5, axis=0))
                    torch.cuda.synchronize()
                    tail_start = time.monotonic()
                    feedback_mask, feedback_centers = task_mask(rgb5, archive[10])
                    feedback = encode(model, rgb5, device)
                    tail, tail_cost, tail_trace = cem(
                        tail_noise, lambda x: score_plans(model, feedback, goal, feedback_mask, x, mean, std))
                    torch.cuda.synchronize()
                    tail_seconds = time.monotonic() - tail_start
                    rgb10 = env.step(np.repeat(tail[0][None], 5, axis=0))
                key = f"r{root}_{method}"
                arrays[f"{key}_rgb5"], arrays[f"{key}_rgb10"] = rgb5, rgb10
                arrays[f"{key}_feedback_mask"] = feedback_mask
                sequences.append(np.concatenate((np.repeat(plan[0][None], 5, axis=0),
                                                 np.repeat(tail[0][None], 5, axis=0))))
                branches.append({"method": method, "initial_plan": plan.tolist(), "initial_cost": cost,
                                 "initial_trace": trace, "tail_plan": tail.tolist(), "tail_cost": tail_cost,
                                 "tail_trace": tail_trace, "rgb5_sha256": image_hash(rgb5),
                                 "rgb10_sha256": image_hash(rgb10), "feedback_centers_xy": feedback_centers,
                                 "initial_seconds": initial_seconds, "feedback_and_tail_seconds": tail_seconds,
                                 "initial_prediction_samples": 48, "tail_prediction_samples": 48,
                                 "predicted_block_transitions": 144, "feedback_encodes": 1,
                                 "attributable_jvp_calls": 2 if method == "spectral_cem" else 0,
                                 "attributable_svd_calls": 1 if method == "spectral_cem" else 0,
                                 "physical_steps": 10})
            if np.array_equal(np.asarray(branches[0]["initial_plan"])[0], np.asarray(branches[1]["initial_plan"])[0]):
                if branches[0]["rgb5_sha256"] != branches[1]["rgb5_sha256"]:
                    raise RuntimeError("matched prefix RGB differs")
            target = dataset.states[root, 10, :2].cpu().numpy().astype(np.float64)
            endpoints = physical.replay_position_endpoints(np.repeat(state[None], 2, axis=0), np.stack(sequences))
            for branch, endpoint in zip(branches, endpoints):
                branch["endpoint_eval_only"] = endpoint.tolist()
                branch["distance_eval_only"] = float(np.linalg.norm(endpoint[:2] - target))
            rows.append({"root": root, "archive_sha256": sha256_file(path), "initial_centers_xy": centers,
                         "goal_position_eval_only": target.tolist(), "task_singular_values": singular.cpu().tolist(),
                         "spectral_direction": direction.tolist(), "structure_seconds": structure_seconds,
                         "injected_proposals": injected.tolist(), "branches": branches})
            print(f"common CEM development root {root} completed", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    helpers = [Path(__file__), ROOT / "experiments/a1_pointmaze_rgb_task_spectral_dev_20261102.py",
               ROOT / "experiments/pointmaze_native_rgb_bridge_20261102.py",
               ROOT / "experiments/a1_pointmaze_native_spectral_20261102.py"]
    report = {"schema": "a1_pointmaze_task_spectral_cem_dev.v1",
              "protocol_sha256": sha256_file(PROTOCOL), "arrays_sha256": sha256_file(ARRAYS),
              "helpers_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in helpers},
              "checkpoint_sha256": sha256_file(args.checkpoint), "states_sha256": sha256_file(args.dataset_root / "states.pth"),
              "actions_sha256": sha256_file(args.dataset_root / "actions.pth"),
              "model_before": before, "model_after": after, "device": torch.cuda.get_device_name(device),
              "physical_backend": physical.position_backend_receipt(), "wall_seconds": time.monotonic() - started,
              "scope": "exposed roots; task RGB heuristic; unequal total compute; not independent P1", "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
