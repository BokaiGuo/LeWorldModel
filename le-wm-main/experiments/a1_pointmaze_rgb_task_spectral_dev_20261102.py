"""Generate task-region action-Jacobian candidates and execute RGB feedback."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_archive_bank_alignment_20261102 import SOURCE as BANK_SOURCE, SOURCE_SHA as BANK_SHA
from experiments.a0_pointmaze_native_feedback_time_20261102 import encode
from experiments.a1_pointmaze_native_spectral_20261102 import directional_candidates, packet
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

ROOTS = (1884, 1885, 1886, 1887)
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_RGB_TASK_SPECTRAL_DEV_PROTOCOL_20261102.md"
OUTPUT = ROOT / "results/a1_pointmaze_rgb_task_spectral_dev_20261102_DEV01.json"
ARRAYS = ROOT / "results/a1_pointmaze_rgb_task_spectral_dev_20261102_DEV01.npz"


def marker(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.shape != (224, 224, 3):
        raise ValueError("invalid RGB image")
    r, g, b = (image[:, :, i].astype(np.float32) for i in range(3))
    y, x = np.where((g > 100) & (g > r * 1.25) & (g > b * 1.15) & (r < 100))
    if not 4 <= len(x) <= 500:
        raise ValueError("missing or excessive green-marker pixels")
    if np.ptp(x) >= 20 or np.ptp(y) >= 20:
        raise ValueError("fragmented or oversized marker")
    cx, cy = float(x.mean()), float(y.mean())
    if min(cx, cy, 223 - cx, 223 - cy) < 14:
        raise ValueError("marker too close to image edge")
    return cx, cy


def task_mask(current, goal):
    mask = np.zeros((16, 16), dtype=bool)
    centers = [marker(current), marker(goal)]
    for x, y in centers:
        ix, iy = int(x // 14), int(y // 14)
        mask[max(iy - 1, 0):min(iy + 2, 16), max(ix - 1, 0):min(ix + 2, 16)] = True
    if not mask.any():
        raise ValueError("empty task mask")
    return mask, centers


def weighted_scores(model, context, goal, bank, mean, std, mask):
    suffix = ((torch.as_tensor(bank, device=mean.device) - mean) / std).repeat(1, 5).unsqueeze(0)
    with torch.no_grad():
        visual = model.unroll(context, act_suffix=suffix)["visual"][-1]
        target = goal["visual"][0, 0]
        if visual.shape[1:] != (1, 16, 16, 384) or target.shape != (1, 16, 16, 384):
            raise ValueError("unexpected spatial token shape")
        token_error = ((visual - target) ** 2).mean(dim=(1, 4))
        weights = torch.as_tensor(mask, device=mean.device)
        scores = token_error[:, weights].mean(dim=1)
    if not torch.isfinite(scores).all():
        raise ValueError("non-finite model score")
    return scores.cpu().numpy()


def main():
    if OUTPUT.exists() or ARRAYS.exists() or sha256_file(BANK_SOURCE) != BANK_SHA:
        raise RuntimeError("output exists or tail bank changed")
    bank = np.asarray(json.loads(BANK_SOURCE.read_text())["rows"][0]["candidate_bank"][:8], dtype=np.float32)
    if bank.shape != (8, 2) or not np.isfinite(bank).all() or np.any(np.abs(bank) > 1):
        raise ValueError("invalid tail bank")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    started, rows, arrays = time.monotonic(), [], {}
    print("Executing RGB task-weighted spectral versus full spectral and random prefixes.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid normalizer")
        for root in ROOTS:
            path = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
            images = torch.load(path, map_location="cpu", weights_only=True).numpy()
            initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            with NativeRGBBridge(initial) as env:
                rgb0 = env.initial_rgb
            mask, centers = task_mask(rgb0, images[10])
            context, goal = encode(model, rgb0, device), encode(model, images[10], device)
            def spatial_latent(u):
                output = model.unroll(context, act_suffix=packet(u, mean, std))["visual"][-1]
                if output.shape != (1, 1, 16, 16, 384):
                    raise ValueError("unexpected Jacobian output shape")
                return output[0]
            torch.cuda.synchronize()
            t0 = time.monotonic()
            origin = torch.zeros(2, device=device)
            j = torch.stack([torch.autograd.functional.jvp(spatial_latent, origin, direction)[1].detach()
                             for direction in torch.eye(2, device=device)], dim=-1)
            full = j.reshape(-1, 2)
            selected = j[0, torch.as_tensor(mask, device=device), :, :].reshape(-1, 2)
            _, full_s, full_vh = torch.linalg.svd(full, full_matrices=False)
            _, task_s, task_vh = torch.linalg.svd(selected, full_matrices=False)
            torch.cuda.synchronize()
            structure_seconds = time.monotonic() - t0
            rng = np.random.default_rng(root + 20261102)
            random_dir = rng.normal(size=2)
            random_dir = (random_dir / np.linalg.norm(random_dir)).astype(np.float32)
            banks = {"task_spectral": directional_candidates(task_vh[0].cpu().numpy()),
                     "full_spectral": directional_candidates(full_vh[0].cpu().numpy()),
                     "random": directional_candidates(random_dir)}
            arrays[f"r{root}_jacobian"] = j.cpu().numpy()
            arrays[f"r{root}_initial_rgb"], arrays[f"r{root}_goal_rgb"] = rgb0, images[10]
            arrays[f"r{root}_initial_mask"] = mask
            branches, sequences = [], []
            for method, prefix_bank in banks.items():
                prefix_scores = weighted_scores(model, context, goal, prefix_bank, mean, std, mask)
                prefix_index = int(np.argmin(prefix_scores))
                with NativeRGBBridge(initial) as env:
                    if not np.array_equal(env.initial_rgb, rgb0):
                        raise RuntimeError("reset RGB changed")
                    rgb5 = env.step(np.repeat(prefix_bank[prefix_index][None], 5, axis=0))
                    feedback_mask, feedback_centers = task_mask(rgb5, images[10])
                    feedback = encode(model, rgb5, device)
                    tail_scores = weighted_scores(model, feedback, goal, bank, mean, std, feedback_mask)
                    tail_index = int(np.argmin(tail_scores))
                    rgb10 = env.step(np.repeat(bank[tail_index][None], 5, axis=0))
                key = f"r{root}_{method}"
                arrays[f"{key}_rgb5"], arrays[f"{key}_rgb10"] = rgb5, rgb10
                arrays[f"{key}_feedback_mask"] = feedback_mask
                sequences.append(np.concatenate((np.repeat(prefix_bank[prefix_index][None], 5, axis=0),
                                                 np.repeat(bank[tail_index][None], 5, axis=0))))
                branches.append({"method": method, "prefix_bank": prefix_bank.tolist(),
                                 "prefix_scores": prefix_scores.tolist(), "prefix_index": prefix_index,
                                 "tail_scores": tail_scores.tolist(), "tail_index": tail_index,
                                 "feedback_centers_xy": feedback_centers,
                                 "rgb5_sha256": image_hash(rgb5), "rgb10_sha256": image_hash(rgb10),
                                 "prefix_prediction_samples": 5, "tail_prediction_samples": 8,
                                 "feedback_encodes": 1, "physical_steps": 10})
            target = dataset.states[root, 10, :2].cpu().numpy().astype(np.float64)
            endpoints = physical.replay_position_endpoints(np.repeat(initial[None], 3, axis=0), np.stack(sequences))
            for branch, endpoint in zip(branches, endpoints):
                branch["endpoint_eval_only"] = endpoint.tolist()
                branch["distance_eval_only"] = float(np.linalg.norm(endpoint[:2] - target))
            rows.append({"root": root, "archive_sha256": sha256_file(path), "goal_position_eval_only": target.tolist(),
                         "initial_centers_xy": centers, "initial_rgb_sha256": image_hash(rgb0),
                         "goal_rgb_sha256": image_hash(images[10]), "full_singular_values": full_s.cpu().tolist(),
                         "task_singular_values": task_s.cpu().tolist(), "structure_seconds": structure_seconds,
                         "jvp_calls_shared": 2, "svd_calls": 2, "branches": branches})
            print(f"task-weighted planning root {root} complete", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    helpers = [Path(__file__), ROOT / "experiments/pointmaze_native_rgb_bridge_20261102.py",
               ROOT / "experiments/a1_pointmaze_native_spectral_20261102.py",
               ROOT / "experiments/a0_pointmaze_native_feedback_time_20261102.py"]
    report = {"schema": "a1_pointmaze_rgb_task_spectral_dev.v1", "protocol_sha256": sha256_file(PROTOCOL),
              "source_bank_sha256": BANK_SHA, "arrays_sha256": sha256_file(ARRAYS),
              "helper_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in helpers},
              "checkpoint_sha256": sha256_file(args.checkpoint), "states_sha256": sha256_file(args.dataset_root / "states.pth"),
              "actions_sha256": sha256_file(args.dataset_root / "actions.pth"), "tail_bank": bank.tolist(),
              "action_mean": mean.cpu().tolist(), "action_std": std.cpu().tolist(), "model_before": before,
              "model_after": after, "device": torch.cuda.get_device_name(device),
              "physical_backend": physical.position_backend_receipt(), "wall_seconds": time.monotonic() - started,
              "scope": "exposed development roots; RGB task heuristic; not independent P1", "rows": rows}
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
