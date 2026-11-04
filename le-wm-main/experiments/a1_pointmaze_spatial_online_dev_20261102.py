"""Execute paired pooled/spatial selectors with real RGB feedback on fixed roots."""
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
    parameter_hash, require_cuda, sha256_file, terminal_latent,
)

ROOTS = (1884, 1885, 1886, 1887)
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_SPATIAL_ONLINE_DEV_PROTOCOL_20261102.md"
OUTPUT = ROOT / "results/a1_pointmaze_spatial_online_dev_20261102_DEV01.json"
ARRAYS = ROOT / "results/a1_pointmaze_spatial_online_dev_20261102_DEV01.npz"


def predict_scores(model, context, goal, bank, mean, std):
    bank_t = torch.as_tensor(bank, device=mean.device)
    suffix = ((bank_t - mean) / std).repeat(1, 5).unsqueeze(0)
    with torch.no_grad():
        visual = model.unroll(context, act_suffix=suffix)["visual"][-1]
        target = goal["visual"][0, 0]
        if visual.shape[1:] != target.shape or visual.shape[0] != len(bank):
            raise ValueError("prediction and target shapes mismatch")
        pooled = ((visual.mean(dim=(1, 2, 3)) - target.mean(dim=(0, 1, 2))) ** 2).mean(dim=1)
        spatial = ((visual - target) ** 2).mean(dim=(1, 2, 3, 4))
    if not torch.isfinite(pooled).all() or not torch.isfinite(spatial).all():
        raise ValueError("non-finite scores")
    return pooled.cpu().numpy(), spatial.cpu().numpy()


def main():
    if OUTPUT.exists() or ARRAYS.exists() or sha256_file(BANK_SOURCE) != BANK_SHA:
        raise RuntimeError("output exists or sealed tail bank changed")
    tail_bank = np.asarray(json.loads(BANK_SOURCE.read_text())["rows"][0]["candidate_bank"][:8], dtype=np.float32)
    if tail_bank.shape != (8, 2) or not np.isfinite(tail_bank).all() or np.any(np.abs(tail_bank) > 1):
        raise ValueError("invalid tail bank")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, arrays, started = [], {}, time.monotonic()
    print("Executing fixed-root paired pooled/spatial planning with real RGB feedback.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid normalizer")
        axis = np.asarray([[0, 0], [.6, 0], [-.6, 0], [0, .6], [0, -.6]], dtype=np.float32)
        for root in ROOTS:
            image_path = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
            images = torch.load(image_path, map_location="cpu", weights_only=True).numpy()
            state = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            goal = encode(model, images[10], device)
            with NativeRGBBridge(state) as env:
                rgb0 = env.initial_rgb
            context = encode(model, rgb0, device)
            def latent(u):
                return terminal_latent(model, context, packet(u, mean, std))[0]
            torch.cuda.synchronize()
            t0 = time.monotonic()
            origin = torch.zeros(2, device=device)
            columns = [torch.autograd.functional.jvp(latent, origin, direction)[1].detach()
                       for direction in torch.eye(2, device=device)]
            jacobian = torch.stack(columns, dim=1)
            if jacobian.shape != (384, 2) or not torch.isfinite(jacobian).all():
                raise ValueError("invalid Jacobian")
            _, singular, vh = torch.linalg.svd(jacobian, full_matrices=False)
            torch.cuda.synchronize()
            derivative_seconds = time.monotonic() - t0
            rng = np.random.default_rng(20261102 + root)
            direction = rng.normal(size=2)
            direction = (direction / np.linalg.norm(direction)).astype(np.float32)
            banks = {"spectral": directional_candidates(vh[0].detach().cpu().numpy()),
                     "random": directional_candidates(direction), "axis": axis}
            arrays[f"r{root}_goal_rgb"], arrays[f"r{root}_initial_rgb"] = images[10], rgb0
            branches, sequences = [], []
            for method, bank in banks.items():
                prefix_pooled, prefix_spatial = predict_scores(model, context, goal, bank, mean, std)
                for metric, prefix_scores in (("pooled", prefix_pooled), ("spatial", prefix_spatial)):
                    selected_prefix = int(np.argmin(prefix_scores))
                    with NativeRGBBridge(state) as env:
                        if not np.array_equal(env.initial_rgb, rgb0):
                            raise RuntimeError("reset RGB mismatch")
                        rgb5 = env.step(np.repeat(bank[selected_prefix][None], 5, axis=0))
                        torch.cuda.synchronize()
                        tail_start = time.monotonic()
                        feedback = encode(model, rgb5, device)
                        tail_pooled, tail_spatial = predict_scores(model, feedback, goal, tail_bank, mean, std)
                        torch.cuda.synchronize()
                        feedback_plan_seconds = time.monotonic() - tail_start
                        tail_scores = tail_pooled if metric == "pooled" else tail_spatial
                        selected_tail = int(np.argmin(tail_scores))
                        rgb10 = env.step(np.repeat(tail_bank[selected_tail][None], 5, axis=0))
                    key = f"r{root}_{method}_{metric}"
                    arrays[f"{key}_rgb5"], arrays[f"{key}_rgb10"] = rgb5, rgb10
                    sequences.append(np.concatenate((np.repeat(bank[selected_prefix][None], 5, axis=0),
                                                     np.repeat(tail_bank[selected_tail][None], 5, axis=0))))
                    branches.append({"method": method, "metric": metric,
                                     "prefix_index": selected_prefix, "prefix_action": bank[selected_prefix].tolist(),
                                     "prefix_pooled_costs": prefix_pooled.tolist(),
                                     "prefix_spatial_costs": prefix_spatial.tolist(),
                                     "tail_index": selected_tail, "tail_action": tail_bank[selected_tail].tolist(),
                                     "tail_pooled_costs": tail_pooled.tolist(), "tail_spatial_costs": tail_spatial.tolist(),
                                     "rgb5_sha256": image_hash(rgb5), "rgb10_sha256": image_hash(rgb10),
                                     "feedback_encode_and_prediction_seconds": feedback_plan_seconds,
                                     "feedback_encodes": 1, "tail_prediction_samples": 8})
                p, s = branches[-2:]
                if p["prefix_index"] == s["prefix_index"] and p["rgb5_sha256"] != s["rgb5_sha256"]:
                    raise RuntimeError("same prefix failed feedback replay")
            # Privileged goal position only enters after every branch has decided.
            target = dataset.states[root, 10, :2].cpu().numpy().astype(np.float64)
            endpoints = physical.replay_position_endpoints(np.repeat(state[None], 6, axis=0), np.stack(sequences))
            for branch, endpoint in zip(branches, endpoints):
                branch["endpoint_eval_only"] = endpoint.tolist()
                branch["distance_eval_only"] = float(np.linalg.norm(endpoint[:2] - target))
            rows.append({"root": root, "goal_position_eval_only": target.tolist(),
                         "archive_sha256": sha256_file(image_path), "initial_rgb_sha256": image_hash(rgb0),
                         "goal_rgb_sha256": image_hash(images[10]), "branches": branches,
                         "singular_values": singular.cpu().tolist(), "jacobian": jacobian.cpu().tolist(),
                         "spectral_jvp_calls": 2, "spectral_svd_calls": 1,
                         "spectral_derivative_seconds": derivative_seconds,
                         "shared_prefix_prediction_samples_per_bank": 5})
            print(f"completed root {root}", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("frozen model changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a1_pointmaze_spatial_online_dev.v1", "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "tail_bank_source_sha256": BANK_SHA,
              "arrays_sha256": sha256_file(ARRAYS), "checkpoint_sha256": sha256_file(args.checkpoint),
              "states_sha256": sha256_file(args.dataset_root / "states.pth"),
              "actions_sha256": sha256_file(args.dataset_root / "actions.pth"),
              "model_before": before, "model_after": after, "tail_bank": tail_bank.tolist(),
              "action_mean": mean.cpu().tolist(), "action_std": std.cpu().tolist(),
              "scope": "development roots, fixed zero proprio, unknown training provenance, not P1",
              "device": torch.cuda.get_device_name(device), "physical_backend": physical.position_backend_receipt(),
              "wall_seconds": time.monotonic() - started, "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
