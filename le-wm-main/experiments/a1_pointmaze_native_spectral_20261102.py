"""Frozen-model action Jacobian directions executed in native PointMaze."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_native_feedback_time_20261102 import encode, goal_latent
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file, terminal_cost, terminal_latent,
)

OUTPUT = ROOT / "results/a1_pointmaze_native_spectral_20261102_DEV01.json"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_SPECTRAL_NATIVE_A1_PROTOCOL_20261102.md"
ROOTS = (1880, 1881, 1882, 1883)
EPSILON = 1e-3


def directional_candidates(direction):
    direction = np.asarray(direction, dtype=np.float32)
    if direction.shape != (2,) or not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction), 1, atol=1e-5):
        raise ValueError("direction must be a finite unit vector")
    return np.stack([np.zeros(2, dtype=np.float32), 0.3 * direction, -0.3 * direction,
                     0.6 * direction, -0.6 * direction])


def packet(action, mean, std):
    return ((action - mean) / std).repeat(5).reshape(1, 1, 10)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    started, protocol_sha = time.monotonic(), sha256_file(PROTOCOL)
    rows = []
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        mean = dataset.action_mean.to(device)
        std = dataset.action_std.to(device)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid normalizer")
        axis = np.asarray([[0, 0], [.6, 0], [-.6, 0], [0, .6], [0, -.6]], dtype=np.float32)
        for root in ROOTS:
            path = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
            images = torch.load(path, map_location="cpu", weights_only=True).numpy()
            state = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            with NativeRGBBridge(state) as env:
                initial_rgb = env.initial_rgb
            context = encode(model, initial_rgb, device)
            goal = goal_latent(encode(model, images[10], device))

            def latent(u):
                return terminal_latent(model, context, packet(u, mean, std))[0]

            torch.cuda.synchronize()
            derivative_started = time.monotonic()
            origin = torch.zeros(2, device=device)
            columns = []
            for direction in torch.eye(2, device=device):
                _, column = torch.autograd.functional.jvp(latent, origin, direction)
                columns.append(column.detach())
            jacobian = torch.stack(columns, dim=1)
            if jacobian.shape != (384, 2) or not torch.isfinite(jacobian).all():
                raise RuntimeError("invalid action Jacobian")
            _, singular, vh = torch.linalg.svd(jacobian, full_matrices=False)
            torch.cuda.synchronize()
            derivative_seconds = time.monotonic() - derivative_started
            spectral_direction = vh[0].detach().cpu().numpy()
            with torch.no_grad():
                fd = torch.stack([(latent(EPSILON * direction) - latent(-EPSILON * direction)) / (2 * EPSILON)
                                  for direction in torch.eye(2, device=device)], dim=1)
            fd_error = float(torch.linalg.vector_norm(fd - jacobian) / torch.linalg.vector_norm(jacobian).clamp_min(1e-12))
            rng = np.random.default_rng(20261102 + root)
            random_direction = rng.normal(size=2)
            random_direction = (random_direction / np.linalg.norm(random_direction)).astype(np.float32)
            candidates = {"spectral": directional_candidates(spectral_direction),
                          "random": directional_candidates(random_direction), "axis": axis}
            branches = []
            for method, bank in candidates.items():
                bank_t = torch.as_tensor(bank, device=device)
                normalized = ((bank_t - mean) / std).repeat(1, 5).unsqueeze(0)
                torch.cuda.synchronize()
                selection_started = time.monotonic()
                with torch.no_grad():
                    costs = terminal_cost(model, context, normalized, goal).cpu().numpy()
                torch.cuda.synchronize()
                selection_seconds = time.monotonic() - selection_started
                choice = int(np.argmin(costs))
                with NativeRGBBridge(state) as env:
                    if not np.array_equal(env.initial_rgb, initial_rgb):
                        raise RuntimeError("reset RGB differs across methods")
                    final_rgb = env.step(np.repeat(bank[choice][None], 5, axis=0))
                branches.append({"method": method, "candidate_actions": bank.tolist(), "model_costs": costs.tolist(),
                                 "choice": choice, "selected_action": bank[choice].tolist(),
                                 "rgb5_sha256": hashlib.sha256(final_rgb.tobytes()).hexdigest(),
                                 "prediction_samples": 5, "predicted_block_transitions": 5,
                                 "candidate_selection_seconds": selection_seconds})
            # Simulator goal state and offline endpoints are evaluation-only.
            target = dataset.states[root, 10, :2].cpu().numpy().astype(np.float64)
            sequences = np.stack([np.repeat(np.asarray(branch["selected_action"])[None], 5, axis=0)
                                  for branch in branches])
            endpoints = physical.replay_position_endpoints(np.repeat(state[None], 3, axis=0), sequences)
            for branch, endpoint in zip(branches, endpoints):
                branch["position_eval_only"] = endpoint[:2].tolist()
                branch["distance_eval_only"] = float(np.linalg.norm(endpoint[:2] - target))
            rows.append({"root": root, "initial_rgb_sha256": hashlib.sha256(initial_rgb.tobytes()).hexdigest(),
                         "goal_rgb_sha256": hashlib.sha256(images[10].tobytes()).hexdigest(),
                         "archive_sha256": sha256_file(path), "singular_values": singular.cpu().tolist(),
                         "jacobian": jacobian.cpu().tolist(), "fd_relative_error": fd_error,
                         "spectral_direction": spectral_direction.tolist(), "random_direction": random_direction.tolist(),
                         "derivative_seconds": derivative_seconds, "jvp_calls": 2,
                         "fd_forward_samples_diagnostic_only": 4, "branches": branches})
            print(f"completed development root {root}", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model weights changed")
    report = {
        "schema": "a1_pointmaze_native_spectral.v1",
        "scope": "exposed development roots, unweighted one-block SVD; not P1 or equal-total-budget",
        "protocol_sha256": protocol_sha, "code_sha256": sha256_file(Path(__file__)),
        "bridge_sha256": sha256_file(ROOT / "experiments/pointmaze_native_rgb_bridge_20261102.py"),
        "checkpoint_sha256": sha256_file(args.checkpoint), "model_before": before, "model_after": after,
        "states_sha256": sha256_file(args.dataset_root / "states.pth"),
        "actions_sha256": sha256_file(args.dataset_root / "actions.pth"),
        "normalizer_source": "all archive trajectories, training provenance unknown",
        "goal_and_context_encoder_calls_per_root_shared": 2,
        "device": torch.cuda.get_device_name(device), "wall_seconds": time.monotonic() - started,
        "physical_backend": physical.position_backend_receipt(), "rows": rows,
    }
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
