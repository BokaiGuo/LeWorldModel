"""Execute sealed A1 prefixes and replan a common tail from real step-5 RGB."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_archive_bank_alignment_20261102 import SOURCE as BANK_SOURCE, SOURCE_SHA as BANK_SHA, score
from experiments.a0_pointmaze_native_feedback_time_20261102 import encode, goal_latent
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE = ROOT / "results/a1_pointmaze_native_spectral_20261102_DEV01.json"
SOURCE_SHA = "2c61566337ae9915807e724913143e067a46ddfa519431de804ee21b452eed4a"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_SPECTRAL_CP5_REPLAN_PROTOCOL_20261102.md"
OUTPUT = ROOT / "results/a1_pointmaze_spectral_cp5_replan_20261102_DEV01.json"
ARRAYS = ROOT / "results/a1_pointmaze_spectral_cp5_replan_20261102_DEV01.npz"
METHODS = ("spectral", "random", "axis", "zero")


def validate_bank(value, shape):
    bank = np.asarray(value, dtype=np.float32)
    if bank.shape != shape or not np.isfinite(bank).all() or np.any(np.abs(bank) > 1):
        raise ValueError("invalid or out-of-bounds physical actions")
    return bank


def choose(costs):
    costs = np.asarray(costs)
    if costs.shape != (8,) or not np.isfinite(costs).all():
        raise ValueError("expected eight finite costs")
    return int(np.argmin(costs))


def image_hash(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing to overwrite result or array archive")
    if sha256_file(SOURCE) != SOURCE_SHA or sha256_file(BANK_SOURCE) != BANK_SHA:
        raise RuntimeError("sealed source hash mismatch")
    original = json.loads(SOURCE.read_text())
    if (original["checkpoint_sha256"] != sha256_file(DEFAULT_CHECKPOINT)
            or original["states_sha256"] != sha256_file(DEFAULT_DATASET / "states.pth")
            or original["actions_sha256"] != sha256_file(DEFAULT_DATASET / "actions.pth")):
        raise RuntimeError("checkpoint or dataset differs from prefix source")
    bank = validate_bank(json.loads(BANK_SOURCE.read_text())["rows"][0]["candidate_bank"][:8], (8, 2))
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    started, rows, arrays = time.monotonic(), [], {}
    print("Executing sealed prefixes and selecting identical-budget tails using actual RGB.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != original["model_before"]:
            raise RuntimeError("frozen model differs from source")
        mean, std = dataset.action_mean.cpu().numpy(), dataset.action_std.cpu().numpy()
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("invalid action normalizer")
        normalized = np.repeat(((bank - mean) / std)[:, None, :], 5, axis=1).reshape(1, 8, 10)
        packets = torch.as_tensor(normalized, device=device)
        for source_row in original["rows"]:
            root = int(source_row["root"])
            path = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
            if sha256_file(path) != source_row["archive_sha256"]:
                raise RuntimeError("archive RGB changed")
            images = torch.load(path, map_location="cpu", weights_only=True).numpy()
            if image_hash(images[10]) != source_row["goal_rgb_sha256"]:
                raise RuntimeError("goal RGB mismatch")
            initial_state = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            goal = goal_latent(encode(model, images[10], device))
            arrays[f"r{root}_goal_rgb"] = images[10]
            arrays[f"r{root}_goal_latent"] = goal.cpu().numpy()
            by_method = {b["method"]: b for b in source_row["branches"]}
            if set(by_method) != set(METHODS[:-1]):
                raise RuntimeError("source branch roster mismatch")
            branches, sequences = [], []
            for method in METHODS:
                previous = by_method.get(method)
                prefix = validate_bank(previous["selected_action"] if previous else [0, 0], (2,))
                with NativeRGBBridge(initial_state) as env:
                    if image_hash(env.initial_rgb) != source_row["initial_rgb_sha256"]:
                        raise RuntimeError("initial RGB differs from source")
                    rgb5 = env.step(np.repeat(prefix[None], 5, axis=0))
                    if previous and image_hash(rgb5) != previous["rgb5_sha256"]:
                        raise RuntimeError("prefix replay differs from source")
                    torch.cuda.synchronize()
                    t0 = time.monotonic()
                    context = encode(model, rgb5, device)
                    costs = score(model, context, goal, packets, 8)
                    torch.cuda.synchronize()
                    planning_seconds = time.monotonic() - t0
                    selected = choose(costs)
                    rgb10 = env.step(np.repeat(bank[selected][None], 5, axis=0))
                    latent5 = goal_latent(context).cpu().numpy()
                    latent10 = goal_latent(encode(model, rgb10, device)).cpu().numpy()
                key = f"r{root}_{method}"
                arrays[f"{key}_rgb5"], arrays[f"{key}_rgb10"] = rgb5, rgb10
                arrays[f"{key}_latent5"], arrays[f"{key}_latent10"] = latent5, latent10
                sequences.append(np.concatenate((np.repeat(prefix[None], 5, axis=0),
                                                 np.repeat(bank[selected][None], 5, axis=0))))
                branches.append({
                    "method": method, "prefix_action": prefix.tolist(), "tail_index": selected,
                    "tail_action": bank[selected].tolist(), "tail_costs": costs.tolist(),
                    "rgb5_sha256": image_hash(rgb5), "rgb10_sha256": image_hash(rgb10),
                    "tail_planning_seconds_including_feedback_encoding": planning_seconds,
                    "feedback_encodes": 1, "tail_prediction_samples": 8,
                    "endpoint_encodes_evaluation_only": 1, "physical_steps": 10,
                    "inherited_prefix_prediction_samples": 0 if method == "zero" else 5,
                    "inherited_prefix_jvp_calls": 2 if method == "spectral" else 0,
                    "inherited_prefix_svd_calls": 1 if method == "spectral" else 0,
                })
            # Privileged goal and state are used only after all branch decisions.
            goal_position = dataset.states[root, 10, :2].cpu().numpy().astype(np.float64)
            endpoints = physical.replay_position_endpoints(
                np.repeat(initial_state[None], len(METHODS), axis=0), np.stack(sequences))
            for branch, endpoint in zip(branches, endpoints):
                branch["endpoint_eval_only"] = endpoint.tolist()
                branch["distance_eval_only"] = float(np.linalg.norm(endpoint[:2] - goal_position))
            rows.append({"root": root, "goal_position_eval_only": goal_position.tolist(),
                         "goal_rgb_sha256": image_hash(images[10]), "branches": branches})
            print(f"completed root {root}: " + ", ".join(
                f"{b['method']}={b['distance_eval_only']:.5f}" for b in branches), flush=True)
        after = parameter_hash(model)
        if after != before:
            raise RuntimeError("frozen model changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {
        "schema": "a1_pointmaze_spectral_cp5_replan.v1",
        "scope": "exposed development roots; no independent P1 or equal-total-budget claim",
        "protocol_sha256": sha256_file(PROTOCOL), "source_sha256": SOURCE_SHA,
        "bank_source_sha256": BANK_SHA, "script_sha256": sha256_file(Path(__file__)),
        "bridge_sha256": sha256_file(ROOT / "experiments/pointmaze_native_rgb_bridge_20261102.py"),
        "checkpoint_sha256": sha256_file(args.checkpoint), "model_before": before, "model_after": after,
        "arrays_sha256": sha256_file(ARRAYS), "bank": bank.tolist(), "rows": rows,
        "action_mean": mean.tolist(), "action_std": std.tolist(),
        "normalizer_source": "full archive; checkpoint training provenance unknown",
        "physical_backend": physical.position_backend_receipt(), "device": torch.cuda.get_device_name(device),
        "wall_seconds": time.monotonic() - started,
    }
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
