"""Physical same-prefix comparison of single-frame and history-tail CEM."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import estimate_state
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import CAMERA, CAMERA_SHA
from experiments.a1_pointmaze_rgb_position_history_diagnostic_20261103 import masked_cost
from experiments.a1_pointmaze_rgb_state_full_planning_20261103 import (
    OUTPUT as SOURCE, SOURCE as NOISE_SOURCE, SOURCE_SHA as NOISE_SHA, image_hash,
)
from experiments.a1_pointmaze_rgb_task_spectral_dev_20261102 import task_mask
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import cem, score_plans
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE_SHA = "4288b73e675e45526db46698a7abdcd68b619fc235e0a73bc20f91bead6d9b99"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_RGB_HISTORY_CEM_FEEDBACK_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a1_pointmaze_rgb_history_cem_feedback_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def history_score(model, context, prefix, proposals, mean, std, goal, mask):
    plans = torch.as_tensor(proposals, device=mean.device, dtype=torch.float32)
    if plans.ndim != 3 or plans.shape[1:] != (1, 2):
        raise ValueError("expected one-packet proposals")
    n = len(plans)
    normalized = ((plans[:, 0] - mean) / std).repeat(1, 5)
    past = ((torch.as_tensor(prefix, device=mean.device) - mean) / std).repeat(5)
    with torch.no_grad():
        out = unroll_with_action_history(model, context, past[None, None].expand(1, n, 10),
                                         normalized[None], window=2)["visual"][-1]
        return masked_cost(out, goal, torch.as_tensor(mask, device=mean.device)).cpu().numpy()


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (NOISE_SOURCE, NOISE_SHA), (CAMERA, CAMERA_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"sealed input changed: {path}")
    source, noise_doc, camera = (json.loads(p.read_text()) for p in (SOURCE, NOISE_SOURCE, CAMERA))
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (NOISE_SOURCE.with_suffix(".npz"), noise_doc["arrays_sha256"]),
                           (DEFAULT_CHECKPOINT, noise_doc["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", noise_doc["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", noise_doc["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    offset = source["body_offset_from_static_scene"]
    rows, arrays, start = [], {}, time.monotonic()
    print("Actual history versus single-frame tail CEM with physical execution; exposed roots.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as old, \
            np.load(NOISE_SOURCE.with_suffix(".npz"), allow_pickle=False) as noise, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise RuntimeError("checkpoint identity mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("invalid action normalizer")
        for row in source["rows"]:
            root = row["root"]
            initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            goal_rgb = noise[f"r{root}_goal_rgb"]
            goal_context = encode_with_state(model, goal_rgb, np.zeros(4), device)
            goal = goal_context["visual"][0, 0]
            sequences, branches = [], []
            for method in ("random_cem", "spectral_cem"):
                parent = next(b for b in row["branches"] if b["policy"] == "rgb_position" and b["method"] == method)
                prefix = np.asarray(parent["initial_plan"], np.float32)[0]
                for mode in ("single", "history"):
                    with NativeRGBBridge(initial) as env:
                        rgb0 = env.initial_rgb
                        if not np.array_equal(rgb0, noise[f"r{root}_rgb0"]):
                            raise RuntimeError("reset RGB mismatch")
                        torch.cuda.synchronize()
                        t0 = time.monotonic()
                        state0 = estimate_state(rgb0, camera["metadata"], offset)[0]
                        first = encode_with_state(model, rgb0, state0, device) if mode == "history" else None
                        torch.cuda.synchronize()
                        initial_encode_seconds = time.monotonic() - t0 if mode == "history" else 0.
                        rgb5 = env.step(np.repeat(prefix[None], 5, axis=0))
                        if image_hash(rgb5) != parent["rgb5_sha256"]:
                            raise RuntimeError("fixed prefix RGB mismatch")
                        torch.cuda.synchronize()
                        t0 = time.monotonic()
                        state5 = estimate_state(rgb5, camera["metadata"], offset)[0]
                        second = encode_with_state(model, rgb5, state5, device)
                        mask, _ = task_mask(rgb5, goal_rgb)
                        context = ({k: torch.cat((first[k], second[k]), dim=1) for k in ("visual", "proprio")}
                                   if first is not None else None)
                        torch.cuda.synchronize()
                        feedback_seconds = time.monotonic() - t0
                        counts = {"batches": 0, "samples": 0}
                        def scorer(proposals):
                            counts["batches"] += 1
                            counts["samples"] += len(proposals)
                            if mode == "single":
                                return score_plans(model, second, goal_context, mask, proposals, mean, std)
                            return history_score(model, context, prefix, proposals, mean, std, goal, mask)
                        torch.cuda.synchronize()
                        t0 = time.monotonic()
                        tail, cost, trace = cem(noise[f"r{root}_tail_noise"], scorer)
                        torch.cuda.synchronize()
                        plan_seconds = time.monotonic() - t0
                        rgb10 = env.step(np.repeat(tail[0][None], 5, axis=0))
                    if mode == "single":
                        if (not np.array_equal(tail, np.asarray(parent["tail"], np.float32))
                                or image_hash(rgb10) != parent["rgb10_sha256"]
                                or abs(cost - parent["tail_cost"]) > 1e-5):
                            raise RuntimeError("single-frame reference mismatch")
                    key = f"r{root}_{method}_{mode}"
                    arrays[f"{key}_rgb0"], arrays[f"{key}_rgb5"], arrays[f"{key}_rgb10"] = rgb0, rgb5, rgb10
                    arrays[f"{key}_mask"] = mask
                    sequences.append(np.concatenate((np.repeat(prefix[None], 5, axis=0),
                                                     np.repeat(tail[0][None], 5, axis=0))))
                    branches.append({"prefix_method": method, "mode": mode, "prefix": prefix.tolist(),
                                     "input0": state0.tolist(), "input5": state5.tolist(), "tail": tail.tolist(),
                                     "tail_cost": cost, "trace": trace, "prediction_counts": counts,
                                     "initial_extra_encode_calls": int(mode == "history"), "feedback_encode_calls": 1,
                                     "initial_extra_encode_seconds": initial_encode_seconds,
                                     "feedback_encode_seconds": feedback_seconds, "cem_seconds": plan_seconds,
                                     "physical_steps": 10, "rgb0_sha256": image_hash(rgb0),
                                     "rgb5_sha256": image_hash(rgb5), "rgb10_sha256": image_hash(rgb10)})
                    print(f"root {root}: {method}/{mode} physical execution complete", flush=True)
            target = dataset.states[root, 10, :2].cpu().numpy().astype(np.float64)
            endpoints = physical.replay_position_endpoints(np.repeat(initial[None], len(sequences), axis=0),
                                                            np.stack(sequences))
            for branch, endpoint in zip(branches, endpoints, strict=True):
                branch["endpoint_eval_only"] = endpoint.tolist()
                branch["distance_eval_only"] = float(np.linalg.norm(endpoint[:2] - target))
            rows.append({"root": root, "goal_position_eval_only": target.tolist(), "branches": branches})
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a1_pointmaze_rgb_history_cem_feedback.v1", "source_sha256": SOURCE_SHA,
              "noise_source_sha256": NOISE_SHA, "camera_sha256": CAMERA_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "model_before": before, "model_after": after,
              "device": torch.cuda.get_device_name(device), "physical_backend": physical.position_backend_receipt(),
              "wall_seconds": time.monotonic() - start,
              "scope": "same fixed developmental prefixes; not independent P1", "rows": rows}
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
