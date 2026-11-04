"""Evaluate model-scored old-tail retention after actual RGB feedback."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_native_feedback_time_20261102 import encode
from experiments.a0_pointmaze_physical_cem_warm_tail_20261103 import warm_cem
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

SOURCE = ROOT / "results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json"
SOURCE_SHA = "ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_VISUAL_WARM_TAIL_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a1_pointmaze_visual_warm_tail_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    if sha256_file(SOURCE) != SOURCE_SHA:
        raise RuntimeError("source report changed")
    source = json.loads(SOURCE.read_text())
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (DEFAULT_CHECKPOINT, source["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", source["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", source["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source input changed: {path}")
    saved = np.load(SOURCE.with_suffix(".npz"), allow_pickle=False)
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, arrays, started = [], {}, time.monotonic()
    print("Real RGB cp5 feedback: cold versus old-tail model-scored CEM.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise RuntimeError("model identity changed")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid normalizer")
        for row in source["rows"]:
            root = row["root"]
            archive = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
            if sha256_file(archive) != row["archive_sha256"]:
                raise RuntimeError("goal archive changed")
            initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            goal_position = dataset.states[root, 10, :2].cpu().numpy().astype(np.float64)
            goal_rgb = saved[f"r{root}_goal_rgb"]
            goal = encode(model, goal_rgb, device)
            all_branches, seqs = [], []
            for parent in row["branches"]:
                method = parent["method"]
                plan = np.asarray(parent["initial_plan"], dtype=np.float32)
                prefix = np.repeat(plan[0][None], 5, axis=0)
                noise = saved[f"r{root}_tail_noise"]
                for policy in ("cold", "retained"):
                    with NativeRGBBridge(initial) as env:
                        if not np.array_equal(env.initial_rgb, saved[f"r{root}_rgb0"]):
                            raise RuntimeError("reset image mismatch")
                        rgb5 = env.step(prefix)
                        if image_hash(rgb5) != parent["rgb5_sha256"]:
                            raise RuntimeError("source prefix image mismatch")
                        torch.cuda.synchronize()
                        timer = time.monotonic()
                        mask, centers = task_mask(rgb5, goal_rgb)
                        feedback = encode(model, rgb5, device)
                        calls = {"model_prediction_batches": 0, "model_prediction_samples": 0}
                        def scorer(proposals):
                            calls["model_prediction_batches"] += 1
                            calls["model_prediction_samples"] += len(proposals)
                            return score_plans(model, feedback, goal, mask, proposals, mean, std)
                        if policy == "cold":
                            tail, cost, trace = cem(noise, scorer)
                            if not np.array_equal(tail, np.asarray(parent["tail_plan"], dtype=np.float32)):
                                raise RuntimeError("cold source selected tail mismatch")
                            if abs(cost-parent["tail_cost"]) > 1e-5:
                                raise RuntimeError("cold source score mismatch")
                        else:
                            tail, cost, trace = warm_cem(noise, plan[1:2], scorer)
                            if cost > trace[0]["scores"][0] + 1e-8:
                                raise RuntimeError("retained candidate lost")
                        torch.cuda.synchronize()
                        planning_seconds = time.monotonic()-timer
                        rgb10 = env.step(np.repeat(tail[0][None], 5, axis=0))
                    if policy == "cold" and image_hash(rgb10) != parent["rgb10_sha256"]:
                        raise RuntimeError("cold source outcome image mismatch")
                    key = f"r{root}_{method}_{policy}"
                    arrays[f"{key}_rgb10"] = rgb10
                    arrays[f"{key}_feedback_mask"] = mask
                    seqs.append(np.concatenate((prefix, np.repeat(tail[0][None], 5, axis=0))))
                    all_branches.append({"initial_method": method, "policy": policy,
                                         "initial_plan": plan.tolist(), "tail": tail.tolist(),
                                         "tail_cost": cost, "trace": trace, "feedback_centers_xy": centers,
                                         "rgb5_sha256": image_hash(rgb5), "rgb10_sha256": image_hash(rgb10),
                                         "feedback_encoder_calls": 1, "tail_prediction": calls,
                                         "initial_prediction_samples": parent["initial_prediction_samples"],
                                         "initial_predicted_block_transitions": 96,
                                         "initial_jvp_calls": parent["attributable_jvp_calls"],
                                         "initial_svd_calls": parent["attributable_svd_calls"],
                                         "tail_predicted_block_transitions": 48, "physical_steps": 10,
                                         "feedback_and_planning_seconds": planning_seconds})
                print(f"root {root}, {method}: cold and retained tails executed", flush=True)
            endpoints = physical.replay_position_endpoints(np.repeat(initial[None], len(seqs), axis=0), np.stack(seqs))
            for branch, endpoint in zip(all_branches, endpoints):
                branch["position_endpoint_eval_only"] = endpoint.tolist()
                branch["distance_eval_only"] = float(np.linalg.norm(endpoint[:2]-goal_position))
            rows.append({"root": root, "goal_position_eval_only": goal_position.tolist(), "branches": all_branches})
        after = parameter_hash(model)
        if after != before:
            raise RuntimeError("frozen parameters changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a1_pointmaze_visual_warm_tail.v1", "source_sha256": SOURCE_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "model_before": before, "model_after": after,
              "device": torch.cuda.get_device_name(device), "physical_backend": physical.position_backend_receipt(),
              "wall_seconds": time.monotonic()-started,
              "scope": "exposed roots, fixed raw-zero proprio, not independent P1", "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
