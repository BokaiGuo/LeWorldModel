"""Compare pooled and spatial terminal objectives on the sealed tail panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_native_feedback_time_20261102 import encode
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE = ROOT / "results/a1_pointmaze_spectral_cp5_replan_20261102_DEV01.json"
SOURCE_SHA = "8f0bba93fa1f093cae4550da317d9b20a6cc018756e37402419f384eabdf6446"
ORACLE = ROOT / "results/a1_pointmaze_cp5_eight_tail_oracle_20261102_DEV01.json"
ORACLE_SHA = "8e8b3a938da82829c722b646b09dfd0dfdb395974960a81d83445c7fb3fec129"
CONTEXT_ARRAYS = ROOT / "results/a1_pointmaze_spectral_cp5_replan_20261102_DEV01.npz"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CP5_SPATIAL_OBJECTIVE_PROTOCOL_20261102.md"
OUTPUT = ROOT / "results/a1_pointmaze_cp5_spatial_objective_20261102_DEV01.json"
ARRAYS = ROOT / "results/a1_pointmaze_cp5_spatial_objective_20261102_DEV01.npz"


def costs(prediction, goal):
    visual, proprio = prediction["visual"][-1], prediction["proprio"][-1]
    target_visual, target_proprio = goal["visual"][0, 0], goal["proprio"][0, 0]
    if visual.ndim != 5 or visual.shape[0] != 8 or visual.shape[1:] != target_visual.shape:
        raise ValueError("unexpected visual shapes")
    if proprio.shape[0] != 8 or proprio.shape[1:] != target_proprio.shape:
        raise ValueError("unexpected proprio shapes")
    pooled = ((visual.mean(dim=(1, 2, 3)) - target_visual.mean(dim=(0, 1, 2))) ** 2).mean(dim=1)
    spatial = ((visual - target_visual) ** 2).mean(dim=(1, 2, 3, 4))
    prop = ((proprio - target_proprio) ** 2).flatten(1).mean(dim=1)
    return pooled, spatial, prop


def main():
    if OUTPUT.exists() or ARRAYS.exists() or sha256_file(SOURCE) != SOURCE_SHA or sha256_file(ORACLE) != ORACLE_SHA:
        raise RuntimeError("output exists or sealed source changed")
    source, oracle = json.loads(SOURCE.read_text()), json.loads(ORACLE.read_text())
    if oracle["source_sha256"] != SOURCE_SHA or sha256_file(CONTEXT_ARRAYS) != source["arrays_sha256"]:
        raise RuntimeError("context ancestry mismatch")
    oracle_arrays = ROOT / "results/a1_pointmaze_cp5_eight_tail_oracle_20261102_DEV01.npz"
    if sha256_file(oracle_arrays) != oracle["arrays_sha256"] or sha256_file(DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]:
        raise RuntimeError("candidate arrays or checkpoint changed")
    bank = np.asarray(source["bank"], dtype=np.float32)
    if bank.shape != (8, 2) or not np.isfinite(bank).all() or np.any(np.abs(bank) > 1):
        raise ValueError("invalid bank")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, arrays, started = [], {}, time.monotonic()
    print("Comparing pooled and spatial objectives on fixed CP5 candidates.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"), \
            np.load(CONTEXT_ARRAYS, allow_pickle=False) as context_data, \
            np.load(oracle_arrays, allow_pickle=False) as outcomes:
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise RuntimeError("model changed")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("invalid normalizer")
        normalized = ((torch.as_tensor(bank, device=device) - mean) / std).repeat(1, 5).unsqueeze(0)
        for root_row, oracle_row in zip(source["rows"], oracle["rows"]):
            root = root_row["root"]
            if root != oracle_row["root"]:
                raise RuntimeError("root mismatch")
            goal_rgb = context_data[f"r{root}_goal_rgb"]
            if image_hash(goal_rgb) != root_row["goal_rgb_sha256"]:
                raise RuntimeError("goal RGB mismatch")
            goal = encode(model, goal_rgb, device)
            arrays[f"r{root}_goal_visual"] = goal["visual"][0, 0].cpu().numpy()
            arrays[f"r{root}_goal_proprio"] = goal["proprio"][0, 0].cpu().numpy()
            branches = []
            for branch, oracle_branch in zip(root_row["branches"], oracle_row["branches"]):
                method = branch["method"]
                if method != oracle_branch["method"]:
                    raise RuntimeError("method mismatch")
                key = f"r{root}_{method}"
                rgb5 = context_data[f"{key}_rgb5"]
                if image_hash(rgb5) != branch["rgb5_sha256"]:
                    raise RuntimeError("CP5 RGB mismatch")
                context = encode(model, rgb5, device)
                with torch.no_grad():
                    rollout = model.unroll(context, act_suffix=normalized)
                    pooled, spatial, prop = costs(rollout, goal)
                if not np.allclose(pooled.cpu().numpy(), branch["tail_costs"], rtol=1e-4, atol=1e-6):
                    raise RuntimeError("old pooled cost mismatch")
                actual_features = []
                images = outcomes[f"{key}_rgb10"]
                if [image_hash(image) for image in images] != oracle_branch["rgb10_sha256"]:
                    raise RuntimeError("candidate RGB mismatch")
                for image in images:
                    actual_features.append(encode(model, image, device)["visual"][0, 0].cpu().numpy())
                actual = np.stack(actual_features)
                target = goal["visual"][0, 0].cpu().numpy()
                actual_cost = ((actual - target) ** 2).mean(axis=(1, 2, 3, 4))
                single = pooled.cpu().numpy()
                spatial_np = spatial.cpu().numpy()
                prop_np = prop.cpu().numpy()
                distances = np.asarray(oracle_branch["physical_distances"])
                indices = {"pooled": int(np.argmin(single)), "spatial": int(np.argmin(spatial_np)),
                           "spatial_proprio": int(np.argmin(spatial_np + 0.1 * prop_np)),
                           "actual_spatial_oracle": int(np.argmin(actual_cost)),
                           "physical_oracle": int(np.argmin(distances))}
                arrays[f"{key}_pred_visual"] = rollout["visual"][-1].cpu().numpy()
                arrays[f"{key}_pred_proprio"] = rollout["proprio"][-1].cpu().numpy()
                arrays[f"{key}_actual_visual"] = actual
                branches.append({"method": method, "indices": indices, "pooled_costs": single.tolist(),
                                 "spatial_costs": spatial_np.tolist(), "proprio_costs": prop_np.tolist(),
                                 "actual_spatial_costs": actual_cost.tolist(),
                                 "physical_distances": distances.tolist(),
                                 "distance_regrets": {name: float(distances[index] - distances.min())
                                                      for name, index in indices.items()}})
            rows.append({"root": root, "branches": branches})
            print(f"spatial objective root {root} complete", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    result = {"schema": "a1_pointmaze_cp5_spatial_objective.v1", "source_sha256": SOURCE_SHA,
              "oracle_sha256": ORACLE_SHA, "arrays_sha256": sha256_file(ARRAYS),
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "model_before": before, "model_after": after, "device": torch.cuda.get_device_name(device),
              "scope": "exposed-root fixed-bank development; not official AP/LeWM parity or P1",
              "wall_seconds": time.monotonic() - started, "rows": rows}
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
