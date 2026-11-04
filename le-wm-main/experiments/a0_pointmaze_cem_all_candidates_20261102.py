"""Replay every sealed initial CEM proposal with time-aligned RGB and physics."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_native_feedback_time_20261102 import encode
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE = ROOT / "results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json"
SOURCE_SHA = "ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade"
SOURCE_ARRAYS = SOURCE.with_suffix(".npz")
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CEM_ALL_CANDIDATES_PROTOCOL_20261102.md"
OUTPUT = ROOT / "results/a0_pointmaze_cem_all_candidates_20261102_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def pairwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != (48,) or b.shape != (48,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("expected finite vectors of 48 scores")
    products = [np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
                for i, j in itertools.combinations(range(48), 2)]
    return {"reversed": int(np.count_nonzero(np.asarray(products) < 0)),
            "ties": int(np.count_nonzero(np.asarray(products) == 0)), "pairs": len(products)}


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing to overwrite results")
    if sha256_file(SOURCE) != SOURCE_SHA:
        raise RuntimeError("source JSON changed")
    source = json.loads(SOURCE.read_text())
    if sha256_file(SOURCE_ARRAYS) != source["arrays_sha256"]:
        raise RuntimeError("source NPZ changed")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    for path, key in ((args.checkpoint, "checkpoint_sha256"), (args.dataset_root / "states.pth", "states_sha256"),
                      (args.dataset_root / "actions.pth", "actions_sha256")):
        if sha256_file(path) != source[key]:
            raise RuntimeError(f"source input changed: {path}")
    saved = np.load(SOURCE_ARRAYS, allow_pickle=False)
    arrays, rows, started = {}, [], time.monotonic()
    print("Replaying all sealed initial CEM proposals; real outcomes remain evaluation-only.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"] or before != source["model_after"]:
            raise RuntimeError("model identity differs")
        for row in source["rows"]:
            root = row["root"]
            if sha256_file(args.dataset_root / "obses" / f"episode_{root:03d}.pth") != row["archive_sha256"]:
                raise RuntimeError("goal archive changed")
            initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            goal_pos = dataset.states[root, 10, :2].cpu().numpy().astype(np.float64)
            goal_rgb, rgb0, mask = (saved[f"r{root}_{suffix}"] for suffix in
                                    ("goal_rgb", "rgb0", "initial_mask"))
            with torch.no_grad():
                goal_visual = encode(model, goal_rgb, device)["visual"][0, 0]
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
            branches = []
            for branch in row["branches"]:
                method = branch["method"]
                trace = branch["initial_trace"]
                proposals = np.asarray([t["proposals"] for t in trace], np.float32).reshape(48, 2, 2)
                prediction = np.asarray([t["scores"] for t in trace], np.float64).reshape(48)
                if not np.isfinite(proposals).all() or np.any(np.abs(proposals) > 1) or not np.isfinite(prediction).all():
                    raise ValueError("invalid candidate")
                chosen = int(np.argmin(prediction))
                if not np.array_equal(proposals[chosen], np.asarray(branch["initial_plan"], np.float32)):
                    raise RuntimeError("source best plan mismatch")
                images, costs = [], []
                for proposal in proposals:
                    with NativeRGBBridge(initial) as env:
                        if not np.array_equal(env.initial_rgb, rgb0):
                            raise RuntimeError("initial RGB mismatch")
                        rgb5 = env.step(np.repeat(proposal[0][None], 5, axis=0))
                        if np.array_equal(proposal[0], proposals[chosen, 0]) and image_hash(rgb5) != branch["rgb5_sha256"]:
                            raise RuntimeError("selected prefix RGB mismatch")
                        rgb10 = env.step(np.repeat(proposal[1][None], 5, axis=0))
                    with torch.no_grad():
                        visual = encode(model, rgb10, device)["visual"][0, 0]
                        err = ((visual - goal_visual) ** 2).mean(dim=(0, 3))
                        costs.append(float(err[mask_t].mean().cpu()))
                    images.append(rgb10)
                sequences = np.concatenate((np.repeat(proposals[:, 0:1], 5, axis=1),
                                            np.repeat(proposals[:, 1:2], 5, axis=1)), axis=1)
                endpoints = physical.replay_position_endpoints(np.repeat(initial[None], 48, axis=0), sequences)
                distances = np.linalg.norm(endpoints[:, :2] - goal_pos, axis=1)
                costs = np.asarray(costs)
                choices = {"model": chosen, "real_rgb_oracle": int(np.argmin(costs)),
                           "physical_oracle": int(np.argmin(distances))}
                key = f"r{root}_{method}"
                arrays[f"{key}_rgb10"] = np.stack(images)
                arrays[f"{key}_endpoints"] = endpoints
                arrays[f"{key}_proposals"] = proposals
                branches.append({"method": method, "prediction": prediction.tolist(),
                                 "actual_rgb_cost": costs.tolist(), "distance_eval_only": distances.tolist(),
                                 "choices": choices,
                                 "distance_regret": {k: float(distances[v] - distances[choices["physical_oracle"]])
                                                     for k, v in choices.items()},
                                 "prediction_vs_actual_rgb": pairwise(prediction, costs),
                                 "actual_rgb_vs_physical": pairwise(costs, distances),
                                 "prediction_vs_physical": pairwise(prediction, distances),
                                 "rgb10_sha256": [image_hash(x) for x in images]})
                print(f"root {root}, {method}: 48 candidates executed", flush=True)
            rows.append({"root": root, "goal_position_eval_only": goal_pos.tolist(), "branches": branches})
        after = parameter_hash(model)
        if after != before:
            raise RuntimeError("model parameters changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a0_pointmaze_cem_all_candidates.v1", "source_sha256": SOURCE_SHA,
              "source_arrays_sha256": source["arrays_sha256"], "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "model_before": before, "model_after": after, "device": torch.cuda.get_device_name(device),
              "physical_backend": physical.position_backend_receipt(), "wall_seconds": time.monotonic() - started,
              "scope": "exposed roots, open-loop plans, evaluation-only oracles", "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
