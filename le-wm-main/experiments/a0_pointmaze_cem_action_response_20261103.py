"""Decompose same-action model and real RGB candidate response geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_cem_feedback_same_action_20261102 import suffix, task_error
from experiments.a0_pointmaze_native_feedback_time_20261102 import encode
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

PREV = ROOT / "results/a0_pointmaze_cem_all_candidates_20261102_DEV01.json"
PREV_SHA = "7dbf437b3f3473c467f6245ac788bd7de07d330f1ac17479b12e9f4b0b1f60fb"
SOURCE = ROOT / "results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json"
SOURCE_SHA = "ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CEM_ACTION_RESPONSE_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a0_pointmaze_cem_action_response_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def decompose(predicted, actual, direction):
    p = np.asarray(predicted, np.float64)
    y = np.asarray(actual, np.float64)
    d = np.asarray(direction, np.float64).reshape(-1)
    if p.ndim != 2 or p.shape != y.shape or p.shape[0] != 48 or p.shape[1] != d.size:
        raise ValueError("bad response shapes")
    if not np.isfinite(p).all() or not np.isfinite(y).all() or not np.isfinite(d).all():
        raise ValueError("nonfinite inputs")
    pc, yc = p - p.mean(axis=0), y - y.mean(axis=0)
    pp, yy, py, dd = np.sum(pc*pc), np.sum(yc*yc), np.sum(pc*yc), np.dot(d, d)
    if min(pp, yy, dd) <= 1e-20:
        raise ValueError("degenerate response or goal direction")
    predicted_projection, actual_projection = pc @ d / np.sqrt(dd), yc @ d / np.sqrt(dd)
    return {"centered_cosine": float(py / np.sqrt(pp*yy)),
            "real_on_predicted_gain": float(py / pp),
            "predicted_to_real_norm": float(np.sqrt(pp/yy)),
            "centered_relative_squared_error": float(np.sum((pc-yc)**2)/yy),
            "centered_predicted_energy": float(pp), "centered_actual_energy": float(yy),
            "centered_cross_inner_product": float(py),
            "predicted_goal_projection": predicted_projection.tolist(),
            "actual_goal_projection": actual_projection.tolist()}


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((PREV, PREV_SHA), (SOURCE, SOURCE_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError("source changed")
    prev, source = json.loads(PREV.read_text()), json.loads(SOURCE.read_text())
    for path, expected in ((PREV.with_suffix(".npz"), prev["arrays_sha256"]),
                           (SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (DEFAULT_CHECKPOINT, source["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", source["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", source["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed input {path}")
    src = np.load(SOURCE.with_suffix(".npz"), allow_pickle=False)
    saved = np.load(PREV.with_suffix(".npz"), allow_pickle=False)
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, arrays, started = [], {}, time.monotonic()
    print("Decomposing frozen model versus real RGB action response, evaluation only.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"] or before != prev["model_before"]:
            raise RuntimeError("model changed")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid normalizer")
        for row, prior in zip(source["rows"], prev["rows"]):
            root = row["root"]
            if root != prior["root"]:
                raise RuntimeError("root mismatch")
            path = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
            if sha256_file(path) != row["archive_sha256"]:
                raise RuntimeError("goal archive changed")
            mask = torch.as_tensor(src[f"r{root}_initial_mask"], dtype=torch.bool, device=device)
            context = encode(model, src[f"r{root}_rgb0"], device)
            goal = encode(model, src[f"r{root}_goal_rgb"], device)["visual"][0, 0]
            direction = (goal-context["visual"][0, 0])[0, mask, :].reshape(-1).cpu().numpy()
            branches = []
            for parent, old in zip(row["branches"], prior["branches"]):
                method = parent["method"]
                if method != old["method"]:
                    raise RuntimeError("method mismatch")
                key = f"r{root}_{method}"
                proposals = saved[f"{key}_proposals"]
                images = saved[f"{key}_rgb10"]
                if proposals.shape != (48, 2, 2) or images.shape != (48, 224, 224, 3):
                    raise ValueError("invalid shape")
                with torch.no_grad():
                    predicted = model.unroll(context, act_suffix=suffix(torch.as_tensor(proposals, device=device),
                                                                         mean, std))["visual"][-1]
                    pcost = task_error(predicted, goal, mask).cpu().numpy()
                    actual_features, actual_costs = [], []
                    for image, expected_hash in zip(images, old["rgb10_sha256"]):
                        if image_hash(image) != expected_hash:
                            raise RuntimeError("RGB hash mismatch")
                        actual = encode(model, image, device)["visual"][0, 0]
                        actual_costs.append(float(task_error(actual[None], goal, mask)[0].cpu()))
                        actual_features.append(actual[0, mask, :].reshape(-1).cpu().numpy())
                    predicted_features = predicted[:, 0, mask, :].reshape(48, -1).cpu().numpy()
                if not np.allclose(pcost, old["prediction"], rtol=1e-5, atol=1e-6):
                    raise RuntimeError("original prediction score mismatch")
                if not np.allclose(actual_costs, old["actual_rgb_cost"], rtol=1e-5, atol=1e-6):
                    raise RuntimeError("original actual score mismatch")
                actual_features = np.stack(actual_features)
                arrays[f"{key}_predicted_features"] = predicted_features
                arrays[f"{key}_actual_features"] = actual_features
                arrays[f"{key}_goal_direction"] = direction
                branches.append({"method": method, **decompose(predicted_features, actual_features, direction)})
                print(f"root {root}, {method}: 48 action responses encoded", flush=True)
            rows.append({"root": root, "branches": branches})
        after = parameter_hash(model)
        if after != before:
            raise RuntimeError("model parameters changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a0_pointmaze_cem_action_response.v1", "previous_sha256": PREV_SHA,
              "source_sha256": SOURCE_SHA, "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "model_before": before, "model_after": after, "device": torch.cuda.get_device_name(device),
              "wall_seconds": time.monotonic()-started,
              "scope": "known-root post-outcome diagnostic; not a deployable calibrated planner", "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
