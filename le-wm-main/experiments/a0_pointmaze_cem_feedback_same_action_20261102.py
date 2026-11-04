"""Replay fixed CEM predictions and compare real-cp5 refreshed prediction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_cem_all_candidates_20261102 import pairwise
from experiments.a0_pointmaze_native_feedback_time_20261102 import encode
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

PREV = ROOT / "results/a0_pointmaze_cem_all_candidates_20261102_DEV01.json"
PREV_SHA = "7dbf437b3f3473c467f6245ac788bd7de07d330f1ac17479b12e9f4b0b1f60fb"
SOURCE = ROOT / "results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json"
SOURCE_SHA = "ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CEM_FEEDBACK_SAME_ACTION_PROTOCOL_20261102.md"
OUTPUT = ROOT / "results/a0_pointmaze_cem_feedback_same_action_20261102_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def task_error(features, target, mask):
    error = ((features - target) ** 2).mean(dim=(1, 4))
    return error[:, mask].mean(dim=1)


def suffix(actions, mean, std):
    normalized = (actions - mean) / std
    return normalized.repeat(1, 1, 5).permute(1, 0, 2)


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing to overwrite sealed output")
    if sha256_file(PREV) != PREV_SHA or sha256_file(SOURCE) != SOURCE_SHA:
        raise RuntimeError("source report changed")
    prev, source = json.loads(PREV.read_text()), json.loads(SOURCE.read_text())
    for path, expected in ((PREV.with_suffix(".npz"), prev["arrays_sha256"]),
                           (SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (DEFAULT_CHECKPOINT, source["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", source["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", source["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed input: {path}")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    source_arrays = np.load(SOURCE.with_suffix(".npz"), allow_pickle=False)
    prev_arrays = np.load(PREV.with_suffix(".npz"), allow_pickle=False)
    rows, arrays, started = [], {}, time.monotonic()
    print("Recompute sealed CEM scores and compare same-action cp5 feedback.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"] or before != prev["model_before"]:
            raise RuntimeError("model identity differs")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid action normalizer")
        for old_row, prior_row in zip(source["rows"], prev["rows"]):
            root = old_row["root"]
            if prior_row["root"] != root:
                raise RuntimeError("root mismatch")
            archive = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
            if sha256_file(archive) != old_row["archive_sha256"]:
                raise RuntimeError("goal archive changed")
            initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            rgb0 = source_arrays[f"r{root}_rgb0"]
            mask = torch.as_tensor(source_arrays[f"r{root}_initial_mask"], dtype=torch.bool, device=device)
            context = encode(model, rgb0, device)
            goal = encode(model, source_arrays[f"r{root}_goal_rgb"], device)["visual"][0, 0]
            branches = []
            for original, prior in zip(old_row["branches"], prior_row["branches"]):
                method = original["method"]
                if method != prior["method"]:
                    raise RuntimeError("method mismatch")
                key = f"r{root}_{method}"
                actions = prev_arrays[f"{key}_proposals"]
                real10 = prev_arrays[f"{key}_rgb10"]
                if actions.shape != (48, 2, 2) or real10.shape != (48, 224, 224, 3):
                    raise ValueError("bad candidate arrays")
                batch = torch.as_tensor(actions, device=device)
                with torch.no_grad():
                    prediction = model.unroll(context, act_suffix=suffix(batch, mean, std))["visual"][-1]
                    recomputed = task_error(prediction, goal, mask).cpu().numpy()
                saved_scores = np.asarray(prior["prediction"])
                error = float(np.max(np.abs(recomputed - saved_scores)))
                if error > 1e-5 or int(np.argmin(recomputed)) != int(np.argmin(saved_scores)):
                    raise RuntimeError(f"source prediction score mismatch: {error}")
                refreshed, actual, mse_open, mse_refresh, images5 = [], [], [], [], []
                for i, action in enumerate(actions):
                    if image_hash(real10[i]) != prior["rgb10_sha256"][i]:
                        raise RuntimeError("real cp10 RGB hash mismatch")
                    with NativeRGBBridge(initial) as env:
                        if not np.array_equal(env.initial_rgb, rgb0):
                            raise RuntimeError("reset image mismatch")
                        rgb5 = env.step(np.repeat(action[0][None], 5, axis=0))
                    if np.array_equal(action[0], np.asarray(original["initial_plan"], np.float32)[0]):
                        if image_hash(rgb5) != original["rgb5_sha256"]:
                            raise RuntimeError("selected prefix image mismatch")
                    with torch.no_grad():
                        c5 = encode(model, rgb5, device)
                        real = encode(model, real10[i], device)["visual"][0, 0]
                        after = model.unroll(c5, act_suffix=suffix(batch[i:i+1, 1:2], mean, std))["visual"][-1]
                        actual.append(float(task_error(real[None], goal, mask)[0].cpu()))
                        refreshed.append(float(task_error(after, goal, mask)[0].cpu()))
                        mse_open.append(float(task_error(prediction[i:i+1], real, mask)[0].cpu()))
                        mse_refresh.append(float(task_error(after, real, mask)[0].cpu()))
                    images5.append(rgb5)
                if not np.allclose(actual, prior["actual_rgb_cost"], rtol=1e-5, atol=1e-6):
                    raise RuntimeError("actual RGB score mismatch")
                arrays[f"{key}_rgb5"] = np.stack(images5)
                branches.append({"method": method, "source_score_max_abs_error": error,
                                 "open_vs_actual_rgb": pairwise(recomputed, actual),
                                 "refresh_vs_actual_rgb": pairwise(refreshed, actual),
                                 "open_scores": recomputed.tolist(), "refresh_scores": refreshed,
                                 "actual_scores": actual, "open_prediction_mse": mse_open,
                                 "refresh_prediction_mse": mse_refresh,
                                 "open_mean_mse": float(np.mean(mse_open)),
                                 "refresh_mean_mse": float(np.mean(mse_refresh)),
                                 "refresh_lower_mse_count": int(np.count_nonzero(np.asarray(mse_refresh) < mse_open)),
                                 "rgb5_sha256": [image_hash(x) for x in images5]})
                print(f"root {root}, {method}: 48 same-action checks completed", flush=True)
            rows.append({"root": root, "branches": branches})
        after_hash = parameter_hash(model)
        if after_hash != before:
            raise RuntimeError("model changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a0_pointmaze_cem_feedback_same_action.v1", "source_sha256": SOURCE_SHA,
              "prior_sha256": PREV_SHA, "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "model_before": before, "model_after": after_hash, "device": torch.cuda.get_device_name(device),
              "wall_seconds": time.monotonic() - started,
              "scope": "known roots; fixed actions; prediction diagnostic only", "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
