"""Privileged initial-proprio counterfactual on fixed action-response banks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from tensordict import TensorDict

from experiments.a0_pointmaze_cem_action_response_20261103 import decompose
from experiments.a0_pointmaze_cem_all_candidates_20261102 import pairwise
from experiments.a0_pointmaze_cem_feedback_same_action_20261102 import suffix, task_error
from experiments.a0_pointmaze_native_feedback_time_20261102 import encode
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

RESPONSE = ROOT / "results/a0_pointmaze_cem_action_response_20261103_DEV01.json"
RESPONSE_SHA = "0f7492d9be9c39d07ee4805c359c252f50a354d20214ce1f4634fce39023b471"
REPLAY = ROOT / "results/a0_pointmaze_cem_all_candidates_20261102_DEV01.json"
REPLAY_SHA = "7dbf437b3f3473c467f6245ac788bd7de07d330f1ac17479b12e9f4b0b1f60fb"
SOURCE = ROOT / "results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json"
SOURCE_SHA = "ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_PROPRIO_ACTION_RESPONSE_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a0_pointmaze_proprio_action_response_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def encode_with_state(model, image, state, device):
    image, state = np.asarray(image), np.asarray(state, dtype=np.float32)
    if image.dtype != np.uint8 or image.shape != (224, 224, 3) or state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("invalid RGB or raw proprio")
    obs = {"visual": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()[None, None].to(device),
           "proprio": torch.as_tensor(state, device=device)[None, None]}
    with torch.no_grad():
        encoded = model.encode(obs)
    return TensorDict({"visual": encoded["visual"], "proprio": encoded["proprio"]},
                      batch_size=[1, 1], device=device)


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((RESPONSE, RESPONSE_SHA), (REPLAY, REPLAY_SHA), (SOURCE, SOURCE_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    response, replay, source = (json.loads(p.read_text()) for p in (RESPONSE, REPLAY, SOURCE))
    for path, expected in ((RESPONSE.with_suffix(".npz"), response["arrays_sha256"]),
                           (REPLAY.with_suffix(".npz"), replay["arrays_sha256"]),
                           (SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (DEFAULT_CHECKPOINT, source["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", source["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", source["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source arrays or input changed: {path}")
    src, rep, geom = (np.load(p.with_suffix(".npz"), allow_pickle=False) for p in (SOURCE, REPLAY, RESPONSE))
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    arrays, rows, started = {}, [], time.monotonic()
    print("Testing privileged initial proprio on fixed action bank; diagnostic only.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"] or before != response["model_before"]:
            raise RuntimeError("model identity changed")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid normalizer")
        for old_row, rep_row, geometry_row in zip(source["rows"], replay["rows"], response["rows"]):
            root = old_row["root"]
            if root != rep_row["root"] or root != geometry_row["root"]:
                raise RuntimeError("root mismatch")
            path = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
            if sha256_file(path) != old_row["archive_sha256"]:
                raise RuntimeError("archive changed")
            state = dataset.states[root, 0].cpu().numpy()
            rgb = src[f"r{root}_rgb0"]
            zero_context = encode(model, rgb, device)
            state_context = encode_with_state(model, rgb, state, device)
            if not torch.equal(zero_context["visual"], state_context["visual"]):
                raise RuntimeError("proprio unexpectedly changed visual encoding")
            goal = encode(model, src[f"r{root}_goal_rgb"], device)["visual"][0, 0]
            mask = torch.as_tensor(src[f"r{root}_initial_mask"], dtype=torch.bool, device=device)
            branches = []
            for previous, observed, geometry in zip(old_row["branches"], rep_row["branches"], geometry_row["branches"]):
                method = previous["method"]
                if method != observed["method"] or method != geometry["method"]:
                    raise RuntimeError("arm mismatch")
                key = f"r{root}_{method}"
                proposals = rep[f"{key}_proposals"]
                actual = geom[f"{key}_actual_features"]
                direction = geom[f"{key}_goal_direction"]
                with torch.no_grad():
                    output = model.unroll(state_context,
                                          act_suffix=suffix(torch.as_tensor(proposals, device=device), mean, std))["visual"][-1]
                    score = task_error(output, goal, mask).cpu().numpy()
                    predicted = output[:, 0, mask, :].reshape(48, -1).cpu().numpy()
                comparison = decompose(predicted, actual, direction)
                original_choice, state_choice = int(np.argmin(observed["prediction"])), int(np.argmin(score))
                distances = observed["distance_eval_only"]
                actual_costs = observed["actual_rgb_cost"]
                arrays[f"{key}_predicted_features"] = predicted
                branches.append({"method": method, "original_choice": original_choice, "state_choice": state_choice,
                                 "original_distance": distances[original_choice], "state_distance": distances[state_choice],
                                 "state_scores": score.tolist(),
                                 "zero_proprio_prediction_mse": float(np.mean((geom[f"{key}_predicted_features"] - actual)**2)),
                                 "state_proprio_prediction_mse": float(np.mean((predicted - actual)**2)),
                                 "original_prediction_vs_rgb": pairwise(observed["prediction"], actual_costs),
                                 "state_prediction_vs_rgb": pairwise(score, actual_costs),
                                 "zero_proprio_centered_cosine": geometry["centered_cosine"],
                                 "zero_proprio_norm_ratio": geometry["predicted_to_real_norm"],
                                 "state_response": comparison})
                print(f"root {root}, {method}: privileged initial proprio scored", flush=True)
            rows.append({"root": root, "initial_state_privileged": state.tolist(), "branches": branches})
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a0_pointmaze_proprio_action_response.v1", "source_sha256": SOURCE_SHA,
              "replay_sha256": REPLAY_SHA, "response_sha256": RESPONSE_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "model_before": before, "model_after": after,
              "device": torch.cuda.get_device_name(device), "wall_seconds": time.monotonic()-started,
              "scope": "privileged initial state diagnostic on exposed roots; not RGB planning or P1", "rows": rows}
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
