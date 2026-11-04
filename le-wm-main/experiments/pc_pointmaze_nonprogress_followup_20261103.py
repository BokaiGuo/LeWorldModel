"""RGB feedback and common-suffix tests for selected obstacle prefixes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA, estimate_state
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import OLD, OLD_SHA
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.pc_pointmaze_obstacle_nonprogress_bank_20261103 import OUTPUT as BANK
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import DATA, GOAL, ROOTS, OUTPUT as SOURCE
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.pc_ap_pointmaze_callback_20261103 import PointMazeAPCallback
from experiments.spectral_reachability.pc_ap_spectral_cem_20261103 import spectral_cem
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

BANK_SHA = "e45ac931ba3da9c480cc37c125c00d202ee7dd4014abba71ffd88c238b928e1b"
SOURCE_SHA = "e5955123b5b88797a66e4c703a1c0f5b321a23c8fbead371dc029c8d44a5ff91"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_NONPROGRESS_FOLLOWUP_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_nonprogress_followup_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")
ARMS = ("selected", "greedy", "zero")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((BANK, BANK_SHA), (SOURCE, SOURCE_SHA), (CAMERA, CAMERA_SHA), (OLD, OLD_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed input: {path}")
    bank_report, source, camera, old = (json.loads(x.read_text()) for x in (BANK, SOURCE, CAMERA, OLD))
    for path, expected in ((BANK.with_suffix(".npz"), bank_report["arrays_sha256"]),
                           (SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (DATA / "states.pth", source["states_sha256"]),
                           (DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed data: {path}")
    if [x["root"] for x in bank_report["rows"]] != list(ROOTS):
        raise ValueError("bank root identity mismatch")
    states = torch.load(DATA / "states.pth", weights_only=True, map_location="cpu")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, arrays, starts, sequences, terminals = [], {}, [], [], []
    start_time = time.monotonic()
    print("Replan all sealed non-progress prefixes and compare common physical suffixes.", flush=True)
    with np.load(BANK.with_suffix(".npz"), allow_pickle=False) as bank, \
            np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as source_arrays, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != bank_report["model_before"]:
            raise RuntimeError("model identity mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or not (std > 0).all():
            raise RuntimeError("invalid normalizer")
        lower = ((torch.full_like(mean, -1.)-mean)/std).repeat(5)
        upper = ((torch.ones_like(mean)-mean)/std).repeat(5)
        prior = ((torch.zeros_like(mean)-mean)/std).repeat(5).reshape(1, 1, 10)
        goal_rgb = source_arrays["goal_rgb"]
        goal_state, _ = estimate_state(goal_rgb, camera["metadata"], old["body_offset_from_static_scene"])
        goal = encode_with_state(model, goal_rgb, goal_state, device)
        for root, old_row in zip(ROOTS, bank_report["rows"]):
            initial = states[root, 40].numpy().astype(np.float64)
            indices = (old_row["selected"], old_row["greedy"], 0)
            if indices[0] is None:
                raise RuntimeError("no eligible selected prefix")
            branches = []
            for arm, idx in zip(ARMS, indices):
                prefix = np.repeat(bank[f"r{root}_bank"][idx][None], 5, axis=0)
                actions, decisions = list(prefix), []
                with NativeRGBBridge(initial) as env:
                    frames = [env.initial_rgb]
                    for action in prefix:
                        frames.append(env.step(action[None]))
                    np.testing.assert_array_equal(frames[-1], bank[f"r{root}_rgb5"][idx])
                    for decision in range(1, 5):
                        torch.cuda.synchronize()
                        tic = time.monotonic()
                        state = causal_state(frames[-2], frames[-1], camera["metadata"],
                                             old["body_offset_from_static_scene"], .05)
                        context = encode_with_state(model, frames[-1], state, device)
                        callback = PointMazeAPCallback(model, context, goal["proprio"][0, 0], device)
                        generator = torch.Generator(device=device).manual_seed(2026110300 + root*10 + decision)
                        chosen, initial_cost, final_cost, refined = spectral_cem(
                            callback, prior, lower, upper, generator, None)
                        if (callback.batches, callback.samples) != (4, 50):
                            raise RuntimeError("optimizer budget mismatch")
                        raw = np.clip((chosen.reshape(5, 2)*std+mean).cpu().numpy(), -1, 1).astype(np.float32)
                        torch.cuda.synchronize()
                        elapsed = time.monotonic()-tic
                        input_hash = image_hash(frames[-1])
                        for action in raw:
                            frames.append(env.step(action[None]))
                        actions.extend(raw)
                        decisions.append({"input_rgb_sha256": input_hash, "output_rgb_sha256": image_hash(frames[-1]),
                                          "initial_cost": initial_cost, "final_cost": final_cost,
                                          "refined_selected": refined, "prediction_batches": callback.batches,
                                          "prediction_samples": callback.samples, "planning_seconds": elapsed})
                    if env.step_count != 25:
                        raise RuntimeError("physical action clock mismatch")
                seq = np.asarray(actions, np.float32)
                arrays[f"r{root}_{arm}_frames"] = np.stack(frames)
                arrays[f"r{root}_{arm}_actions"] = seq
                starts.append(initial)
                sequences.append(seq)
                terminals.append(frames[-1])
                branches.append({"prefix": arm, "bank_index": int(idx), "decisions": decisions})
                print(f"root {root} {arm}: completed", flush=True)
            rows.append({"root": root, "branches": branches})
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("frozen model changed")
    starts, sequences = np.asarray(starts), np.asarray(sequences)
    cp5 = physical.replay_position_endpoints(starts, sequences[:, :5])
    cp25 = physical.replay_position_endpoints(starts, sequences)
    for i, (initial, seq, expected) in enumerate(zip(starts, sequences, terminals)):
        with NativeRGBBridge(initial) as env:
            for action in seq:
                frame = env.step(action[None])
            if not np.array_equal(frame, expected):
                raise RuntimeError(f"terminal RGB mismatch {i}")
    arrays["cp5_eval_only"], arrays["cp25_eval_only"] = cp5, cp25
    cross_starts, cross_sequences, ids = [], [], []
    for i, root in enumerate(ROOTS):
        tails = {"selected": sequences[3*i, 5:], "greedy": sequences[3*i+1, 5:],
                 "zero": np.zeros((20, 2), np.float32)}
        for p, prefix in enumerate(ARMS):
            for suffix in ARMS:
                ids.append((root, prefix, suffix))
                cross_starts.append(starts[3*i+p])
                cross_sequences.append(np.concatenate((sequences[3*i+p, :5], tails[suffix])))
    cross_starts, cross_sequences = np.asarray(cross_starts), np.asarray(cross_sequences)
    cross_end = physical.replay_position_endpoints(cross_starts, cross_sequences)
    lookup = {key: i for i, key in enumerate(ids)}
    cross_rows = []
    for i, (root, prefix, suffix) in enumerate(ids):
        position = cross_end[i, :2]
        zero_position = cross_end[lookup[root, prefix, "zero"], :2]
        distance = float(np.linalg.norm(position-GOAL[:2]))
        if prefix == suffix and prefix != "zero":
            parent_index = 3*ROOTS.index(root)+ARMS.index(prefix)
            np.testing.assert_array_equal(position, cp25[parent_index, :2])
        cross_rows.append({"root": root, "prefix": prefix, "suffix": suffix, "endpoint_distance": distance,
                           "distance_minus_zero_suffix": distance-float(np.linalg.norm(zero_position-GOAL[:2])),
                           "displacement_minus_zero_suffix": (position-zero_position).tolist()})
    arrays["cross_sequences"], arrays["cross_end_eval_only"] = cross_sequences, cross_end
    for i, row in enumerate(rows):
        for j, branch in enumerate(row["branches"]):
            k = 3*i+j
            branch.update({"initial_distance": float(np.linalg.norm(starts[k, :2]-GOAL[:2])),
                           "cp5_distance": float(np.linalg.norm(cp5[k, :2]-GOAL[:2])),
                           "endpoint_distance": float(np.linalg.norm(cp25[k, :2]-GOAL[:2]))})
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "pc_pointmaze_nonprogress_followup.v1", "rows": rows, "cross_rows": cross_rows,
              "bank_sha256": BANK_SHA, "source_sha256": SOURCE_SHA, "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "model_before": before, "model_after": after, "feedback_trajectories": len(sequences),
              "rgb_replays": len(sequences), "cross_sequences": len(cross_sequences),
              "wall_seconds": time.monotonic()-start_time, "scope": "exposed development roots; no P1"}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
