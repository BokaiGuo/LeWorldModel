"""Execute sealed optimized action pairs and diagnose prediction alignment."""
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
from experiments.pc_pointmaze_nonprogress_lookahead_20261103 import OUTPUT as LOOKAHEAD
from experiments.pc_pointmaze_obstacle_nonprogress_bank_20261103 import OUTPUT as BANK
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import DATA, GOAL, ROOTS, OUTPUT as SOURCE
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

LOOKAHEAD_SHA = "fef972f1527f14b6439bb8fc6a7f60061b1be37a0324756d0c3baa92f256259a"
BANK_SHA = "e45ac931ba3da9c480cc37c125c00d202ee7dd4014abba71ffd88c238b928e1b"
SOURCE_SHA = "e5955123b5b88797a66e4c703a1c0f5b321a23c8fbead371dc029c8d44a5ff91"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_LOOKAHEAD_SAME_ACTION_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_lookahead_same_action_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def pair_reversals(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid ranking inputs")
    i, j = np.triu_indices(len(a), k=1)
    x, y = a[i]-a[j], b[i]-b[j]
    usable = (x != 0) & (y != 0)
    return {"pairs": int(len(x)), "comparable": int(usable.sum()),
            "reversed": int(((x*y < 0) & usable).sum())}


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((LOOKAHEAD, LOOKAHEAD_SHA), (BANK, BANK_SHA), (SOURCE, SOURCE_SHA),
                           (CAMERA, CAMERA_SHA), (OLD, OLD_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed input: {path}")
    look, bank_doc, source, camera, old = (json.loads(x.read_text()) for x in (LOOKAHEAD, BANK, SOURCE, CAMERA, OLD))
    for path, expected in ((LOOKAHEAD.with_suffix(".npz"), look["arrays_sha256"]),
                           (BANK.with_suffix(".npz"), bank_doc["arrays_sha256"]),
                           (SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (DATA / "states.pth", source["states_sha256"]),
                           (DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed data: {path}")
    states = torch.load(DATA / "states.pth", weights_only=True, map_location="cpu")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, arrays, starts, sequences = [], {}, [], []
    begun = time.monotonic()
    print("Execute all sealed prefixes and optimized tails; compare same-action model predictions.", flush=True)
    with np.load(LOOKAHEAD.with_suffix(".npz"), allow_pickle=False) as sealed, \
            np.load(BANK.with_suffix(".npz"), allow_pickle=False) as bank, \
            np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as source_arrays, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != look["model_before"]:
            raise RuntimeError("model identity mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or not (std > 0).all():
            raise RuntimeError("invalid normalizer")
        goal_rgb = source_arrays["goal_rgb"]
        goal_state, _ = estimate_state(goal_rgb, camera["metadata"], old["body_offset_from_static_scene"])
        target = encode_with_state(model, goal_rgb, goal_state, device)["proprio"][0, 0]
        arrays["target_token"] = target.detach().cpu().numpy()
        for root, parent in zip(ROOTS, look["rows"]):
            if root != parent["root"]:
                raise RuntimeError("root order mismatch")
            initial = states[root, 40].numpy().astype(np.float64)
            initial_rgb = source_arrays[f"r{root}_random_frames"][0]
            initial_state, _ = estimate_state(initial_rgb, camera["metadata"], old["body_offset_from_static_scene"])
            context = encode_with_state(model, initial_rgb, initial_state, device)
            all_frames, all_actions, open_tokens, refresh_tokens, actual_tokens, branches = [], [], [], [], [], []
            for idx in range(9):
                prefix = np.repeat(bank[f"r{root}_bank"][idx][None], 5, axis=0)
                normalized_tail = torch.as_tensor(sealed[f"r{root}_optimized_normalized_tails"][idx],
                                                  dtype=torch.float32, device=device).reshape(5, 2)
                tail = np.clip((normalized_tail*std+mean).cpu().numpy(), -1, 1).astype(np.float32)
                actual_norm = (torch.as_tensor(tail, device=device)-mean)/std
                roundtrip = float(torch.max(torch.abs(actual_norm-normalized_tail)).cpu())
                prefix_norm = (torch.as_tensor(prefix, device=device)-mean)/std
                packets = torch.stack((prefix_norm.reshape(10), actual_norm.reshape(10))).unsqueeze(1)
                with torch.no_grad():
                    rollout = model.unroll(context, act_suffix=packets)["proprio"]
                    predicted_open = rollout[-1, 0]
                    open_cost = float((predicted_open-target).square().mean())
                np.testing.assert_allclose(open_cost, parent["best_cost"][idx], rtol=2e-5, atol=2e-7)
                with NativeRGBBridge(initial) as env:
                    if not np.array_equal(env.initial_rgb, initial_rgb):
                        raise RuntimeError("reset image mismatch")
                    frames = [env.initial_rgb]
                    for action in np.concatenate((prefix, tail)):
                        frames.append(env.step(action[None]))
                    if env.step_count != 10:
                        raise RuntimeError("action clock mismatch")
                np.testing.assert_array_equal(frames[5], bank[f"r{root}_rgb5"][idx])
                state5 = causal_state(frames[4], frames[5], camera["metadata"], old["body_offset_from_static_scene"], .05)
                state10 = causal_state(frames[9], frames[10], camera["metadata"], old["body_offset_from_static_scene"], .05)
                real5 = encode_with_state(model, frames[5], state5, device)
                real10 = encode_with_state(model, frames[10], state10, device)
                with torch.no_grad():
                    predicted_refresh = model.unroll(real5, act_suffix=actual_norm.reshape(1, 1, 10))["proprio"][-1, 0]
                    actual_token = real10["proprio"][0, 0]
                    refresh_cost = float((predicted_refresh-target).square().mean())
                    actual_cost = float((actual_token-target).square().mean())
                all_frames.append(np.stack(frames))
                actions = np.concatenate((prefix, tail))
                all_actions.append(actions)
                open_tokens.append(predicted_open.cpu().numpy())
                refresh_tokens.append(predicted_refresh.cpu().numpy())
                actual_tokens.append(actual_token.cpu().numpy())
                starts.append(initial)
                sequences.append(actions)
                branches.append({"index": idx, "open_cost": open_cost, "refresh_cost": refresh_cost,
                                 "actual_encoded_cost": actual_cost, "normalized_roundtrip_max": roundtrip,
                                 "state5_estimate": state5.tolist(), "state10_estimate": state10.tolist()})
            arrays[f"r{root}_frames"] = np.stack(all_frames)
            arrays[f"r{root}_actions"] = np.stack(all_actions)
            arrays[f"r{root}_open_tokens"] = np.stack(open_tokens)
            arrays[f"r{root}_refresh_tokens"] = np.stack(refresh_tokens)
            arrays[f"r{root}_actual_tokens"] = np.stack(actual_tokens)
            rows.append({"root": root, "branches": branches})
            print(f"root {root}: nine action chains executed and encoded", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    endpoints = physical.replay_position_endpoints(np.asarray(starts), np.asarray(sequences))
    arrays["endpoints_eval_only"] = endpoints
    for i, row in enumerate(rows):
        for j, b in enumerate(row["branches"]):
            b["physical_distance"] = float(np.linalg.norm(endpoints[9*i+j, :2]-GOAL[:2]))
        b = row["branches"]
        row["ranking"] = {"open_vs_actual": pair_reversals([x["open_cost"] for x in b], [x["actual_encoded_cost"] for x in b]),
                          "refresh_vs_actual": pair_reversals([x["refresh_cost"] for x in b], [x["actual_encoded_cost"] for x in b]),
                          "actual_vs_physical": pair_reversals([x["actual_encoded_cost"] for x in b], [x["physical_distance"] for x in b])}
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "pc_pointmaze_lookahead_same_action.v1", "rows": rows,
              "lookahead_sha256": LOOKAHEAD_SHA, "bank_sha256": BANK_SHA, "source_sha256": SOURCE_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "model_before": before, "model_after": after,
              "physical_trajectories": len(sequences), "open_two_packet_predictions": len(sequences),
              "refresh_one_packet_predictions": len(sequences), "actual_rgb_encodings": 2*len(sequences),
              "wall_seconds": time.monotonic()-begun, "scope": "exposed roots; no independent P1"}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
