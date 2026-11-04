"""Two-packet model lookahead for the sealed non-progress spectral bank."""
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
from experiments.pc_pointmaze_obstacle_nonprogress_bank_20261103 import OUTPUT as BANK
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import DATA, GOAL, ROOTS, OUTPUT as SOURCE
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.pc_ap_spectral_cem_20261103 import spectral_cem
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

BANK_SHA = "e45ac931ba3da9c480cc37c125c00d202ee7dd4014abba71ffd88c238b928e1b"
SOURCE_SHA = "e5955123b5b88797a66e4c703a1c0f5b321a23c8fbead371dc029c8d44a5ff91"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_NONPROGRESS_LOOKAHEAD_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_nonprogress_lookahead_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")
ARMS = ("lookahead", "immediate", "greedy", "zero")


def select_lookahead(costs, eligible, greedy):
    values = np.asarray(costs, dtype=np.float64)
    if values.shape != (9, 2) or not np.isfinite(values).all():
        raise ValueError("expected finite 9-by-2 costs")
    candidates = np.asarray(eligible, dtype=np.int64)
    if candidates.ndim != 1 or len(np.unique(candidates)) != len(candidates):
        raise ValueError("invalid eligible indices")
    if np.any(candidates < 1) or np.any(candidates >= 9):
        raise ValueError("eligible prefixes must be nonzero bank indices")
    if not isinstance(greedy, (int, np.integer)) or not 0 <= greedy < 9:
        raise ValueError("invalid greedy index")
    best_tail = np.argmin(values, axis=1)
    best_cost = values[np.arange(9), best_tail]
    selected = int(candidates[np.argmin(best_cost[candidates])]) if len(candidates) else None
    if selected is not None and not (best_cost[selected] < best_cost[0] and best_cost[selected] < best_cost[greedy]):
        selected = None
    return selected, best_tail, best_cost


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((BANK, BANK_SHA), (SOURCE, SOURCE_SHA), (CAMERA, CAMERA_SHA), (OLD, OLD_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed input: {path}")
    bank_doc, source, camera, old = (json.loads(x.read_text()) for x in (BANK, SOURCE, CAMERA, OLD))
    for path, expected in ((BANK.with_suffix(".npz"), bank_doc["arrays_sha256"]),
                           (SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (DATA / "states.pth", source["states_sha256"]),
                           (DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed data: {path}")
    if [r["root"] for r in bank_doc["rows"]] != list(ROOTS):
        raise RuntimeError("root identity mismatch")
    states = torch.load(DATA / "states.pth", weights_only=True, map_location="cpu")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, arrays, starts, sequences = [], {}, [], []
    begun = time.monotonic()
    print("Optimize equally budgeted model suffixes and execute preselected prefixes.", flush=True)
    with np.load(BANK.with_suffix(".npz"), allow_pickle=False) as bank, \
            np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as src, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != bank_doc["model_before"]:
            raise RuntimeError("model identity mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or not (std > 0).all():
            raise RuntimeError("invalid normalization")
        lower = ((torch.full_like(mean, -1.)-mean)/std).repeat(5)
        upper = ((torch.ones_like(mean)-mean)/std).repeat(5)
        prior = ((torch.zeros_like(mean)-mean)/std).repeat(5).reshape(1, 1, 10)
        goal_rgb = src["goal_rgb"]
        goal_state, _ = estimate_state(goal_rgb, camera["metadata"], old["body_offset_from_static_scene"])
        target = encode_with_state(model, goal_rgb, goal_state, device)["proprio"][0, 0]
        for root, parent in zip(ROOTS, bank_doc["rows"]):
            initial = states[root, 40].numpy().astype(np.float64)
            rgb = src[f"r{root}_random_frames"][0]
            state, _ = estimate_state(rgb, camera["metadata"], old["body_offset_from_static_scene"])
            context = encode_with_state(model, rgb, state, device)
            raw = torch.as_tensor(bank[f"r{root}_bank"], device=device)
            packets = ((raw-mean)/std).repeat(1, 5)
            torch.cuda.synchronize()
            tic = time.monotonic()
            terminal, first_scores, tails, counts = [], [], [], []
            for prefix_index in range(9):
                calls = {"batches": 0, "samples": 0}
                observed_first = []
                def callback(proposals):
                    n = proposals.shape[1]
                    prefix = packets[prefix_index].expand(n, -1)
                    actions = torch.stack((prefix, proposals[0, :, 0]))
                    with torch.no_grad():
                        predicted = model.unroll(context, act_suffix=actions)["proprio"]
                        if predicted.shape[0] != 3 or predicted.shape[1] != n:
                            raise RuntimeError("model rollout time or batch mismatch")
                        values = (predicted-target).reshape(3, n, -1).square().mean(dim=-1)
                    observed_first.extend(values[1].cpu().tolist())
                    calls["batches"] += 1
                    calls["samples"] += n
                    return values[2].unsqueeze(0)
                generator = torch.Generator(device=device).manual_seed(2026110370 + root)
                chosen, prior_cost, refined_cost, _ = spectral_cem(
                    callback, prior, lower, upper, generator, None)
                if calls != {"batches": 4, "samples": 50}:
                    raise RuntimeError("suffix optimization budget mismatch")
                np.testing.assert_allclose(observed_first, bank[f"r{root}_scores"][prefix_index],
                                           rtol=2e-5, atol=2e-7)
                terminal.append([prior_cost, refined_cost])
                first_scores.append(observed_first)
                tails.append(chosen[0, 0].cpu().numpy())
                counts.append(calls)
            terminal = np.asarray(terminal, np.float64)
            first_scores = np.asarray(first_scores, np.float64)
            selected, best_tail, best_cost = select_lookahead(terminal, parent["eligible"], parent["greedy"])
            torch.cuda.synchronize()
            elapsed = time.monotonic()-tic
            indices = (selected, parent["selected"], parent["greedy"], 0)
            branches = []
            for arm, idx in zip(ARMS, indices):
                if idx is None:
                    branches.append({"arm": arm, "bank_index": None})
                    continue
                action = bank[f"r{root}_bank"][idx]
                with NativeRGBBridge(initial) as env:
                    if not np.array_equal(env.initial_rgb, rgb):
                        raise RuntimeError("reset RGB mismatch")
                    for _ in range(5):
                        image = env.step(action[None])
                np.testing.assert_array_equal(image, bank[f"r{root}_rgb5"][idx])
                starts.append(initial)
                sequences.append(np.repeat(action[None], 5, axis=0))
                branches.append({"arm": arm, "bank_index": int(idx)})
            arrays[f"r{root}_terminal_costs"] = terminal
            arrays[f"r{root}_first_costs"] = first_scores
            arrays[f"r{root}_optimized_normalized_tails"] = np.stack(tails)
            rows.append({"root": root, "eligible": parent["eligible"], "branches": branches,
                         "best_tail": best_tail.tolist(), "best_cost": best_cost.tolist(),
                         "planning_seconds": elapsed, "two_packet_samples": sum(x["samples"] for x in counts),
                         "predicted_packet_transitions": 2*sum(x["samples"] for x in counts)})
            print(f"root {root}: lookahead={selected}, immediate={parent['selected']}", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("frozen model changed")
    ends = physical.replay_position_endpoints(np.asarray(starts), np.asarray(sequences))
    arrays["endpoints_eval_only"] = ends
    k = 0
    for row in rows:
        for branch in row["branches"]:
            if branch["bank_index"] is None:
                continue
            branch["initial_distance"] = float(np.linalg.norm(starts[k][:2]-GOAL[:2]))
            branch["cp5_distance"] = float(np.linalg.norm(ends[k, :2]-GOAL[:2]))
            k += 1
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "pc_pointmaze_nonprogress_lookahead.v1", "rows": rows,
              "bank_sha256": BANK_SHA, "source_sha256": SOURCE_SHA, "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "model_before": before, "model_after": after, "physical_prefixes": len(sequences),
              "wall_seconds": time.monotonic()-begun, "scope": "exposed roots; no independent P1"}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
