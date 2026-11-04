"""Affine-score future-value selector with real RGB feedback on exposed roots."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA, estimate_state
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import OLD, OLD_SHA
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state
from experiments.pc_pointmaze_affine_feedback_cem_20261103 import projected_cost
from experiments.pc_pointmaze_affine_inverse_20261103 import MEAN, STD, get_encoder
from experiments.pc_pointmaze_affine_spectral_cem_20261103 import LENGTHS_SHA
from experiments.pc_pointmaze_nonprogress_lookahead_20261103 import BANK_SHA, SOURCE_SHA
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

PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_AFFINE_NONPROGRESS_LOOKAHEAD_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_affine_nonprogress_lookahead_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def choose(first_scores, initial_score, terminal_costs):
    first, terminal = np.asarray(first_scores, np.float64), np.asarray(terminal_costs, np.float64)
    if (first.shape != (9,) or terminal.shape != (9, 2) or not np.isfinite(first).all()
            or not np.isfinite(terminal).all() or not np.isfinite(initial_score)):
        raise ValueError("expected nine finite first and terminal costs")
    greedy = int(np.argmin(first))
    eligible = np.flatnonzero((first >= initial_score) & (np.arange(9) != 0))
    best = terminal.min(axis=1)
    selected = int(eligible[np.argmin(best[eligible])]) if len(eligible) else None
    if selected is not None and not (best[selected] < best[0] and best[selected] < best[greedy]):
        selected = None
    return selected, greedy, eligible.tolist()


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((BANK, BANK_SHA), (SOURCE, SOURCE_SHA), (CAMERA, CAMERA_SHA), (OLD, OLD_SHA),
                           (DEFAULT_DATASET / "seq_lengths.pth", LENGTHS_SHA)):
        if sha256_file(path) != expected:
            raise ValueError(f"source changed: {path}")
    bank_doc, source, camera, old = (json.loads(p.read_text()) for p in (BANK, SOURCE, CAMERA, OLD))
    for path, expected in ((BANK.with_suffix(".npz"), bank_doc["arrays_sha256"]),
                           (SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (DATA / "states.pth", source["states_sha256"]),
                           (DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if sha256_file(path) != expected:
            raise ValueError(f"source changed: {path}")
    if [r["root"] for r in bank_doc["rows"]] != list(ROOTS):
        raise ValueError("root roster changed")
    states = torch.load(DATA / "states.pth", weights_only=True, map_location="cpu")
    weight, _, _ = get_encoder()
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    arrays, rows, starts, sequences, terminals = {}, [], [], [], []
    begun = time.monotonic()
    print("Select nonprogress prefixes by affine future score and replan from real RGB.", flush=True)
    with np.load(BANK.with_suffix(".npz"), allow_pickle=False) as bank, \
            np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as src, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != bank_doc["model_before"]:
            raise RuntimeError("model mismatch")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), STD, rtol=0, atol=1e-7)):
            raise RuntimeError("proprio normalizer mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()) or not bool((std > 0).all()):
            raise ValueError("invalid action normalizer")
        lower = ((torch.full_like(mean, -1.) - mean) / std).repeat(5)
        upper = ((torch.ones_like(mean) - mean) / std).repeat(5)
        prior = ((torch.zeros_like(mean) - mean) / std).repeat(5).reshape(1, 1, 10)
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(STD[:2], dtype=torch.float32, device=device)
        goal_rgb = src["goal_rgb"]
        goal_state, _ = estimate_state(goal_rgb, camera["metadata"], old["body_offset_from_static_scene"])
        target = encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        for root in ROOTS:
            initial = states[root, 40].numpy().astype(np.float64)
            rgb = src[f"r{root}_random_frames"][0]
            state, _ = estimate_state(rgb, camera["metadata"], old["body_offset_from_static_scene"])
            context = encode_with_state(model, rgb, state, device)
            initial_score = float(projected_cost(context["proprio"].reshape(1, 256, 16),
                                                  target, pinv, pos_std).item())
            raw_bank = np.asarray(bank[f"r{root}_bank"], np.float32)
            if raw_bank.shape != (9, 2) or not np.isfinite(raw_bank).all() or np.any(np.abs(raw_bank) > 1):
                raise ValueError("invalid candidate bank")
            packets = ((torch.as_tensor(raw_bank, device=device) - mean) / std).repeat(1, 5)
            first, terminal, chosen_tails = [], [], []
            tic = time.monotonic()
            for index in range(9):
                first_samples, calls, saved_batches = [], 0, []
                def evaluate(proposals):
                    nonlocal calls
                    n = proposals.shape[1]
                    actions = torch.stack((packets[index].expand(n, -1), proposals[0, :, 0]))
                    with torch.no_grad():
                        predicted = model.unroll(context, act_suffix=actions)["proprio"]
                        if predicted.shape[:2] != (3, n):
                            raise ValueError("rollout clock or batch mismatch")
                        first_cost = projected_cost(predicted[1].reshape(n, 256, 16), target, pinv, pos_std)
                        end_cost = projected_cost(predicted[2].reshape(n, 256, 16), target, pinv, pos_std)
                    first_samples.extend(first_cost.cpu().tolist())
                    saved_batches.append((proposals.detach().cpu().numpy(), end_cost.cpu().numpy()))
                    calls += n
                    return end_cost.unsqueeze(0)
                generator = torch.Generator(device=device).manual_seed(2026110370 + root)
                chosen, prior_cost, refined_cost, _ = spectral_cem(evaluate, prior, lower, upper, generator, None)
                if calls != 50 or len(saved_batches) != 4:
                    raise RuntimeError("lookahead budget mismatch")
                np.testing.assert_allclose(first_samples, first_samples[0], rtol=1e-5, atol=1e-6)
                first.append(first_samples[0])
                terminal.append([prior_cost, refined_cost])
                chosen_tails.append(chosen[0, 0].cpu().numpy())
                for j, (proposals, costs) in enumerate(saved_batches):
                    arrays[f"r{root}_prefix{index}_lookahead{j}_proposals"] = proposals
                    arrays[f"r{root}_prefix{index}_lookahead{j}_scores"] = costs
            selected, greedy, eligible = choose(first, initial_score, terminal)
            torch.cuda.synchronize()
            planning_seconds = time.monotonic() - tic
            branches = []
            for arm, idx in (("selected", selected), ("greedy", greedy), ("zero", 0)):
                if idx is None:
                    branches.append({"arm": arm, "bank_index": None})
                    continue
                prefix = np.repeat(raw_bank[idx][None], 5, axis=0)
                with NativeRGBBridge(initial) as env:
                    if not np.array_equal(env.initial_rgb, rgb):
                        raise RuntimeError("reset RGB mismatch")
                    history = [env.initial_rgb]
                    for action in prefix:
                        history.append(env.step(action[None]))
                    if not np.array_equal(history[-1], bank[f"r{root}_rgb5"][idx]):
                        raise RuntimeError("sealed bank RGB mismatch")
                    state5 = causal_state(history[-2], history[-1], camera["metadata"],
                                          old["body_offset_from_static_scene"], .05)
                    feedback = encode_with_state(model, history[-1], state5, device)
                    calls, batches = 0, []
                    def feedback_evaluate(proposals):
                        nonlocal calls
                        n = proposals.shape[1]
                        with torch.no_grad():
                            predicted = model.unroll(feedback, act_suffix=proposals[0, :, 0].unsqueeze(0))["proprio"][-1]
                            values = projected_cost(predicted.reshape(n, 256, 16), target, pinv, pos_std)
                        calls += n
                        batches.append((proposals.detach().cpu().numpy(), values.cpu().numpy()))
                        return values.unsqueeze(0)
                    generator = torch.Generator(device=device).manual_seed(2026110380 + root)
                    chosen, prior_cost, refined_cost, refined = spectral_cem(
                        feedback_evaluate, prior, lower, upper, generator, None)
                    if calls != 50 or len(batches) != 4:
                        raise RuntimeError("feedback budget mismatch")
                    tail = np.clip((chosen.reshape(5, 2) * std + mean).cpu().numpy(), -1, 1).astype(np.float32)
                    for action in tail:
                        history.append(env.step(action[None]))
                    if env.step_count != 10:
                        raise RuntimeError("physical clock mismatch")
                sequence = np.concatenate((prefix, tail))
                starts.append(initial)
                sequences.append(sequence)
                terminals.append(history[-1])
                arrays[f"r{root}_{arm}_history"] = np.stack(history)
                arrays[f"r{root}_{arm}_actions"] = sequence
                for j, (proposals, scores) in enumerate(batches):
                    arrays[f"r{root}_{arm}_feedback{j}_proposals"] = proposals
                    arrays[f"r{root}_{arm}_feedback{j}_scores"] = scores
                branches.append({"arm": arm, "bank_index": int(idx), "cp5_input": state5.tolist(),
                                 "feedback_prior_cost": prior_cost, "feedback_refined_cost": refined_cost,
                                 "feedback_refined_selected": refined, "feedback_samples": calls})
            arrays[f"r{root}_first_scores"] = np.asarray(first)
            arrays[f"r{root}_terminal_scores"] = np.asarray(terminal)
            arrays[f"r{root}_lookahead_tails"] = np.asarray(chosen_tails)
            rows.append({"root": root, "initial_affine_score": initial_score, "eligible": eligible,
                         "selected": selected, "greedy": greedy, "branches": branches,
                         "lookahead_samples": 450, "lookahead_seconds": planning_seconds})
            print(f"root {root}: selected={selected}, greedy={greedy}, eligible={len(eligible)}", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    starts, sequences = np.asarray(starts), np.asarray(sequences)
    cp5 = physical.replay_position_endpoints(starts, sequences[:, :5])
    cp10 = physical.replay_position_endpoints(starts, sequences)
    for initial, sequence, expected in zip(starts, sequences, terminals):
        with NativeRGBBridge(initial) as env:
            for action in sequence:
                image = env.step(action[None])
            if not np.array_equal(image, expected):
                raise RuntimeError("terminal RGB replay mismatch")
    cursor = 0
    for row in rows:
        for branch in row["branches"]:
            if branch["bank_index"] is None:
                continue
            branch.update({"initial_distance": float(np.linalg.norm(starts[cursor, :2] - GOAL[:2])),
                           "cp5_distance": float(np.linalg.norm(cp5[cursor, :2] - GOAL[:2])),
                           "cp10_distance": float(np.linalg.norm(cp10[cursor, :2] - GOAL[:2]))})
            cursor += 1
    arrays["starts"], arrays["sequences"], arrays["cp5_eval_only"], arrays["cp10_eval_only"] = starts, sequences, cp5, cp10
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "pc_pointmaze_affine_nonprogress_lookahead.v1", "rows": rows,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "bank_sha256": BANK_SHA, "source_sha256": SOURCE_SHA, "arrays_sha256": sha256_file(ARRAYS),
              "seq_lengths_sha256": LENGTHS_SHA, "model_before": before, "model_after": after,
              "action_mean": mean.cpu().tolist(), "action_std": std.cpu().tolist(),
              "physical_trajectories": len(starts), "physical_replays": len(starts),
              "wall_seconds": time.monotonic() - begun,
              "scope": "exposed obstacle roots; 10 steps; not equal total cost or independent P1"}
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {OUTPUT} sha256={sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
