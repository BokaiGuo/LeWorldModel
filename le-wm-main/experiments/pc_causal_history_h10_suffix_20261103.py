"""Longer suffix search and real-RGB replanning on a sealed development bank."""
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
from experiments.pc_pointmaze_affine_nonprogress_lookahead_20261103 import choose
from experiments.pc_pointmaze_affine_spectral_cem_20261103 import LENGTHS_SHA
from experiments.pc_pointmaze_nonprogress_lookahead_20261103 import SOURCE_SHA as OBSTACLE_SHA
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import GOAL, ROOTS, OUTPUT as OBSTACLE
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.pc_bounded_packet_cem_20261103 import packet_cem
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE = ROOT / "results/pc_causal_history_full_bank_replan_20261103_DEV01.json"
SOURCE_SHA = "d04c8b56024dbe3cbb923bdc5a76f930f9af1e5438b5d5e1c87d72ae41fa316a"
PROTOCOL = ROOT.parent / "docs/PC_CAUSAL_HISTORY_H10_SUFFIX_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_causal_history_h10_suffix_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, digest in ((SOURCE, SOURCE_SHA), (OBSTACLE, OBSTACLE_SHA), (CAMERA, CAMERA_SHA),
                         (OLD, OLD_SHA), (DEFAULT_DATASET / "seq_lengths.pth", LENGTHS_SHA)):
        if sha256_file(path) != digest:
            raise ValueError(f"source changed: {path}")
    source, obstacle, camera, old = (json.loads(p.read_text()) for p in (SOURCE, OBSTACLE, CAMERA, OLD))
    for path, digest in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                         (OBSTACLE.with_suffix(".npz"), obstacle["arrays_sha256"]),
                         (DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if sha256_file(path) != digest:
            raise ValueError(f"source changed: {path}")
    if [row["root"] for row in source["rows"]] != list(ROOTS):
        raise ValueError("unexpected root roster")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    arrays, rows, starts, sequences = {}, [], [], []
    begun = time.monotonic()
    print("Plan fixed prefixes with ten-step suffixes and replan from each real RGB successor.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as old_arrays, \
            np.load(OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle_arrays, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise ValueError("frozen model changed")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), STD, rtol=0, atol=1e-7)):
            raise ValueError("proprio normalizer mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("invalid action normalization")
        lower = ((-torch.ones_like(mean) - mean) / std).repeat(10)
        upper = ((torch.ones_like(mean) - mean) / std).repeat(10)
        prior = ((torch.zeros_like(mean) - mean) / std).repeat(10).reshape(1, 1, 20)
        weight, _, _ = get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(STD[:2], dtype=torch.float32, device=device)
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        goal_rgb = obstacle_arrays["goal_rgb"]
        goal_state, _ = estimate_state(goal_rgb, metadata, offset)
        target = encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        all_starts = old_arrays["starts"]
        if all_starts.shape != (36, 4):
            raise ValueError("invalid saved resets")
        for k, root in enumerate(ROOTS):
            initial = all_starts[9*k].astype(np.float64)
            if not np.array_equal(all_starts[9*k:9*k+9], np.broadcast_to(initial, (9, 4))):
                raise ValueError("prefix reset differs")
            first_history = old_arrays[f"r{root}_p0_history"]
            state2 = causal_state(first_history[1], first_history[2], metadata, offset, .05)
            np.testing.assert_allclose(state2, source["rows"][k]["initial_rgb_state_after_prelude"],
                                       rtol=0, atol=1e-7)
            context = encode_with_state(model, first_history[2], state2, device)
            current_score = float(projected_cost(context["proprio"].reshape(1, 256, 16),
                                                 target, pinv, pos_std).item())
            np.testing.assert_allclose(current_score, source["rows"][k]["current_score"], rtol=1e-5, atol=1e-6)
            bank = old_arrays[f"r{root}_bank"]
            if bank.shape != (9, 5, 2) or not np.isfinite(bank).all() or np.any(np.abs(bank) > 1):
                raise ValueError("invalid sealed prefix bank")
            first_scores, terminal_scores, old_tails = [], [], []
            for index, prefix in enumerate(bank):
                normalized = ((torch.as_tensor(prefix, device=device) - mean) / std).reshape(1, 10)
                first_samples, batches = [], []

                def evaluate(proposals):
                    n = proposals.shape[1]
                    tail = proposals[0, :, 0].reshape(n, 2, 10).permute(1, 0, 2)
                    actions = torch.cat((normalized.expand(n, -1).unsqueeze(0), tail))
                    with torch.no_grad():
                        prediction = model.unroll(context, act_suffix=actions)["proprio"]
                        if prediction.shape[:2] != (4, n):
                            raise ValueError("three-packet clock mismatch")
                        first = projected_cost(prediction[1].reshape(n, 256, 16), target, pinv, pos_std)
                        terminal = projected_cost(prediction[3].reshape(n, 256, 16), target, pinv, pos_std)
                    first_samples.extend(first.cpu().tolist())
                    batches.append((proposals.detach().cpu().numpy(), terminal.cpu().numpy()))
                    return terminal.unsqueeze(0)

                generator = torch.Generator(device=device).manual_seed(2026110370 + root)
                chosen, score0, score1, _ = packet_cem(evaluate, prior, lower, upper, generator)
                if len(first_samples) != 50 or len(batches) != 4:
                    raise RuntimeError("lookahead budget mismatch")
                np.testing.assert_allclose(first_samples, first_samples[0], rtol=1e-5, atol=1e-6)
                first_scores.append(first_samples[0])
                terminal_scores.append([score0, score1])
                old_tails.append(chosen[0, 0].cpu().numpy())
                for j, (proposals, scores) in enumerate(batches):
                    arrays[f"r{root}_p{index}_lookahead{j}_proposals"] = proposals
                    arrays[f"r{root}_p{index}_lookahead{j}_scores"] = scores
            np.testing.assert_allclose(first_scores, old_arrays[f"r{root}_first_scores"], rtol=1e-5, atol=1e-6)
            selected, greedy, eligible = choose(first_scores, current_score, terminal_scores)
            arrays[f"r{root}_bank"] = bank.copy()
            arrays[f"r{root}_first_scores"] = np.asarray(first_scores)
            arrays[f"r{root}_terminal_scores"] = np.asarray(terminal_scores)
            arrays[f"r{root}_old_tails"] = np.asarray(old_tails)
            print(f"root {root}: sealed selected={selected}, greedy={greedy}, eligible={len(eligible)}", flush=True)
            branches = []
            for index, prefix in enumerate(bank):
                prior_history = old_arrays[f"r{root}_p{index}_history"]
                with NativeRGBBridge(initial) as env:
                    history = [env.initial_rgb]
                    for action in np.concatenate((np.zeros((2, 2), np.float32), prefix)):
                        history.append(env.step(action[None]))
                    if not np.array_equal(np.stack(history), prior_history[:8]):
                        raise RuntimeError("source prefix RGB history mismatch")
                    state7 = causal_state(history[6], history[7], metadata, offset, .05)
                    feedback = encode_with_state(model, history[7], state7, device)
                    batches, count = [], 0

                    def feedback_evaluate(proposals):
                        nonlocal count
                        n = proposals.shape[1]
                        tail = proposals[0, :, 0].reshape(n, 2, 10).permute(1, 0, 2)
                        with torch.no_grad():
                            prediction = model.unroll(feedback, act_suffix=tail)["proprio"]
                            if prediction.shape[:2] != (3, n):
                                raise ValueError("two-packet feedback clock mismatch")
                            scores = projected_cost(prediction[2].reshape(n, 256, 16), target, pinv, pos_std)
                        count += n
                        batches.append((proposals.detach().cpu().numpy(), scores.cpu().numpy()))
                        return scores.unsqueeze(0)

                    generator = torch.Generator(device=device).manual_seed(2026110380 + root)
                    chosen, score0, score1, refined = packet_cem(feedback_evaluate, prior, lower, upper, generator)
                    if count != 50 or len(batches) != 4:
                        raise RuntimeError("feedback budget mismatch")
                    tail = np.clip((chosen.reshape(10, 2) * std + mean).cpu().numpy(), -1, 1).astype(np.float32)
                    for action in tail:
                        history.append(env.step(action[None]))
                    if env.step_count != 17:
                        raise RuntimeError("physical clock mismatch")
                actions = np.concatenate((np.zeros((2, 2), np.float32), prefix, tail))
                arrays[f"r{root}_p{index}_actions"] = actions
                arrays[f"r{root}_p{index}_history"] = np.stack(history)
                for j, (proposals, scores) in enumerate(batches):
                    arrays[f"r{root}_p{index}_feedback{j}_proposals"] = proposals
                    arrays[f"r{root}_p{index}_feedback{j}_scores"] = scores
                starts.append(initial)
                sequences.append(actions)
                branches.append({"bank_index": index, "cp7_rgb_state": state7.tolist(),
                                 "feedback_prior_score": score0, "feedback_refined_score": score1,
                                 "feedback_refined": refined, "feedback_samples": count})
            rows.append({"root": root, "current_score": current_score, "eligible": eligible,
                         "selected": selected, "greedy": greedy, "branches": branches})
        torch.cuda.synchronize()
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    starts, sequences = np.asarray(starts), np.asarray(sequences)
    cp2 = physical.replay_position_endpoints(starts, sequences[:, :2])
    cp7 = physical.replay_position_endpoints(starts, sequences[:, :7])
    cp17 = physical.replay_position_endpoints(starts, sequences)
    arrays["starts"], arrays["sequences"] = starts, sequences
    arrays["cp2_eval_only"], arrays["cp7_eval_only"], arrays["cp17_eval_only"] = cp2, cp7, cp17
    for k, row in enumerate(rows):
        for i, branch in enumerate(row["branches"]):
            index = k*9 + i
            for label, position in (("cp2", cp2[index]), ("cp7", cp7[index]), ("cp17", cp17[index])):
                branch[f"{label}_distance"] = float(np.linalg.norm(position[:2] - GOAL[:2]))
            branch["physical_nonprogress_cp7"] = branch["cp7_distance"] >= branch["cp2_distance"]
    with ARRAYS.open("xb") as out:
        np.savez_compressed(out, **arrays)
    report = {"schema": "pc_causal_history_h10_suffix.v1", "rows": rows,
              "scope": "exposed fixed-prefix development; no independent or equal-total-budget admission",
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "cem_kernel_sha256": sha256_file(Path(packet_cem.__code__.co_filename)),
              "source_sha256": SOURCE_SHA, "arrays_sha256": sha256_file(ARRAYS),
              "checkpoint_sha256": sha256_file(DEFAULT_CHECKPOINT), "model_before": before, "model_after": after,
              "counts": {"goal_encodes": 1, "initial_context_encodes": 4, "feedback_encodes": 36,
                         "lookahead_three_packet_samples": 1800, "feedback_two_packet_samples": 1800,
                         "branch_physical_steps": 612, "position_backend_replay_steps": 936},
              "inherited_upstream_cost": {"jvp_calls": 40, "finite_difference_samples": 80,
                                          "note": "source-bank construction; no new JVP in this run"},
              "wall_seconds": time.monotonic() - begun}
    with OUTPUT.open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(f"Saved {OUTPUT} sha256={sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
