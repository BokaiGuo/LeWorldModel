"""Real RGB prelude, predicted program selection and per-prefix physical replanning."""
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
from experiments.pc_pointmaze_five_action_response_bank_20261103 import make_bank
from experiments.pc_pointmaze_nonprogress_lookahead_20261103 import SOURCE_SHA
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import DATA, GOAL, ROOTS, OUTPUT as SOURCE
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.pc_ap_spectral_cem_20261103 import spectral_cem
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

PROTOCOL = ROOT.parent / "docs/PC_CAUSAL_HISTORY_FULL_BANK_REPLAN_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_causal_history_full_bank_replan_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (CAMERA, CAMERA_SHA), (OLD, OLD_SHA),
                           (DEFAULT_DATASET / "seq_lengths.pth", LENGTHS_SHA)):
        if sha256_file(path) != expected:
            raise ValueError(f"input changed: {path}")
    source, camera, old = (json.loads(p.read_text()) for p in (SOURCE, CAMERA, OLD))
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (DATA / "states.pth", source["states_sha256"]),
                           (DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if sha256_file(path) != expected:
            raise ValueError(f"input changed: {path}")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    states = torch.load(DATA / "states.pth", weights_only=True, map_location="cpu")
    arrays, rows, starts, sequences, terminal_frames = {}, [], [], [], []
    tic = time.monotonic()
    print("Acquire causal RGB history, select model programs, replan every real prefix.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise ValueError("model hash mismatch")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), STD, rtol=0, atol=1e-7)):
            raise ValueError("proprio normalization mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("invalid action normalization")
        lower = ((-torch.ones_like(mean) - mean) / std).repeat(5)
        upper = ((torch.ones_like(mean) - mean) / std).repeat(5)
        prior = ((torch.zeros_like(mean) - mean) / std).repeat(5).reshape(1, 1, 10)
        weight, _, _ = get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(STD[:2], dtype=torch.float32, device=device)
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        goal_rgb = saved["goal_rgb"]
        goal_state, _ = estimate_state(goal_rgb, metadata, offset)
        target = encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        for root in ROOTS:
            initial = states[root, 40].numpy().astype(np.float64)
            with NativeRGBBridge(initial) as env:
                image0 = env.initial_rgb
                if not np.array_equal(image0, saved[f"r{root}_random_frames"][0]):
                    raise RuntimeError("source initial RGB mismatch")
                image1 = env.step(np.zeros((1, 2), dtype=np.float32))
                image2 = env.step(np.zeros((1, 2), dtype=np.float32))
            state2 = causal_state(image1, image2, metadata, offset, .05)
            context = encode_with_state(model, image2, state2, device)
            current_score = float(projected_cost(context["proprio"].reshape(1, 256, 16),
                                                 target, pinv, pos_std).item())

            def task_output(raw):
                packet = ((raw.reshape(5, 2) - mean) / std).reshape(1, 1, 10)
                pred = model.unroll(context, act_suffix=packet)["proprio"]
                if pred.shape[:2] != (2, 1):
                    raise ValueError("one-packet clock mismatch")
                return (pred[-1].reshape(256, 16).mean(dim=0) @ pinv.T)[:2] * pos_std

            zero, basis = torch.zeros(10, device=device), torch.eye(10, device=device)
            jac = torch.stack([torch.autograd.functional.jvp(task_output, zero, v)[1].detach()
                               for v in basis], dim=1).cpu().numpy()
            with torch.no_grad():
                fd = np.stack([((task_output(zero + .001 * v) - task_output(zero - .001 * v)) / .002)
                               .cpu().numpy() for v in basis], axis=1)
            bank, singular, directions = make_bank(jac)
            arrays[f"r{root}_bank"], arrays[f"r{root}_jacobian"] = bank, jac
            arrays[f"r{root}_fd_jacobian"], arrays[f"r{root}_directions"] = fd, directions
            fd_rel = float(np.linalg.norm(fd - jac) / max(np.linalg.norm(jac), 1e-12))
            first_scores, terminal_scores, old_tails = [], [], []
            for index, prefix in enumerate(bank):
                normalized = ((torch.as_tensor(prefix, device=device) - mean) / std).reshape(1, 10)
                first_samples, batches = [], []

                def evaluate(proposals):
                    n = proposals.shape[1]
                    actions = torch.stack((normalized.expand(n, -1), proposals[0, :, 0]))
                    with torch.no_grad():
                        predicted = model.unroll(context, act_suffix=actions)["proprio"]
                        if predicted.shape[:2] != (3, n):
                            raise ValueError("two-packet clock mismatch")
                        a = projected_cost(predicted[1].reshape(n, 256, 16), target, pinv, pos_std)
                        b = projected_cost(predicted[2].reshape(n, 256, 16), target, pinv, pos_std)
                    first_samples.extend(a.cpu().tolist())
                    batches.append((proposals.detach().cpu().numpy(), b.cpu().numpy()))
                    return b.unsqueeze(0)

                generator = torch.Generator(device=device).manual_seed(2026110370 + root)
                chosen, score0, score1, _ = spectral_cem(evaluate, prior, lower, upper, generator)
                if len(first_samples) != 50 or len(batches) != 4:
                    raise RuntimeError("lookahead budget mismatch")
                np.testing.assert_allclose(first_samples, first_samples[0], rtol=1e-5, atol=1e-6)
                first_scores.append(first_samples[0])
                terminal_scores.append([score0, score1])
                old_tails.append(chosen[0, 0].cpu().numpy())
                for j, (proposals, scores) in enumerate(batches):
                    arrays[f"r{root}_p{index}_lookahead{j}_proposals"] = proposals
                    arrays[f"r{root}_p{index}_lookahead{j}_scores"] = scores
            selected, greedy, eligible = choose(first_scores, current_score, terminal_scores)
            arrays[f"r{root}_first_scores"] = np.asarray(first_scores)
            arrays[f"r{root}_terminal_scores"] = np.asarray(terminal_scores)
            arrays[f"r{root}_old_tails"] = np.asarray(old_tails)
            branches = []
            print(f"root {root}: sealed selected={selected}, greedy={greedy}, eligible={len(eligible)}",
                  flush=True)
            for index, prefix in enumerate(bank):
                with NativeRGBBridge(initial) as env:
                    if not np.array_equal(env.initial_rgb, image0):
                        raise RuntimeError("branch reset RGB mismatch")
                    history = [env.initial_rgb, env.step(np.zeros((1, 2), dtype=np.float32)),
                               env.step(np.zeros((1, 2), dtype=np.float32))]
                    if not (np.array_equal(history[1], image1) and np.array_equal(history[2], image2)):
                        raise RuntimeError("branch prelude RGB mismatch")
                    for action in prefix:
                        history.append(env.step(action[None]))
                    state7 = causal_state(history[6], history[7], metadata, offset, .05)
                    feedback = encode_with_state(model, history[7], state7, device)
                    batches, count = [], 0

                    def feedback_evaluate(proposals):
                        nonlocal count
                        n = proposals.shape[1]
                        with torch.no_grad():
                            prediction = model.unroll(feedback, act_suffix=proposals[0, :, 0].unsqueeze(0))["proprio"]
                            if prediction.shape[:2] != (2, n):
                                raise ValueError("feedback clock mismatch")
                            values = projected_cost(prediction[-1].reshape(n, 256, 16),
                                                    target, pinv, pos_std)
                        count += n
                        batches.append((proposals.detach().cpu().numpy(), values.cpu().numpy()))
                        return values.unsqueeze(0)

                    generator = torch.Generator(device=device).manual_seed(2026110380 + root)
                    tail, score0, score1, refined = spectral_cem(feedback_evaluate, prior, lower, upper, generator)
                    if count != 50 or len(batches) != 4:
                        raise RuntimeError("feedback budget mismatch")
                    raw_tail = np.clip((tail.reshape(5, 2) * std + mean).cpu().numpy(), -1, 1).astype(np.float32)
                    for action in raw_tail:
                        history.append(env.step(action[None]))
                    if env.step_count != 12:
                        raise RuntimeError("real action clock mismatch")
                actions = np.concatenate((np.zeros((2, 2), np.float32), prefix, raw_tail))
                arrays[f"r{root}_p{index}_history"] = np.stack(history)
                arrays[f"r{root}_p{index}_actions"] = actions
                for j, (proposals, scores) in enumerate(batches):
                    arrays[f"r{root}_p{index}_feedback{j}_proposals"] = proposals
                    arrays[f"r{root}_p{index}_feedback{j}_scores"] = scores
                starts.append(initial)
                sequences.append(actions)
                terminal_frames.append(history[-1])
                branches.append({"bank_index": index, "cp7_rgb_state": state7.tolist(),
                                 "feedback_prior_score": score0, "feedback_refined_score": score1,
                                 "feedback_refined": refined, "feedback_samples": count})
            rows.append({"root": root, "initial_rgb_state_after_prelude": state2.tolist(),
                         "current_score": current_score, "singular_values": singular.tolist(),
                         "fd_relative_error": fd_rel, "eligible": eligible,
                         "selected": selected, "greedy": greedy, "branches": branches})
        torch.cuda.synchronize()
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    starts, sequences = np.asarray(starts), np.asarray(sequences)
    cp2 = physical.replay_position_endpoints(starts, sequences[:, :2])
    cp7 = physical.replay_position_endpoints(starts, sequences[:, :7])
    cp12 = physical.replay_position_endpoints(starts, sequences)
    arrays["starts"], arrays["sequences"] = starts, sequences
    arrays["cp2_eval_only"], arrays["cp7_eval_only"], arrays["cp12_eval_only"] = cp2, cp7, cp12
    for k, row in enumerate(rows):
        for j, branch in enumerate(row["branches"]):
            index = 9 * k + j
            branch["cp2_distance"] = float(np.linalg.norm(cp2[index, :2] - GOAL[:2]))
            branch["cp7_distance"] = float(np.linalg.norm(cp7[index, :2] - GOAL[:2]))
            branch["cp12_distance"] = float(np.linalg.norm(cp12[index, :2] - GOAL[:2]))
            branch["physical_nonprogress_cp7"] = branch["cp7_distance"] >= branch["cp2_distance"]
    with ARRAYS.open("xb") as out:
        np.savez_compressed(out, **arrays)
    report = {"schema": "pc_causal_history_full_bank_replan.v1", "rows": rows,
              "scope": "exposed development roots; no independent or equal-total-budget admission",
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "source_sha256": SOURCE_SHA, "arrays_sha256": sha256_file(ARRAYS),
              "checkpoint_sha256": sha256_file(DEFAULT_CHECKPOINT),
              "config_sha256": sha256_file(DEFAULT_CONFIG), "model_before": before, "model_after": after,
              "counts": {"goal_encodes": 1, "initial_context_encodes": 4, "feedback_encodes": 36,
                         "jvp_calls": 40, "finite_difference_samples": 80,
                         "lookahead_samples": 1800, "feedback_samples": 1800,
                         "prelude_acquisition_steps": 8, "branch_physical_steps": 432,
                         "position_backend_replay_steps": 756},
              "wall_seconds": time.monotonic() - tic}
    with OUTPUT.open("x") as out:
        json.dump(report, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
    print(f"Saved {OUTPUT} sha256={sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
