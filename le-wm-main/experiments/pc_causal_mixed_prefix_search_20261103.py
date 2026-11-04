"""Exposed-root mixed spectral versus raw prefix planning with real feedback."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import pc_causal_history_h10_suffix_20261103 as h

PROTOCOL = h.ROOT.parent / "docs/PC_CAUSAL_MIXED_PREFIX_SEARCH_PROTOCOL_20261103.md"
OUTPUT = h.ROOT / "results/pc_causal_mixed_prefix_search_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def candidates(jacobian, seed):
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.shape != (2, 10) or not np.isfinite(jacobian).all():
        raise ValueError("expected finite 2x10 task Jacobian")
    _, singular, vh = np.linalg.svd(jacobian, full_matrices=True)
    if singular[0] <= 0 or singular[1] <= 1e-5 * singular[0]:
        raise ValueError("task Jacobian not rank two")
    rng = np.random.default_rng(seed)
    coeff = rng.normal(size=(4, 10))
    coeff /= np.linalg.norm(coeff, axis=1, keepdims=True)
    coeff *= rng.uniform(size=(4, 1)) ** .1
    raw = .5 * coeff
    spectral = .5 * (coeff * np.array([.25, .25] + [1.] * 8)) @ vh
    result = np.stack((raw, spectral)).reshape(2, 4, 5, 2).astype(np.float32)
    if not np.isfinite(result).all() or np.any(np.abs(result) > .5 + 1e-6):
        raise ValueError("invalid action candidate")
    return result, singular


def select(first, terminal, current):
    first, terminal = np.asarray(first), np.asarray(terminal)
    if first.shape != (6,) or terminal.shape != (6,) or not np.isfinite(
            np.r_[first, terminal, current]).all():
        raise ValueError("invalid candidate scores")
    winner = int(np.argmin(terminal))
    eligible = [i for i in range(1, 6) if first[i] >= current]
    qualified = min(eligible, key=lambda i: (terminal[i], i)) if eligible else None
    if qualified is not None and not (terminal[qualified] < terminal[0]
                                       and terminal[qualified] < terminal[1]):
        qualified = None
    return winner, qualified, eligible


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((h.SOURCE, h.SOURCE_SHA), (h.OBSTACLE, h.OBSTACLE_SHA),
                           (h.CAMERA, h.CAMERA_SHA), (h.OLD, h.OLD_SHA),
                           (h.DEFAULT_DATASET / "seq_lengths.pth", h.LENGTHS_SHA)):
        if h.sha256_file(path) != expected:
            raise ValueError(f"source changed: {path}")
    source, obstacle, camera, old = (json.loads(p.read_text())
                                     for p in (h.SOURCE, h.OBSTACLE, h.CAMERA, h.OLD))
    for path, expected in ((h.SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (h.OBSTACLE.with_suffix(".npz"), obstacle["arrays_sha256"]),
                           (h.DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if h.sha256_file(path) != expected:
            raise ValueError(f"source changed: {path}")
    if [r["root"] for r in source["rows"]] != list(h.ROOTS):
        raise ValueError("root roster mismatch")
    args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                              config=h.DEFAULT_CONFIG, dinov2_source=h.ROOT / "external/local_dinov2")
    arrays, rows, starts, sequences = {}, [], [], []
    begun = time.monotonic()
    print("Compare mixed prefix sources with real-RGB ten-step feedback replanning.", flush=True)
    with np.load(h.SOURCE.with_suffix(".npz"), allow_pickle=False) as old_arrays, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle_arrays, \
            h.lock.gpu_lock(h.ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        model, dataset = h.load_model_and_dataset(args, device)
        before = h.parameter_hash(model)
        if before != source["model_before"]:
            raise ValueError("frozen model mismatch")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), h.MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), h.STD, rtol=0, atol=1e-7)):
            raise ValueError("proprio normalizer mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("invalid action normalizer")
        lower, upper = ((-torch.ones_like(mean) - mean) / std).repeat(10), ((torch.ones_like(mean) - mean) / std).repeat(10)
        prior = ((torch.zeros_like(mean) - mean) / std).repeat(10).reshape(1, 1, 20)
        weight, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        goal_rgb = obstacle_arrays["goal_rgb"]
        goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
        target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        for k, root in enumerate(h.ROOTS):
            initial = old_arrays["starts"][9*k].astype(np.float64)
            saved_history = old_arrays[f"r{root}_p0_history"]
            state2 = h.causal_state(saved_history[1], saved_history[2], metadata, offset, .05)
            np.testing.assert_allclose(state2, source["rows"][k]["initial_rgb_state_after_prelude"],
                                       rtol=0, atol=1e-7)
            context = h.encode_with_state(model, saved_history[2], state2, device)
            current = float(h.projected_cost(context["proprio"].reshape(1, 256, 16),
                                             target, pinv, pos_std).item())
            np.testing.assert_allclose(current, source["rows"][k]["current_score"], rtol=1e-5, atol=1e-6)
            proposals, singular = candidates(old_arrays[f"r{root}_jacobian"], 2026110390 + root)
            source_bank = old_arrays[f"r{root}_bank"]
            source_first = old_arrays[f"r{root}_first_scores"]
            source_greedy = int(np.argmin(source_first))
            shared = np.stack((np.zeros((5, 2), np.float32), source_bank[source_greedy]))
            banks = np.concatenate((np.repeat(shared[None], 2, axis=0), proposals), axis=1)
            arrays[f"r{root}_banks"] = banks
            all_first, all_terminal, all_tails = [], [], []
            for arm in range(2):
                first, terminal, tails = [], [], []
                for i, prefix in enumerate(banks[arm]):
                    norm = ((torch.as_tensor(prefix, device=device) - mean) / std).reshape(1, 10)
                    first_batch, saved_batches = [], []

                    def evaluate(candidates_tensor):
                        n = candidates_tensor.shape[1]
                        tail = candidates_tensor[0, :, 0].reshape(n, 2, 10).permute(1, 0, 2)
                        actions = torch.cat((norm.expand(n, -1).unsqueeze(0), tail))
                        with torch.no_grad():
                            pred = model.unroll(context, act_suffix=actions)["proprio"]
                            if pred.shape[:2] != (4, n):
                                raise ValueError("lookahead clock mismatch")
                            scores = h.projected_cost(pred[3].reshape(n, 256, 16), target, pinv, pos_std)
                            fp = h.projected_cost(pred[1].reshape(n, 256, 16), target, pinv, pos_std)
                        first_batch.extend(fp.cpu().tolist())
                        saved_batches.append((candidates_tensor.detach().cpu().numpy(), scores.cpu().numpy()))
                        return scores.unsqueeze(0)

                    rng = torch.Generator(device=device).manual_seed(2026110370 + root)
                    chosen, prior_score, refined_score, refined = h.packet_cem(evaluate, prior, lower, upper, rng)
                    if len(first_batch) != 50 or len(saved_batches) != 4:
                        raise RuntimeError("lookahead budget mismatch")
                    np.testing.assert_allclose(first_batch, first_batch[0], rtol=1e-5, atol=1e-6)
                    first.append(first_batch[0])
                    terminal.append(min(prior_score, refined_score))
                    tails.append(chosen[0, 0].cpu().numpy())
                    for j, (c, s) in enumerate(saved_batches):
                        arrays[f"r{root}_a{arm}_p{i}_lookahead{j}_candidates"] = c
                        arrays[f"r{root}_a{arm}_p{i}_lookahead{j}_scores"] = s
                all_first.append(first)
                all_terminal.append(terminal)
                all_tails.append(tails)
            np.testing.assert_allclose(np.asarray(all_first)[:, :2], np.asarray(all_first)[0, :2][None].repeat(2, 0), rtol=1e-5, atol=1e-6)
            decisions = [select(all_first[a], all_terminal[a], current) for a in range(2)]
            arrays[f"r{root}_first_scores"] = np.asarray(all_first)
            arrays[f"r{root}_terminal_scores"] = np.asarray(all_terminal)
            arrays[f"r{root}_lookahead_tails"] = np.asarray(all_tails)
            print(f"root {root}: raw={decisions[0][:2]}, spectral={decisions[1][:2]}", flush=True)
            branches = []
            fixed = [("zero", shared[0]), ("greedy", shared[1]),
                     ("raw_winner", banks[0, decisions[0][0]]),
                     ("spectral_winner", banks[1, decisions[1][0]])]
            for label, prefix in fixed:
                with h.NativeRGBBridge(initial) as env:
                    history = [env.initial_rgb]
                    for action in np.concatenate((np.zeros((2, 2), np.float32), prefix)):
                        history.append(env.step(action[None]))
                    if not np.array_equal(np.stack(history[:3]), saved_history[:3]):
                        raise RuntimeError("prelude RGB mismatch")
                    state7 = h.causal_state(history[6], history[7], metadata, offset, .05)
                    feedback = h.encode_with_state(model, history[7], state7, device)
                    saved_batches, n_scores = [], 0

                    def evaluate_feedback(candidates_tensor):
                        nonlocal n_scores
                        n = candidates_tensor.shape[1]
                        tail = candidates_tensor[0, :, 0].reshape(n, 2, 10).permute(1, 0, 2)
                        with torch.no_grad():
                            pred = model.unroll(feedback, act_suffix=tail)["proprio"]
                            if pred.shape[:2] != (3, n):
                                raise ValueError("feedback clock mismatch")
                            scores = h.projected_cost(pred[2].reshape(n, 256, 16), target, pinv, pos_std)
                        n_scores += n
                        saved_batches.append((candidates_tensor.detach().cpu().numpy(), scores.cpu().numpy()))
                        return scores.unsqueeze(0)

                    rng = torch.Generator(device=device).manual_seed(2026110380 + root)
                    chosen, score0, score1, refined = h.packet_cem(evaluate_feedback, prior, lower, upper, rng)
                    if n_scores != 50 or len(saved_batches) != 4:
                        raise RuntimeError("feedback budget mismatch")
                    tail = np.clip((chosen.reshape(10, 2) * std + mean).cpu().numpy(), -1, 1).astype(np.float32)
                    for action in tail:
                        history.append(env.step(action[None]))
                    if env.step_count != 17:
                        raise RuntimeError("physical clock mismatch")
                actions = np.concatenate((np.zeros((2, 2), np.float32), prefix, tail))
                arrays[f"r{root}_{label}_actions"] = actions
                arrays[f"r{root}_{label}_history"] = np.stack(history)
                arrays[f"r{root}_{label}_normalized_tail"] = chosen.cpu().numpy()
                for j, (c, s) in enumerate(saved_batches):
                    arrays[f"r{root}_{label}_feedback{j}_candidates"] = c
                    arrays[f"r{root}_{label}_feedback{j}_scores"] = s
                starts.append(initial)
                sequences.append(actions)
                branches.append({"name": label, "cp7_rgb_state": state7.tolist(),
                                 "feedback_prior_score": score0, "feedback_refined_score": score1,
                                 "feedback_refined": refined})
            rows.append({"root": root, "current_score": current, "source_greedy": source_greedy,
                         "singular_values": singular.tolist(),
                         "decisions": [{"arm": name, "winner": decision[0],
                                        "qualified_nonprogress": decision[1], "eligible": decision[2]}
                                       for name, decision in zip(("raw", "spectral"), decisions)],
                         "branches": branches})
        torch.cuda.synchronize()
        after = h.parameter_hash(model)
        if before != after:
            raise ValueError("model parameters changed")
    starts, sequences = np.asarray(starts), np.asarray(sequences)
    cp2 = h.physical.replay_position_endpoints(starts, sequences[:, :2])
    cp7 = h.physical.replay_position_endpoints(starts, sequences[:, :7])
    cp17 = h.physical.replay_position_endpoints(starts, sequences)
    arrays["starts"], arrays["sequences"] = starts, sequences
    arrays["cp2_eval_only"], arrays["cp7_eval_only"], arrays["cp17_eval_only"] = cp2, cp7, cp17
    for k, row in enumerate(rows):
        for i, branch in enumerate(row["branches"]):
            for label, position in (("cp2", cp2[4*k+i]), ("cp7", cp7[4*k+i]), ("cp17", cp17[4*k+i])):
                branch[f"{label}_distance"] = float(np.linalg.norm(position[:2] - h.GOAL[:2]))
            branch["physical_nonprogress_cp7"] = branch["cp7_distance"] >= branch["cp2_distance"]
    with ARRAYS.open("xb") as out:
        np.savez_compressed(out, **arrays)
    report = {"schema": "pc_causal_mixed_prefix_search.v1", "rows": rows,
              "scope": "exposed development, no independent or total-cost admission",
              "source_sha256": h.SOURCE_SHA, "arrays_sha256": h.sha256_file(ARRAYS),
              "protocol_sha256": h.sha256_file(PROTOCOL), "script_sha256": h.sha256_file(Path(__file__)),
              "cem_kernel_sha256": h.sha256_file(Path(h.packet_cem.__code__.co_filename)),
              "checkpoint_sha256": h.sha256_file(h.DEFAULT_CHECKPOINT), "model_before": before, "model_after": after,
              "counts": {"goal_encodes": 1, "initial_encodes": 4, "feedback_encodes": 16,
                         "lookahead_three_packet_samples": 2400, "feedback_two_packet_samples": 800,
                         "branch_physical_steps": 272, "position_backend_replay_steps": 416},
              "inherited_upstream_cost": {"jvp_calls": 40, "finite_difference_samples": 80},
              "wall_seconds": time.monotonic() - begun}
    with OUTPUT.open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(f"Saved {OUTPUT} sha256={h.sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
