"""Joint nonlinear prefix/suffix search with strict post-search eligibility."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import pc_causal_history_h10_suffix_20261103 as h

SOURCE = h.ROOT / "results/pc_causal_mixed_prefix_search_20261103_DEV01.json"
SOURCE_SHA = "f9d8f3014b44a95dfd14b0afd3d49c0b2d4b776a8112d9fa20370cb7f52fe368"
PROTOCOL = h.ROOT.parent / "docs/PC_CAUSAL_JOINT_PREFIX_GRADIENT_PROTOCOL_20261103.md"
OUTPUT = h.ROOT / "results/pc_causal_joint_prefix_gradient_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def optimize(initial, score, current, *, steps=12, lr=.05, penalty=10.):
    """Projected Adam with fixed control prefixes and complete iterate trace."""
    if initial.ndim != 3 or initial.shape[1:] != (15, 2) or initial.shape[0] < 3:
        raise ValueError("invalid action initialization")
    if not bool(torch.isfinite(initial).all()) or bool((initial.abs() > 1).any()):
        raise ValueError("invalid bounded actions")
    if steps < 1 or not np.isfinite(current):
        raise ValueError("invalid search parameters")
    action = initial.detach().clone()
    fixed = action[:2, :5].clone()
    m, v = torch.zeros_like(action), torch.zeros_like(action)
    plans, firsts, ends, grads = [], [], [], []
    for t in range(steps + 1):
        action = action.detach().requires_grad_(t < steps)
        first, end = score(action)
        if first.shape != (len(action),) or end.shape != first.shape:
            raise ValueError("invalid model score shape")
        if not bool(torch.isfinite(first).all() and torch.isfinite(end).all()):
            raise ValueError("invalid model score")
        plans.append(action.detach().cpu().numpy().copy())
        firsts.append(first.detach().cpu().numpy().copy())
        ends.append(end.detach().cpu().numpy().copy())
        if t == steps:
            break
        loss = end + penalty * torch.relu(current - first).square()
        gradient, = torch.autograd.grad(loss.sum(), action)
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("nonfinite action gradient")
        gradient = gradient.detach()
        gradient[:2, :5] = 0
        grads.append(gradient.cpu().numpy().copy())
        with torch.no_grad():
            m = .9*m + .1*gradient
            v = .999*v + .001*gradient.square()
            update = (m / (1-.9**(t+1))) / (torch.sqrt(v / (1-.999**(t+1))) + 1e-8)
            action = torch.clamp(action - lr*update, -1., 1.)
            action[:2, :5] = fixed
    return np.asarray(plans), np.asarray(firsts), np.asarray(ends), np.asarray(grads)


def decide(first, terminal, current):
    if (first.shape != terminal.shape or first.ndim != 2 or first.shape[1] != 10
            or not np.isfinite(first).all() or not np.isfinite(terminal).all()):
        raise ValueError("invalid search trace")
    controls = [int(np.argmin(terminal[:, i])) for i in (0, 1)]
    best_controls = [float(terminal[t, i]) for i, t in enumerate(controls)]
    result = []
    for indices in (range(2, 6), range(6, 10)):
        eligible = [(float(terminal[t, i]), i, t)
                    for i in indices for t in range(len(first)) if first[t, i] >= current]
        qualified = min(eligible) if eligible else None
        if qualified is not None and not all(qualified[0] < x for x in best_controls):
            qualified = None
        pool = [(float(terminal[t, i]), i, t) for i in indices for t in range(len(first))]
        pool += [(float(terminal[t, i]), i, t) for i, t in enumerate(controls)]
        result.append({"unconditional": list(min(pool)[1:]),
                       "qualified": list(qualified[1:]) if qualified is not None else None,
                       "eligible_iterates": len(eligible)})
    return controls, result


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (h.SOURCE, h.SOURCE_SHA),
                           (h.OBSTACLE, h.OBSTACLE_SHA), (h.CAMERA, h.CAMERA_SHA),
                           (h.OLD, h.OLD_SHA),
                           (h.DEFAULT_DATASET / "seq_lengths.pth", h.LENGTHS_SHA)):
        if h.sha256_file(path) != expected:
            raise ValueError(f"source changed: {path}")
    source, oldsource, obstacle, camera, old = (json.loads(p.read_text())
                                               for p in (SOURCE, h.SOURCE, h.OBSTACLE, h.CAMERA, h.OLD))
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (h.SOURCE.with_suffix(".npz"), oldsource["arrays_sha256"]),
                           (h.OBSTACLE.with_suffix(".npz"), obstacle["arrays_sha256"]),
                           (h.DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if h.sha256_file(path) != expected:
            raise ValueError(f"source changed: {path}")
    if [r["root"] for r in source["rows"]] != list(h.ROOTS):
        raise ValueError("invalid root roster")
    args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                              config=h.DEFAULT_CONFIG, dinov2_source=h.ROOT / "external/local_dinov2")
    arrays, rows, starts, sequences = {}, [], [], []
    begun = time.monotonic()
    print("Jointly optimize nonprogress prefixes and suffixes, then test selected plans in real feedback.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as bank, \
            np.load(h.SOURCE.with_suffix(".npz"), allow_pickle=False) as prior, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle_arrays, \
            h.lock.gpu_lock(h.ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        model, dataset = h.load_model_and_dataset(args, device)
        before = h.parameter_hash(model)
        if before != source["model_before"]:
            raise ValueError("model hash mismatch")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), h.MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), h.STD, rtol=0, atol=1e-7)):
            raise ValueError("proprio normalizer mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("invalid action normalizer")
        lower, upper = ((-torch.ones_like(mean)-mean)/std).repeat(10), ((torch.ones_like(mean)-mean)/std).repeat(10)
        cem_prior = ((torch.zeros_like(mean)-mean)/std).repeat(10).reshape(1, 1, 20)
        weight, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        goal_rgb = obstacle_arrays["goal_rgb"]
        goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
        target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        for k, root in enumerate(h.ROOTS):
            initial = bank["starts"][4*k].astype(np.float64)
            initial_frames = prior[f"r{root}_p0_history"]
            state2 = h.causal_state(initial_frames[1], initial_frames[2], metadata, offset, .05)
            context = h.encode_with_state(model, initial_frames[2], state2, device)
            current = float(h.projected_cost(context["proprio"].reshape(1, 256, 16), target, pinv, pos_std).item())
            np.testing.assert_allclose(current, source["rows"][k]["current_score"], rtol=1e-5, atol=1e-6)
            banks = bank[f"r{root}_banks"]
            prefixes = np.concatenate((banks[0, :2], banks[0, 2:], banks[1, 2:]))
            initial_plans = np.concatenate((prefixes, np.zeros((10, 10, 2), np.float32)), axis=1)

            def score(action):
                n = len(action)
                packets = ((action - mean) / std).reshape(n, 3, 10).permute(1, 0, 2)
                pred = model.unroll(context, act_suffix=packets)["proprio"]
                if pred.shape[:2] != (4, n):
                    raise ValueError("joint planning clock mismatch")
                return (h.projected_cost(pred[1].reshape(n, 256, 16), target, pinv, pos_std),
                        h.projected_cost(pred[3].reshape(n, 256, 16), target, pinv, pos_std))

            plans, first, end, gradients = optimize(torch.as_tensor(initial_plans, device=device), score, current)
            controls, decisions = decide(first, end, current)
            arrays[f"r{root}_plans"], arrays[f"r{root}_first"], arrays[f"r{root}_end"] = plans, first, end
            arrays[f"r{root}_gradients"] = gradients
            fixed = [("zero", [0, controls[0]]), ("greedy", [1, controls[1]]),
                     ("raw_winner", decisions[0]["unconditional"]), ("spectral_winner", decisions[1]["unconditional"])]
            for arm, decision in zip(("raw", "spectral"), decisions):
                if decision["qualified"] is not None and decision["qualified"] not in [x[1] for x in fixed]:
                    fixed.append((f"{arm}_qualified", decision["qualified"]))
            print(f"root {root}: {decisions}", flush=True)
            branches = []
            for label, (candidate, iteration) in fixed:
                prefix = plans[iteration, candidate, :5]
                with h.NativeRGBBridge(initial) as env:
                    history = [env.initial_rgb]
                    for action in np.concatenate((np.zeros((2, 2), np.float32), prefix)):
                        history.append(env.step(action[None]))
                    if not np.array_equal(np.stack(history[:3]), initial_frames[:3]):
                        raise RuntimeError("prelude mismatch")
                    state7 = h.causal_state(history[6], history[7], metadata, offset, .05)
                    feedback = h.encode_with_state(model, history[7], state7, device)
                    batches, count = [], 0

                    def evaluate(proposals):
                        nonlocal count
                        n = proposals.shape[1]
                        tail = proposals[0, :, 0].reshape(n, 2, 10).permute(1, 0, 2)
                        with torch.no_grad():
                            pred = model.unroll(feedback, act_suffix=tail)["proprio"]
                            if pred.shape[:2] != (3, n):
                                raise ValueError("feedback clock mismatch")
                            scores = h.projected_cost(pred[2].reshape(n, 256, 16), target, pinv, pos_std)
                        count += n
                        batches.append((proposals.detach().cpu().numpy(), scores.cpu().numpy()))
                        return scores.unsqueeze(0)

                    rng = torch.Generator(device=device).manual_seed(2026110380 + root)
                    chosen, prior_score, refined_score, refined = h.packet_cem(
                        evaluate, cem_prior, lower, upper, rng)
                    if count != 50 or len(batches) != 4:
                        raise RuntimeError("feedback CEM budget mismatch")
                    tail = np.clip((chosen.reshape(10, 2)*std+mean).cpu().numpy(), -1, 1).astype(np.float32)
                    for action in tail:
                        history.append(env.step(action[None]))
                    if env.step_count != 17:
                        raise RuntimeError("physical clock mismatch")
                actions = np.concatenate((np.zeros((2, 2), np.float32), prefix, tail))
                arrays[f"r{root}_{label}_actions"], arrays[f"r{root}_{label}_history"] = actions, np.stack(history)
                for j, (p, s) in enumerate(batches):
                    arrays[f"r{root}_{label}_feedback{j}_candidates"] = p
                    arrays[f"r{root}_{label}_feedback{j}_scores"] = s
                starts.append(initial)
                sequences.append(actions)
                branches.append({"name": label, "candidate": candidate, "iteration": iteration,
                                 "predicted_prefix_score": float(first[iteration, candidate]),
                                 "predicted_terminal_score": float(end[iteration, candidate]),
                                 "feedback_prior_score": prior_score, "feedback_refined_score": refined_score,
                                 "feedback_refined": refined})
            rows.append({"root": root, "current_score": current, "control_iterations": controls,
                         "decisions": decisions, "branches": branches})
        torch.cuda.synchronize()
        after = h.parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    starts, sequences = np.asarray(starts), np.asarray(sequences)
    cp2 = h.physical.replay_position_endpoints(starts, sequences[:, :2])
    cp7 = h.physical.replay_position_endpoints(starts, sequences[:, :7])
    cp17 = h.physical.replay_position_endpoints(starts, sequences)
    arrays["starts"], arrays["sequences"] = starts, sequences
    arrays["cp2_eval_only"], arrays["cp7_eval_only"], arrays["cp17_eval_only"] = cp2, cp7, cp17
    j = 0
    for row in rows:
        for branch in row["branches"]:
            for tag, pos in (("cp2", cp2[j]), ("cp7", cp7[j]), ("cp17", cp17[j])):
                branch[f"{tag}_distance"] = float(np.linalg.norm(pos[:2] - h.GOAL[:2]))
            branch["physical_nonprogress_cp7"] = branch["cp7_distance"] >= branch["cp2_distance"]
            j += 1
    with ARRAYS.open("xb") as out:
        np.savez_compressed(out, **arrays)
    report = {"schema": "pc_causal_joint_prefix_gradient.v1", "rows": rows,
              "scope": "exposed development; no common-suffix, independent or fair-total-compute admission",
              "source_sha256": SOURCE_SHA, "arrays_sha256": h.sha256_file(ARRAYS),
              "protocol_sha256": h.sha256_file(PROTOCOL), "script_sha256": h.sha256_file(Path(__file__)),
              "cem_kernel_sha256": h.sha256_file(Path(h.packet_cem.__code__.co_filename)),
              "checkpoint_sha256": h.sha256_file(h.DEFAULT_CHECKPOINT), "model_before": before, "model_after": after,
              "counts": {"goal_encodes": 1, "initial_encodes": 4, "feedback_encodes": len(sequences),
                         "joint_three_packet_samples": 4*13*10, "batched_action_backward_calls": 4*12,
                         "feedback_two_packet_samples": 50*len(sequences),
                         "rgb_backend_steps": 17*len(sequences), "position_backend_replay_steps": 26*len(sequences)},
              "inherited_upstream_cost": source["counts"], "wall_seconds": time.monotonic() - begun}
    with OUTPUT.open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(f"Saved {OUTPUT} sha256={h.sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
