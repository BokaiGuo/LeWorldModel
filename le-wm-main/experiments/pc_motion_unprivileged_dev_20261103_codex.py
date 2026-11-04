"""Real moving RGB history -> generated bank -> sealed plans -> RGB execution."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_pointmaze_cp12_rgb_history_planning_20261103_codex as prior
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history
from experiments.spectral_reachability import pc_rgb_candidate_policy_20261103_codex as policy

h, ROOT = prior.h, prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_MOTION_HISTORY_UNPRIVILEGED_DEV_PROTOCOL_20261103_CODEX.md"
OUT = ROOT / "results/pc_motion_unprivileged_dev_20261103_codex"
HISTORY = OUT.with_name(OUT.name + "_history.npz")
DECISIONS = OUT.with_name(OUT.name + "_decisions.json")
EXECUTION = OUT.with_name(OUT.name + "_execution.npz")
RECEIPT = OUT.with_name(OUT.name + "_execution.json")
STARTS = ((.8, .8), (.8, 1.8), (2.8, .8), (2.8, 1.8))
WARMUP = np.asarray([[.2, 0.]]*5 + [[0., .2]]*5 + [[-.2, 0.]]*5, dtype=np.float64)


def digest_image(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def do_actions(env, actions):
    frames, times = [], []
    for action in actions:
        before = time.monotonic_ns()
        frame = env.step(np.asarray(action, dtype=np.float64).reshape(1, 2)).copy()
        after = time.monotonic_ns()
        frames.append(frame)
        times.append([before, after])
    return frames, times


def write_json(path, doc):
    with path.open("x") as handle:
        json.dump(doc, handle, allow_nan=False)
        handle.write("\n")


def main():
    if any(p.exists() for p in (HISTORY, DECISIONS, EXECUTION, RECEIPT)):
        raise FileExistsError("partial or complete run exists; do not overwrite")
    camera, calibration = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    arrays, timing, rows = {}, [], []
    with np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        before = h.parameter_hash(model)
        if prior.sha(h.DEFAULT_CHECKPOINT) != prior.EXPECTED[h.DEFAULT_CHECKPOINT]:
            raise ValueError("checkpoint drift")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("invalid action normalization")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal_rgb = obstacle["goal_rgb"]
        goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
        with torch.no_grad():
            target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        for root, start in enumerate(STARTS):
            with ShadowRGBBridge(np.asarray([*start, 0., 0.])) as env:
                initial = env.initial_rgb.copy()
                frames, times = do_actions(env, WARMUP)
            frames = [initial] + frames
            arrays[f"root{root}_history_rgb"] = np.stack(frames)
            timing.append(times)
            contexts = []
            with torch.no_grad():
                for t in (5, 10, 15):
                    state = h.causal_state(frames[t-1], frames[t], metadata, offset, .05)
                    contexts.append(h.encode_with_state(model, frames[t], state, device))
            ctx = {key: torch.cat([c[key] for c in contexts], dim=1) for key in ("visual", "proprio")}
            current_ctx = {key: value[:, -1:] for key, value in ctx.items()}
            history_raw = torch.as_tensor(WARMUP[5:], dtype=torch.float32, device=device)
            history = ((history_raw-mean)/std).reshape(2, 1, 10)
            with torch.no_grad():
                initial_score = float(h.projected_cost(ctx["proprio"][:, -1:].reshape(1, 256, 16),
                                                       target, pinv, pos_std)[0])
            bank = policy.build_fixed_bank(2026110321 + root)
            names, scores, candidates = [], [], []
            lower, upper = (-1-mean.repeat(10))/std.repeat(10), (1-mean.repeat(10))/std.repeat(10)
            zero = -mean.repeat(10)/std.repeat(10)
            noise = torch.randn((2, 3, 16, 20),
                                generator=torch.Generator().manual_seed(2026110340 + root)).to(device)

            def predict(packets, history_mode=True):
                batch = packets.shape[1]
                hist = history.expand(2, batch, 10) if history_mode else packets[:0]
                with torch.no_grad():
                    prediction = unroll_with_action_history(
                        model, ctx if history_mode else current_ctx, hist, packets,
                        3 if history_mode else 1)["proprio"][-1]
                    return h.projected_cost(prediction.reshape(batch, 256, 16), target, pinv, pos_std)

            start_time = time.monotonic()
            for item in bank:
                prefix = torch.as_tensor(item["prefix"], dtype=torch.float32, device=device)
                prefix = ((prefix-mean)/std).reshape(10)
                first = float(predict(prefix.reshape(1, 1, 10))[0])
                evaluations = []
                for restart in range(2):
                    loc, scale = zero.clone(), torch.full_like(zero, 1/3)
                    for generation in range(3):
                        samples = (loc + noise[restart, generation]*scale).clamp(lower, upper)
                        samples[0] = loc.clamp(lower, upper)
                        packets = torch.cat((prefix.expand(16, 10), samples), dim=1).reshape(16, 3, 10).transpose(0, 1)
                        values = predict(packets)
                        evaluations.extend(zip(samples.cpu().tolist(), values.cpu().tolist()))
                        elite = torch.argsort(values, stable=True)[:4]
                        loc = samples[elite].mean(0)
                        scale = samples[elite].std(0, correction=1).clamp_min(1e-5)
                    final = torch.stack((zero, loc)).clamp(lower, upper)
                    packets = torch.cat((prefix.expand(2, 10), final), dim=1).reshape(2, 3, 10).transpose(0, 1)
                    values = predict(packets)
                    evaluations.extend(zip(final.cpu().tolist(), values.cpu().tolist()))
                best = int(np.argmin([value for _, value in evaluations]))
                suffix_norm = torch.as_tensor(evaluations[best][0], device=device)
                suffix = (suffix_norm*std.repeat(10)+mean.repeat(10)).reshape(10, 2).cpu().numpy()
                chain = torch.cat((prefix, suffix_norm)).reshape(3, 1, 10)
                single_score = float(predict(chain, False)[0])
                candidate = {"name": item["name"], "source": item["source"], "prefix": item["prefix"],
                             "prefix_score": first, "terminal_score": evaluations[best][1],
                             "suffix": suffix.tolist()}
                names.append(item["name"])
                scores.append(single_score)
                candidates.append({"program": candidate, "single_frame_score": single_score,
                                   "evaluations": [{"suffix_normalized": action, "score": value}
                                                   for action, value in evaluations]})
            choice = policy.select_candidates([c["program"] for c in candidates], initial_score)
            rows.append({"root_index": root, "candidate_seed": 2026110321 + root,
                         "cem_seed": 2026110340 + root, "warmup": WARMUP.tolist(),
                         "history_action_packets_raw": WARMUP[5:].reshape(2, 5, 2).tolist(),
                         "history_action_packets_normalized": history[:, 0].cpu().tolist(),
                         "rgb_sha256": [digest_image(frame) for frame in frames],
                         "initial_score": initial_score, "candidates": candidates, "choice": choice,
                         "single_frame_selected_index": int(np.argmin(scores)),
                         "prediction_samples": 11*100 + 11 + 11,
                         "planning_wall_seconds": time.monotonic()-start_time})
            print(f"root {root}: selected={choice['selected_name']}", flush=True)
        if h.parameter_hash(model) != before:
            raise ValueError("model weights changed")
        action_mean, action_std = mean.cpu().tolist(), std.cpu().tolist()
    with HISTORY.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    decision = {"schema": "pc_motion_unprivileged_decision.v1", "rows": rows, "timings": timing,
                "history_sha256": prior.sha(HISTORY), "protocol_sha256": prior.sha(PROTOCOL),
                "generator_sha256": prior.sha(Path(policy.__file__)), "runner_sha256": prior.sha(Path(__file__)),
                "checkpoint_sha256": prior.sha(h.DEFAULT_CHECKPOINT), "config_sha256": prior.sha(h.DEFAULT_CONFIG),
                "camera_sha256": prior.sha(h.CAMERA), "calibration_sha256": prior.sha(h.OLD),
                "goal_sha256": digest_image(goal_rgb), "model_hash": before,
                "action_mean": action_mean, "action_std": action_std}
    write_json(DECISIONS, decision)
    print(f"all decisions sealed: {prior.sha(DECISIONS)}", flush=True)
    results, exec_arrays = [], {}
    for root, start in enumerate(STARTS):
        row = rows[root]
        program = row["candidates"][row["choice"]["selected_index"]]["program"]
        with ShadowRGBBridge(np.asarray([*start, 0., 0.])) as env:
            frames = [env.initial_rgb.copy()]
            warm_frames, warm_times = do_actions(env, WARMUP)
            frames += warm_frames
            if [digest_image(x) for x in frames] != row["rgb_sha256"]:
                raise RuntimeError("real RGB warmup replay mismatch")
            actions = program["prefix"] + program["suffix"]
            out_frames, out_times = do_actions(env, actions)
        exec_arrays[f"root{root}_rgb"] = np.stack(out_frames)
        results.append({"root_index": root, "selected_name": program["name"], "actions": actions,
                        "warmup_times_ns": warm_times, "execution_times_ns": out_times,
                        "rgb_sha256": [digest_image(frame) for frame in out_frames]})
        print(f"root {root}: selected chain executed", flush=True)
    with EXECUTION.open("xb") as handle:
        np.savez_compressed(handle, **exec_arrays)
    write_json(RECEIPT, {"schema": "pc_motion_unprivileged_execution.v1", "rows": results,
                         "decision_sha256": prior.sha(DECISIONS), "arrays_sha256": prior.sha(EXECUTION)})


if __name__ == "__main__":
    main()
