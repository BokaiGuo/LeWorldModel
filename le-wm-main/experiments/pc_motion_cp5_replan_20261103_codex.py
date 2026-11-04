"""Real cp5 RGB same-budget suffix optimization for three fixed roles."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_motion_unprivileged_dev_20261103_codex as prior
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history

h, ROOT = prior.h, prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_MOTION_CP5_REPLAN_PROTOCOL_20261103_CODEX.md"
OUT = ROOT / "results/pc_motion_cp5_replan_20261103_codex"
CONTEXT = OUT.with_name(OUT.name + "_context.npz")
DECISION = OUT.with_name(OUT.name + "_decisions.json")
EXECUTION = OUT.with_name(OUT.name + "_execution.npz")
RECEIPT = OUT.with_name(OUT.name + "_execution.json")


def main():
    if any(path.exists() for path in (CONTEXT, DECISION, EXECUTION, RECEIPT)):
        raise FileExistsError("partial or complete cp5 run")
    source = json.loads(prior.DECISIONS.read_text())
    if source["runner_sha256"] != prior.prior.sha(Path(prior.__file__)):
        raise ValueError("source runner mismatch")
    camera, calibration = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    arrays, rows = {}, []
    with np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        before = h.parameter_hash(model)
        if before != source["model_hash"]:
            raise ValueError("frozen model mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not np.allclose(mean.cpu(), source["action_mean"], atol=0, rtol=0) or not np.allclose(std.cpu(), source["action_std"], atol=0, rtol=0):
            raise ValueError("action normalization mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal_rgb = obstacle["goal_rgb"]
        goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
        with torch.no_grad():
            target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        for root, source_row in enumerate(source["rows"]):
            names = (source_row["choice"]["selected_name"], "zero", "random_0")
            for role, name in zip(("selected", "zero", "random_0"), names, strict=True):
                prefix = next(c["program"]["prefix"] for c in source_row["candidates"]
                              if c["program"]["name"] == name)
                with ShadowRGBBridge(np.asarray([*prior.STARTS[root], 0., 0.])) as env:
                    frames = [env.initial_rgb.copy()]
                    warm, warm_time = prior.do_actions(env, prior.WARMUP)
                    frames += warm
                    if [prior.digest_image(frame) for frame in frames] != source_row["rgb_sha256"]:
                        raise RuntimeError("warmup differs")
                    following, prefix_time = prior.do_actions(env, prefix)
                    frames += following
                key = f"root{root}_{role}"
                arrays[key] = np.stack(frames)
                contexts = []
                with torch.no_grad():
                    for t in (10, 15, 20):
                        state = h.causal_state(frames[t-1], frames[t], metadata, offset, .05)
                        contexts.append(h.encode_with_state(model, frames[t], state, device))
                ctx = {k: torch.cat([entry[k] for entry in contexts], dim=1) for k in ("visual", "proprio")}
                raw_history = np.concatenate((prior.WARMUP[10:15], np.asarray(prefix)), axis=0)
                hist = ((torch.as_tensor(raw_history, device=device, dtype=torch.float32)-mean)/std).reshape(2, 1, 10)
                lower, upper = (-1-mean.repeat(10))/std.repeat(10), (1-mean.repeat(10))/std.repeat(10)
                zero = -mean.repeat(10)/std.repeat(10)
                noise = torch.randn((2, 3, 16, 20), generator=torch.Generator().manual_seed(2026110360+root)).to(device)
                evaluations, start = [], time.monotonic()

                def predict(samples):
                    batch = len(samples)
                    future = samples.reshape(batch, 2, 10).transpose(0, 1)
                    with torch.no_grad():
                        pred = unroll_with_action_history(model, ctx, hist.expand(2, batch, 10),
                                                          future, 3)["proprio"][-1]
                        return h.projected_cost(pred.reshape(batch, 256, 16), target, pinv, pos_std)

                for restart in range(2):
                    loc, scale = zero.clone(), torch.full_like(zero, 1/3)
                    for generation in range(3):
                        samples = (loc+noise[restart, generation]*scale).clamp(lower, upper)
                        samples[0] = loc.clamp(lower, upper)
                        scores = predict(samples)
                        evaluations.extend(zip(samples.cpu().tolist(), scores.cpu().tolist()))
                        elite = torch.argsort(scores, stable=True)[:4]
                        loc = samples[elite].mean(0)
                        scale = samples[elite].std(0, correction=1).clamp_min(1e-5)
                    final = torch.stack((zero, loc)).clamp(lower, upper)
                    scores = predict(final)
                    evaluations.extend(zip(final.cpu().tolist(), scores.cpu().tolist()))
                index = int(np.argmin([v for _, v in evaluations]))
                suffix_norm = torch.as_tensor(evaluations[index][0], device=device)
                suffix = (suffix_norm*std.repeat(10)+mean.repeat(10)).reshape(10, 2).cpu().tolist()
                rows.append({"root_index": root, "role": role, "prefix_name": name, "prefix": prefix,
                             "cp5_rgb_sha256": prior.digest_image(frames[20]),
                             "history_actions_raw": raw_history.tolist(), "history_normalized": hist[:, 0].cpu().tolist(),
                             "suffix": suffix, "best_index": index, "model_score": evaluations[index][1],
                             "evaluations": [{"suffix_normalized": a, "score": b} for a, b in evaluations],
                             "warmup_times_ns": warm_time, "prefix_times_ns": prefix_time,
                             "planning_wall_seconds": time.monotonic()-start})
                print(f"root {root} role {role}: suffix planned", flush=True)
        if h.parameter_hash(model) != before:
            raise ValueError("model weights changed")
    with CONTEXT.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    prior.write_json(DECISION, {"schema": "pc_motion_cp5_replan.v1", "rows": rows,
                                "source_sha256": prior.prior.sha(prior.DECISIONS),
                                "context_sha256": prior.prior.sha(CONTEXT),
                                "protocol_sha256": prior.prior.sha(PROTOCOL),
                                "runner_sha256": prior.prior.sha(Path(__file__)), "model_hash": before})
    print(f"all cp5 plans sealed {prior.prior.sha(DECISION)}", flush=True)
    outputs, execution = {}, []
    for row in rows:
        root, role = row["root_index"], row["role"]
        with ShadowRGBBridge(np.asarray([*prior.STARTS[root], 0., 0.])) as env:
            frames, warm_time = prior.do_actions(env, prior.WARMUP)
            prefix_frames, prefix_time = prior.do_actions(env, row["prefix"])
            if prior.digest_image(prefix_frames[-1]) != row["cp5_rgb_sha256"]:
                raise RuntimeError("cp5 replay differs")
            final_frames, final_time = prior.do_actions(env, row["suffix"])
        outputs[f"root{root}_{role}"] = np.stack(final_frames)
        execution.append({"root_index": root, "role": role, "prefix": row["prefix"], "suffix": row["suffix"],
                          "suffix_times_ns": final_time,
                          "suffix_rgb_sha256": [prior.digest_image(frame) for frame in final_frames]})
    with EXECUTION.open("xb") as handle:
        np.savez_compressed(handle, **outputs)
    prior.write_json(RECEIPT, {"schema": "pc_motion_cp5_replan_execution.v1", "rows": execution,
                               "decision_sha256": prior.prior.sha(DECISION),
                               "arrays_sha256": prior.prior.sha(EXECUTION)})
    print("all cp5 suffixes physically dispatched", flush=True)


if __name__ == "__main__":
    main()
