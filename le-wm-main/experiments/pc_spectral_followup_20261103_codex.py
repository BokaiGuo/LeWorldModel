"""Sealed candidate follow-up with real RGB cp5 replanning."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_spectral_bank_rgb_dev_20261103_codex as base
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history

old, h, ROOT = base.old, base.h, base.ROOT
PROTOCOL = ROOT.parent / "docs/PC_SPECTRAL_CANDIDATE_FOLLOWUP_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT / "results/pc_spectral_candidate_followup_20261103_codex"
DECISIONS = PREFIX.with_name(PREFIX.name+"_decisions.json")
CONTEXT = PREFIX.with_name(PREFIX.name+"_context.npz")
RECEIPT = PREFIX.with_name(PREFIX.name+"_execution.json")
IMAGES = PREFIX.with_name(PREFIX.name+"_execution.npz")
FOLLOWUP = {0: ("spectral_0_+1",), 1: ("direction_2", "direction_4", "random_1")}


def roster(row):
    programs = {item["name"]: item for item in row["plan"]["programs"]}
    roles = [(role, row["arms"][role]) for role in ("selected", "greedy", "zero", "full_horizon_cem")]
    roles.extend(("candidate_"+name, programs[name]) for name in FOLLOWUP.get(row["root"], ()))
    return roles


def main():
    if any(path.exists() for path in (DECISIONS, CONTEXT, RECEIPT, IMAGES)):
        raise FileExistsError("follow-up output already exists")
    sha = old.prior.sha
    source, execution = (json.loads(p.read_text()) for p in (base.OUTPUT, base.EXECUTION))
    if (source["runner_sha256"] != sha(Path(base.__file__))
            or source["protocol_sha256"] != sha(base.PROTOCOL)
            or execution["decision_sha256"] != sha(base.OUTPUT)
            or execution["arrays_sha256"] != sha(base.IMAGES)):
        raise ValueError("source evidence mismatch")
    camera, calibration = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    old_source = json.loads(old.DECISIONS.read_text())
    rows, arrays = [], {}
    with np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT/"external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        if h.parameter_hash(model) != source["model_hash"] or sha(h.DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]:
            raise ValueError("checkpoint mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not np.array_equal(mean.cpu().numpy(), old_source["action_mean"]) or not np.array_equal(std.cpu().numpy(), old_source["action_std"]):
            raise ValueError("normalization mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal_rgb = obstacle["goal_rgb"]
        if old.digest_image(goal_rgb) != old_source["goal_sha256"]:
            raise ValueError("goal mismatch")
        goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
        with torch.no_grad():
            target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        for source_row in source["rows"]:
            root = source_row["root"]
            for role, program in roster(source_row):
                prefix = program["prefix"]
                with ShadowRGBBridge(np.asarray([*old.STARTS[root], 0., 0.])) as env:
                    frames = [env.initial_rgb.copy()]
                    warm, warm_times = old.do_actions(env, old.WARMUP)
                    frames += warm
                    if [old.digest_image(x) for x in frames] != old_source["rows"][root]["rgb_sha256"]:
                        raise ValueError("warmup RGB mismatch")
                    part, prefix_times = old.do_actions(env, prefix)
                    frames += part
                arrays[f"root{root}_{role}"] = np.stack(frames)
                start = time.monotonic()
                with torch.no_grad():
                    contexts = [h.encode_with_state(model, frames[t], h.causal_state(frames[t-1], frames[t], metadata, offset, .05), device)
                                for t in (10, 15, 20)]
                    torch.cuda.synchronize()
                encode_seconds = time.monotonic()-start
                ctx = {key: torch.cat([x[key] for x in contexts], dim=1) for key in ("visual", "proprio")}
                raw_history = np.concatenate((old.WARMUP[10:15], prefix))
                history = ((torch.as_tensor(raw_history, dtype=torch.float32, device=device)-mean)/std).reshape(2, 1, 10)
                lower, upper = (-1-mean.repeat(10))/std.repeat(10), (1-mean.repeat(10))/std.repeat(10)
                zero = -mean.repeat(10)/std.repeat(10)
                noise = torch.randn((2, 3, 16, 20), generator=torch.Generator().manual_seed(2026110360+root)).to(device)
                records, counts = [], {"batches": 0, "samples": 0}
                def predict(samples):
                    batch = len(samples)
                    counts["batches"] += 1
                    counts["samples"] += batch
                    future = samples.reshape(batch, 2, 10).transpose(0, 1)
                    with torch.no_grad():
                        prediction = unroll_with_action_history(model, ctx, history.expand(2, batch, 10), future, 3)["proprio"][-1]
                        result = h.projected_cost(prediction.reshape(batch, 256, 16), target, pinv, pos_std)
                        torch.cuda.synchronize()
                        return result
                start = time.monotonic()
                for restart in range(2):
                    loc, scale = zero.clone(), torch.full_like(zero, 1/3)
                    for generation in range(3):
                        samples = (loc+noise[restart, generation]*scale).clamp(lower, upper)
                        samples[0] = loc.clamp(lower, upper)
                        scores = predict(samples)
                        records.extend(zip(samples.cpu().tolist(), scores.cpu().tolist()))
                        elite = torch.argsort(scores, stable=True)[:4]
                        loc, scale = samples[elite].mean(0), samples[elite].std(0, correction=1).clamp_min(1e-5)
                    samples = torch.stack((zero, loc)).clamp(lower, upper)
                    scores = predict(samples)
                    records.extend(zip(samples.cpu().tolist(), scores.cpu().tolist()))
                index = int(np.argmin([x[1] for x in records]))
                suffix = (torch.as_tensor(records[index][0], device=device)*std.repeat(10)+mean.repeat(10)).reshape(10, 2).cpu().tolist()
                rows.append({"root": root, "role": role, "name": program["name"], "prefix": prefix,
                             "old_suffix": program["suffix"], "old_model_score": program["terminal_score"],
                             "suffix": suffix, "cp5_rgb_sha256": old.digest_image(frames[20]),
                             "history_actions": raw_history.tolist(), "history_normalized": history[:, 0].cpu().tolist(),
                             "evaluations": [{"actions_normalized": a, "score": b} for a, b in records],
                             "best_index": index, "counts": counts, "encoding_calls": 3,
                             "encoding_seconds": encode_seconds, "search_seconds": time.monotonic()-start,
                             "warmup_times_ns": warm_times, "prefix_times_ns": prefix_times})
                print(f"root {root} {role}: planned", flush=True)
        if h.parameter_hash(model) != source["model_hash"]:
            raise ValueError("model mutated")
        with CONTEXT.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        old.write_json(DECISIONS, {"rows": rows, "source_sha256": sha(base.OUTPUT), "context_sha256": sha(CONTEXT),
                                   "runner_sha256": sha(Path(__file__)), "protocol_sha256": sha(PROTOCOL),
                                   "model_hash": source["model_hash"], "full_cost_certified": False})
        print("all plans sealed; executing both frozen and replanned suffixes", flush=True)
        images, results = {}, []
        for row in rows:
            for kind, suffix in (("original", row["old_suffix"]), ("replanned", row["suffix"])):
                root, role = row["root"], row["role"]
                with ShadowRGBBridge(np.asarray([*old.STARTS[root], 0., 0.])) as env:
                    old.do_actions(env, old.WARMUP)
                    prefix_frames, _ = old.do_actions(env, row["prefix"])
                    if old.digest_image(prefix_frames[-1]) != row["cp5_rgb_sha256"]:
                        raise ValueError("cp5 replay mismatch")
                    frames, times = old.do_actions(env, suffix)
                last_state = h.causal_state(frames[-2], frames[-1], metadata, offset, .05)
                with torch.no_grad():
                    encoded = h.encode_with_state(model, frames[-1], last_state, device)["proprio"]
                    true_rgb_score = float(h.projected_cost(encoded.reshape(1, 256, 16), target, pinv, pos_std)[0])
                key = f"root{root}_{role}_{kind}"
                images[key] = np.stack(frames)
                results.append({"root": root, "role": role, "kind": kind, "prefix": row["prefix"],
                                "suffix": suffix, "rgb_sha256": [old.digest_image(x) for x in frames],
                                "rgb_terminal_score": true_rgb_score, "times_ns": times})
        with IMAGES.open("xb") as handle:
            np.savez_compressed(handle, **images)
        old.write_json(RECEIPT, {"rows": results, "decision_sha256": sha(DECISIONS), "arrays_sha256": sha(IMAGES)})
        print(f"executed {len(results)} programs", flush=True)


if __name__ == "__main__":
    main()
