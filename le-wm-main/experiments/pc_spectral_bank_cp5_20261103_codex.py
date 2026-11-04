"""Real RGB cp5 replanning of sealed spectral-bank comparison roles."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_spectral_bank_rgb_dev_20261103_codex as prior
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history

old, h, ROOT = prior.old, prior.h, prior.ROOT
BASE = ROOT / "results/pc_spectral_bank_cp5_20261103_codex"
CONTEXT = BASE.with_name(BASE.name+"_context.npz")
DECISION = BASE.with_name(BASE.name+"_decisions.json")
EXECUTION = BASE.with_name(BASE.name+"_execution.json")
IMAGES = BASE.with_name(BASE.name+"_execution.npz")


def main():
    if any(p.exists() for p in (CONTEXT, DECISION, EXECUTION, IMAGES)):
        raise FileExistsError("cp5 output exists")
    sha = old.prior.sha
    source = json.loads(prior.OUTPUT.read_text())
    receipt = json.loads(prior.EXECUTION.read_text())
    if (source["runner_sha256"] != sha(Path(prior.__file__)) or source["protocol_sha256"] != sha(prior.PROTOCOL)
            or receipt["decision_sha256"] != sha(prior.OUTPUT) or receipt["arrays_sha256"] != sha(prior.IMAGES)):
        raise ValueError("sealed source mismatch")
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
        if h.parameter_hash(model) != source["model_hash"]:
            raise ValueError("model mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal_rgb = obstacle["goal_rgb"]
        goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
        with torch.no_grad():
            target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        old_source = json.loads(old.DECISIONS.read_text())
        for source_row in source["rows"]:
            root = source_row["root"]
            for role in ("selected", "greedy", "zero", "random", "no_spectral"):
                prefix = source_row["arms"][role]["prefix"]
                with ShadowRGBBridge(np.asarray([*old.STARTS[root], 0., 0.])) as env:
                    frames = [env.initial_rgb.copy()]
                    warm, warm_times = old.do_actions(env, old.WARMUP)
                    frames += warm
                    if [old.digest_image(x) for x in frames] != old_source["rows"][root]["rgb_sha256"]:
                        raise ValueError("warmup mismatch")
                    tail, prefix_times = old.do_actions(env, prefix)
                    frames += tail
                arrays[f"root{root}_{role}"] = np.stack(frames)
                contexts = []
                started = time.monotonic()
                with torch.no_grad():
                    for t in (10, 15, 20):
                        state = h.causal_state(frames[t-1], frames[t], metadata, offset, .05)
                        contexts.append(h.encode_with_state(model, frames[t], state, device))
                    torch.cuda.synchronize()
                encode_seconds = time.monotonic()-started
                ctx = {k: torch.cat([c[k] for c in contexts], dim=1) for k in ("visual", "proprio")}
                raw_history = np.concatenate((old.WARMUP[10:15], prefix))
                history = ((torch.as_tensor(raw_history, dtype=torch.float32, device=device)-mean)/std).reshape(2, 1, 10)
                lower, upper = (-1-mean.repeat(10))/std.repeat(10), (1-mean.repeat(10))/std.repeat(10)
                zero = -mean.repeat(10)/std.repeat(10)
                noise = torch.randn((2, 3, 16, 20), generator=torch.Generator().manual_seed(2026110360+root)).to(device)
                evaluations = []
                prediction_samples, prediction_calls = 0, 0
                def predict(samples):
                    nonlocal prediction_samples, prediction_calls
                    batch = len(samples)
                    prediction_samples += batch
                    prediction_calls += 1
                    future = samples.reshape(batch, 2, 10).transpose(0, 1)
                    with torch.no_grad():
                        pred = unroll_with_action_history(model, ctx, history.expand(2, batch, 10), future, 3)["proprio"][-1]
                        values = h.projected_cost(pred.reshape(batch, 256, 16), target, pinv, pos_std)
                        torch.cuda.synchronize()
                        return values
                started = time.monotonic()
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
                index = int(np.argmin([score for _, score in evaluations]))
                suffix = (torch.as_tensor(evaluations[index][0], device=device)*std.repeat(10)+mean.repeat(10)).reshape(10, 2).cpu().tolist()
                rows.append({"root": root, "role": role, "prefix": prefix, "suffix": suffix,
                             "history_actions": raw_history.tolist(), "history_normalized": history[:, 0].cpu().tolist(),
                             "cp5_rgb_sha256": old.digest_image(frames[20]), "warmup_times_ns": warm_times,
                             "prefix_times_ns": prefix_times, "best_index": index,
                             "evaluations": [{"suffix_normalized": a, "score": b} for a, b in evaluations],
                             "prediction_samples": prediction_samples, "prediction_calls": prediction_calls,
                             "encoding_calls": 3, "encoding_seconds": encode_seconds,
                             "search_seconds": time.monotonic()-started})
                print(f"root {root} {role}: planned", flush=True)
        if h.parameter_hash(model) != source["model_hash"]:
            raise ValueError("model mutated")
    with CONTEXT.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    old.write_json(DECISION, {"rows": rows, "source_sha256": sha(prior.OUTPUT), "context_sha256": sha(CONTEXT),
                              "runner_sha256": sha(Path(__file__)), "protocol_sha256": sha(prior.PROTOCOL),
                              "full_cost_certified": False})
    print("all cp5 plans sealed", flush=True)
    outputs, results = {}, []
    for row in rows:
        root, role = row["root"], row["role"]
        with ShadowRGBBridge(np.asarray([*old.STARTS[root], 0., 0.])) as env:
            old.do_actions(env, old.WARMUP)
            prefix_frames, _ = old.do_actions(env, row["prefix"])
            if old.digest_image(prefix_frames[-1]) != row["cp5_rgb_sha256"]:
                raise ValueError("cp5 replay mismatch")
            frames, times = old.do_actions(env, row["suffix"])
        outputs[f"root{root}_{role}"] = np.stack(frames)
        results.append({"root": root, "role": role, "prefix": row["prefix"], "suffix": row["suffix"],
                        "rgb_sha256": [old.digest_image(x) for x in frames], "times_ns": times})
    with IMAGES.open("xb") as handle:
        np.savez_compressed(handle, **outputs)
    old.write_json(EXECUTION, {"rows": results, "decision_sha256": sha(DECISION), "arrays_sha256": sha(IMAGES)})
    print("all cp5 plans executed", flush=True)


if __name__ == "__main__":
    main()
