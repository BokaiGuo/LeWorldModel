"""Paired RGB feedback planning with a common token-sample resource policy."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_motion_unprivileged_dev_20261103_codex as old
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history
from experiments.spectral_reachability.pc_rgb_candidate_policy_20261103_codex import build_fixed_bank

h, ROOT = old.h, old.ROOT
PROTOCOL = ROOT.parent/"docs/PC_EQUAL_RESOURCE_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT/"results/pc_equal_resource_dev_20261103_codex"
ROOT_PLAN = PREFIX.with_name(PREFIX.name+"_root.json")
CONTEXT = PREFIX.with_name(PREFIX.name+"_context.npz")
CP_PLAN = PREFIX.with_name(PREFIX.name+"_cp5.json")
EXEC = PREFIX.with_name(PREFIX.name+"_execution.json")
IMAGES = PREFIX.with_name(PREFIX.name+"_execution.npz")
METHODS = ("spectral", "no_spectral", "greedy", "cem")


class Meter:
    def __init__(self, token_cap):
        self.token_cap = token_cap
        self.token_samples = self.samples = self.calls = 0
        self.seconds = 0.

    def predict(self, actions, function):
        tokens = len(actions)*len(actions[0])//5
        if self.token_samples + tokens > self.token_cap:
            raise RuntimeError("prediction token budget exceeded")
        self.token_samples += tokens
        self.samples += len(actions)
        self.calls += 1
        start = time.monotonic()
        result = function(actions)
        self.seconds += time.monotonic()-start
        return result

    def as_dict(self):
        return {"prediction_calls": self.calls, "trajectory_samples": self.samples,
                "prediction_token_samples": self.token_samples, "prediction_seconds": self.seconds,
                "token_cap": self.token_cap, "backward_calls": 0}


def optimize(prefix, horizon, count, rng, predict):
    """Bounded multigeneration CEM; prefix can be empty."""
    prefix = np.asarray(prefix, dtype=float).reshape(-1, 2)
    remaining = horizon-len(prefix)
    if remaining <= 0 or count % 12:
        raise ValueError("unsupported optimizer allocation")
    loc, scale = np.zeros((remaining, 2)), np.full((remaining, 2), 1/3)
    best, score, records = None, float("inf"), []
    for generation in range(count//12):
        samples = np.clip(loc+rng.normal(size=(12, remaining, 2))*scale, -1, 1)
        samples[0] = loc
        chains = np.concatenate((np.broadcast_to(prefix, (12, len(prefix), 2)), samples), axis=1)
        values = predict(chains)
        for chain, value in zip(chains, values, strict=True):
            records.append(float(value))
            if value < score:
                best, score = chain.copy(), float(value)
        elite = samples[np.argsort(values, kind="stable")[:4]]
        loc, scale = elite.mean(0), np.maximum(elite.std(0, ddof=1), 1e-5)
    return best, score, records


def main():
    if any(p.exists() for p in (ROOT_PLAN, CONTEXT, CP_PLAN, EXEC, IMAGES)):
        raise FileExistsError("refusing to overwrite prior execution")
    sha = old.prior.sha
    source = json.loads(old.DECISIONS.read_text())
    if sha(old.HISTORY) != source["history_sha256"]:
        raise ValueError("history mismatch")
    camera, calibration = (json.loads(p.read_text()) for p in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    root_rows, cp_rows, context_arrays, images, executions = [], [], {}, {}, []
    with np.load(old.HISTORY, allow_pickle=False) as histories, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT/"results/e3_p2_host_cuda_cooperative.lock"):
        start = time.monotonic()
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT/"external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        load_seconds = time.monotonic()-start
        model_hash = h.parameter_hash(model)
        if model_hash != source["model_hash"] or sha(h.DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]:
            raise ValueError("checkpoint mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not np.array_equal(mean.cpu().numpy(), source["action_mean"]) or not np.array_equal(std.cpu().numpy(), source["action_std"]):
            raise ValueError("normalization mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal = obstacle["goal_rgb"]
        if old.digest_image(goal) != source["goal_sha256"]:
            raise ValueError("goal mismatch")
        start = time.monotonic()
        with torch.no_grad():
            goal_state, _ = h.estimate_state(goal, metadata, offset)
            target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
            torch.cuda.synchronize()
        goal_seconds = time.monotonic()-start

        def encode(frames, ticks, raw_history):
            start = time.monotonic()
            with torch.no_grad():
                entries = [h.encode_with_state(model, frames[t],
                           h.causal_state(frames[t-1], frames[t], metadata, offset, .05), device) for t in ticks]
                torch.cuda.synchronize()
            context = {k: torch.cat([entry[k] for entry in entries], dim=1) for k in ("visual", "proprio")}
            history = ((torch.as_tensor(raw_history, dtype=torch.float32, device=device)-mean)/std).reshape(2, 1, 10)
            return context, history, time.monotonic()-start

        def predictor(context, history, meter):
            def raw(actions):
                batch, steps, _ = actions.shape
                packets = ((torch.as_tensor(actions, dtype=torch.float32, device=device)-mean)/std).reshape(batch, steps//5, 10).transpose(0, 1)
                with torch.no_grad():
                    pred = unroll_with_action_history(model, context, history.expand(2, batch, 10), packets, 3)["proprio"][-1]
                    features = pred.reshape(batch, 256, 16)
                    position = (features.mean(1) @ pinv.T)[:, :2]*pos_std
                    scores = h.projected_cost(features, target, pinv, pos_std)
                    torch.cuda.synchronize()
                return position.cpu().numpy(), scores.cpu().numpy()
            return lambda a: meter.predict(np.asarray(a, dtype=np.float32), raw)

        for root in range(4):
            frames = histories[f"root{root}_history_rgb"]
            if [old.digest_image(x) for x in frames] != source["rows"][root]["rgb_sha256"]:
                raise ValueError("root history mismatch")
            for method in METHODS:
                started = time.monotonic()
                context, history, encode_seconds = encode(frames, (5, 10, 15), old.WARMUP[5:])
                meter = Meter(1300)
                predict = predictor(context, history, meter)
                rng = np.random.default_rng(2026110500+root)
                bank = build_fixed_bank(2026110321+root)
                svd_seconds, singular = 0., []
                prefix_evaluations = []
                if method in ("spectral", "greedy"):
                    columns = []
                    for unit in np.eye(10).reshape(10, 5, 2):
                        pos_plus, _ = predict((.05*unit)[None])
                        pos_minus, _ = predict((-.05*unit)[None])
                        columns.append((pos_plus[0]-pos_minus[0])/.1)
                    jacobian = np.stack(columns, axis=1)
                    start_svd = time.monotonic()
                    _, singular, vh = np.linalg.svd(jacobian, full_matrices=False)
                    svd_seconds = time.monotonic()-start_svd
                    for i, vector in enumerate(vh[:2]):
                        if singular[i] <= 1e-10:
                            continue
                        vector = vector.reshape(5, 2)
                        for sign in (1, -1):
                            bank.append({"name": f"spectral_{i}_{sign:+d}", "source": "rgb_model",
                                         "prefix": (sign*.5*vector/np.max(np.abs(vector))).tolist()})
                if method == "cem":
                    chain, score, scores = optimize([], 15, 432, rng, lambda x: predict(x)[1])
                    candidates = [{"name": "full_horizon_cem", "prefix": chain[:5].tolist(),
                                   "suffix": chain[5:].tolist(), "terminal_score": score, "scores": scores}]
                    chosen = candidates[0]
                else:
                    prefixes = np.asarray([x["prefix"] for x in bank])
                    _, immediate = predict(prefixes)
                    prefix_evaluations = [{"name": item["name"], "score": float(value)}
                                          for item, value in zip(bank, immediate, strict=True)]
                    if method == "greedy":
                        selected = int(np.argmin(immediate))
                        candidates_bank = [bank[selected]]
                        count = ((1300-meter.token_samples)//36)*12
                    else:
                        candidates_bank = bank
                        count = 24 if method == "spectral" else 36
                    candidates = []
                    for item in candidates_bank:
                        # Common random numbers isolate prefix-dependent evaluations.
                        local_rng = np.random.default_rng(2026110520+root)
                        chain, score, scores = optimize(item["prefix"], 15, count, local_rng, lambda x: predict(x)[1])
                        candidates.append({**item, "suffix": chain[5:].tolist(), "terminal_score": score, "scores": scores})
                    chosen = min(candidates, key=lambda x: x["terminal_score"])
                root_rows.append({"root": root, "method": method, "chosen": chosen, "candidates": candidates,
                                  "prefix_evaluations": prefix_evaluations,
                                  "counts": meter.as_dict(), "svd_calls": int(method in ("spectral", "greedy")),
                                  "svd_seconds": svd_seconds, "singular_values": np.asarray(singular).tolist(),
                                  "encoding_calls": 3, "encoding_seconds": encode_seconds,
                                  "planning_wall_seconds": time.monotonic()-started})
                print(f"root {root} {method}: {chosen['name']}, tokens={meter.token_samples}", flush=True)
        if h.parameter_hash(model) != model_hash:
            raise ValueError("model mutated")
        old.write_json(ROOT_PLAN, {"rows": root_rows, "protocol_sha256": sha(PROTOCOL), "history_sha256": sha(old.HISTORY),
                                   "source_sha256": sha(old.DECISIONS), "runner_sha256": sha(Path(__file__)),
                                   "model_hash": model_hash, "model_load_seconds_shared": load_seconds,
                                   "goal_encoding_calls_shared": 1, "goal_encoding_seconds_shared": goal_seconds})
        print("all root plans sealed", flush=True)

        for root_row in root_rows:
            root, method, chosen = root_row["root"], root_row["method"], root_row["chosen"]
            with ShadowRGBBridge(np.asarray([*old.STARTS[root], 0., 0.])) as env:
                frames = [env.initial_rgb.copy()]
                warm, _ = old.do_actions(env, old.WARMUP)
                frames += warm
                if [old.digest_image(x) for x in frames] != source["rows"][root]["rgb_sha256"]:
                    raise ValueError("warmup RGB differs")
                prefix_frames, prefix_times = old.do_actions(env, chosen["prefix"])
                frames += prefix_frames
            context_arrays[f"root{root}_{method}"] = np.stack(frames)
            context, history, encode_seconds = encode(frames, (10, 15, 20),
                                                       np.concatenate((old.WARMUP[10:15], chosen["prefix"])))
            meter = Meter(200)
            predict = predictor(context, history, meter)
            rng = np.random.default_rng(2026110540+root)
            # 96 actual samples fit under the 200-token cap for all methods.
            started = time.monotonic()
            suffix, score, scores = optimize([], 10, 96, rng, lambda x: predict(x)[1])
            cp_rows.append({"root": root, "method": method, "prefix": chosen["prefix"], "suffix": suffix.tolist(),
                            "cp5_rgb_sha256": old.digest_image(frames[20]), "model_score": score, "scores": scores,
                            "counts": meter.as_dict(), "encoding_calls": 3, "encoding_seconds": encode_seconds,
                            "planning_wall_seconds": time.monotonic()-started, "prefix_times_ns": prefix_times})
            print(f"root {root} {method}: cp5 planned", flush=True)
        with CONTEXT.open("xb") as handle:
            np.savez_compressed(handle, **context_arrays)
        old.write_json(CP_PLAN, {"rows": cp_rows, "root_plan_sha256": sha(ROOT_PLAN), "context_sha256": sha(CONTEXT),
                                 "protocol_sha256": sha(PROTOCOL), "full_cost_certified": False})
        print("all cp5 plans sealed", flush=True)
        for row in cp_rows:
            root, method = row["root"], row["method"]
            with ShadowRGBBridge(np.asarray([*old.STARTS[root], 0., 0.])) as env:
                old.do_actions(env, old.WARMUP)
                prefix_frames, _ = old.do_actions(env, row["prefix"])
                if old.digest_image(prefix_frames[-1]) != row["cp5_rgb_sha256"]:
                    raise ValueError("cp5 RGB differs")
                frames, times = old.do_actions(env, row["suffix"])
            images[f"root{root}_{method}"] = np.stack(frames)
            executions.append({"root": root, "method": method, "prefix": row["prefix"], "suffix": row["suffix"],
                               "rgb_sha256": [old.digest_image(x) for x in frames], "times_ns": times})
        with IMAGES.open("xb") as handle:
            np.savez_compressed(handle, **images)
        old.write_json(EXEC, {"rows": executions, "cp5_sha256": sha(CP_PLAN), "images_sha256": sha(IMAGES)})
        print("all 16 physical programs executed", flush=True)


if __name__ == "__main__":
    main()
