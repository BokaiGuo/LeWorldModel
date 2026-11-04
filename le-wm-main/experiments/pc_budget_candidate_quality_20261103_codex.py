"""Real-RGB equal-feedback comparison of spectral and nominal candidates."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_equal_resource_dev_20261103_codex as base
from experiments.pc_budget_allocation_20261103_codex import allocate_samples, optimize
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history
from experiments.spectral_reachability import pc_multi_nominal_candidates_20261103_codex as compiler

old, h, ROOT = base.old, base.h, base.ROOT
PROTOCOL = ROOT.parent/"docs/PC_BUDGET_ALLOCATION_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT/"results/pc_budget_candidate_quality_20261103_codex"
ROOT_PLAN = PREFIX.with_name(PREFIX.name+"_root.json")
CONTEXT = PREFIX.with_name(PREFIX.name+"_context.npz")
CP_PLAN = PREFIX.with_name(PREFIX.name+"_cp10.json")
EXEC = PREFIX.with_name(PREFIX.name+"_execution.json")
IMAGES = PREFIX.with_name(PREFIX.name+"_execution.npz")
METHODS = ("spectral", "random", "nominal", "cem")
BUDGETS = (900, 1800)


def nominals():
    return [{"name": f"h{length}_n{index}", "horizon": length,
             "prefix": np.tile([value, 0.], (length, 1)).tolist()}
            for length in (5, 10) for index, value in enumerate((0., .25, -.25))]


def diverse(group, keep=4):
    selected, remaining = [group[0]], group[1:]
    while remaining and len(selected) < keep:
        index = max(range(len(remaining)), key=lambda k: min(
            np.linalg.norm(np.asarray(remaining[k]["predicted_response"])-x["predicted_response"]) for x in selected))
        selected.append(remaining.pop(index))
    return selected


def make_bank(method, predict, svd, seed):
    bank = nominals()
    if method == "spectral":
        generated = compiler.build_candidates(predict, svd=svd)
        bank += generated["candidates"]
        evidence = generated["evidence"]
    elif method == "random":
        rng = np.random.default_rng(seed)
        evidence = []
        for length in (5, 10):
            group = []
            for item in (x for x in bank if x["horizon"] == length):
                nominal = np.asarray(item["prefix"])
                direction = rng.normal(size=nominal.shape)
                direction /= np.max(np.abs(direction))
                for sign in (1, -1):
                    action = np.clip(nominal+sign*.25*direction, -1, 1)
                    group.append({"name": item["name"]+f"_r{sign:+d}", "horizon": length,
                                  "prefix": action.tolist(), "predicted_response": predict(action).tolist()})
            bank += diverse(group)
    elif method == "nominal":
        evidence = []
    else:
        raise ValueError("unknown candidate method")
    return bank, evidence


def main():
    if any(path.exists() for path in (ROOT_PLAN, CONTEXT, CP_PLAN, EXEC, IMAGES)):
        raise FileExistsError("refusing overwrite")
    sha = old.prior.sha
    source = json.loads(old.DECISIONS.read_text())
    if sha(old.HISTORY) != source["history_sha256"]:
        raise ValueError("history mismatch")
    camera, calibration = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    root_rows, cp_rows, arrays, images, executions = [], [], {}, {}, []
    with np.load(old.HISTORY, allow_pickle=False) as histories, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT/"results/e3_p2_host_cuda_cooperative.lock"):
        load_start = time.monotonic()
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT/"external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        load_seconds = time.monotonic()-load_start
        model_hash = h.parameter_hash(model)
        if model_hash != source["model_hash"] or sha(h.DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]:
            raise ValueError("checkpoint mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not np.array_equal(mean.cpu().numpy(), source["action_mean"]) or not np.array_equal(std.cpu().numpy(), source["action_std"]):
            raise ValueError("action normalization mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal = obstacle["goal_rgb"]
        if old.digest_image(goal) != source["goal_sha256"]:
            raise ValueError("goal mismatch")
        start = time.monotonic()
        with torch.no_grad():
            target_state, _ = h.estimate_state(goal, metadata, offset)
            target = h.encode_with_state(model, goal, target_state, device)["proprio"].reshape(256, 16)
            torch.cuda.synchronize()
        goal_seconds = time.monotonic()-start

        def encode(frames, ticks, actions):
            start = time.monotonic()
            with torch.no_grad():
                entries = [h.encode_with_state(model, frames[t],
                           h.causal_state(frames[t-1], frames[t], metadata, offset, .05), device) for t in ticks]
                torch.cuda.synchronize()
            context = {key: torch.cat([x[key] for x in entries], dim=1) for key in ("visual", "proprio")}
            history = ((torch.as_tensor(actions, dtype=torch.float32, device=device)-mean)/std).reshape(2, 1, 10)
            return context, history, time.monotonic()-start

        def predictor(ctx, history, meter):
            def raw(actions):
                batch, steps, _ = actions.shape
                packets = ((torch.as_tensor(actions, dtype=torch.float32, device=device)-mean)/std).reshape(batch, steps//5, 10).transpose(0, 1)
                with torch.no_grad():
                    pred = unroll_with_action_history(model, ctx, history.expand(2, batch, 10), packets, 3)["proprio"][-1]
                    features = pred.reshape(batch, 256, 16)
                    position = (features.mean(1) @ pinv.T)[:, :2]*pos_std
                    scores = h.projected_cost(features, target, pinv, pos_std)
                    torch.cuda.synchronize()
                return position.cpu().numpy(), scores.cpu().numpy()
            return lambda actions: meter.predict(np.asarray(actions, dtype=np.float32), raw)

        sealed_path = ROOT/"results/pc_multinominal_compare_20261103_codex_root.json"
        sealed = json.loads(sealed_path.read_text())
        previous = ROOT/"experiments/pc_multinominal_compare_20261103_codex.py"
        previous_protocol = ROOT.parent/"docs/PC_MULTINOMINAL_COMPARISON_PROTOCOL_20261103_CODEX.md"
        if (sealed["runner_sha256"] != sha(previous)
                or sealed["protocol_sha256"] != sha(previous_protocol)
                or sealed["history_sha256"] != sha(old.HISTORY)):
            raise ValueError("fixed bank provenance mismatch")
        for root in range(4):
            frames = histories[f"root{root}_history_rgb"]
            if [old.digest_image(x) for x in frames] != source["rows"][root]["rgb_sha256"]:
                raise ValueError("history RGB mismatch")
            started = time.monotonic()
            ctx, hist, encode_seconds = encode(frames, (5, 10, 15), old.WARMUP[5:])
            rows = [next(row for row in sealed["rows"] if row["root"] == root
                         and row["budget"] == 900 and row["method"] == method)
                    for method in ("spectral", "random")]
            bank, seen = [], set()
            for method, row in zip(("spectral", "random"), rows, strict=True):
                for item in row["programs"]:
                    prefix = np.asarray(item["prefix"], dtype=np.float64)
                    key = (len(prefix), prefix.tobytes())
                    if key not in seen:
                        seen.add(key)
                        bank.append({"name": item["name"], "source_bank": method,
                                     "horizon": item["horizon"], "prefix": item["prefix"]})
            meter = base.Meter(len(bank)*48*3)
            predict = predictor(ctx, hist, meter)
            programs = []
            for item in bank:
                chain, score, scores = optimize(item["prefix"], 15, 48,
                    np.random.default_rng(2026110630+root), lambda x: predict(x)[1])
                programs.append({**item, "chain": chain.tolist(), "terminal_score": score,
                                 "search_samples": 48, "scores": scores})
            root_rows.append({"root": root, "programs": programs, "counts": meter.as_dict(),
                              "encoding_calls": 3, "encoding_seconds": encode_seconds,
                              "planning_wall_seconds": time.monotonic()-started})
            print(f"root {root}: {len(bank)} candidates", flush=True)
        if h.parameter_hash(model) != model_hash:
            raise ValueError("model mutated")
        old.write_json(ROOT_PLAN, {"rows": root_rows, "protocol_sha256": sha(PROTOCOL), "source_sha256": sha(old.DECISIONS),
                                   "fixed_bank_sha256": sha(sealed_path),
                                   "history_sha256": sha(old.HISTORY), "runner_sha256": sha(Path(__file__)),
                                   "generator_sha256": sha(Path(compiler.__file__)), "model_hash": model_hash,
                                   "shared_model_load_seconds": load_seconds, "shared_goal_encoding_seconds": goal_seconds})
        print("all root plans sealed", flush=True)
        return
        for row in root_rows:
            root, budget, method = row["root"], row["budget"], row["method"]
            actions = row["chosen"]["chain"][:10]
            with ShadowRGBBridge(np.asarray([*old.STARTS[root], 0., 0.])) as env:
                frames = [env.initial_rgb.copy()]
                warm, _ = old.do_actions(env, old.WARMUP)
                frames += warm
                if [old.digest_image(x) for x in frames] != source["rows"][root]["rgb_sha256"]:
                    raise ValueError("warmup mismatch")
                observed, times = old.do_actions(env, actions)
                frames += observed
            key = f"root{root}_b{budget}_{method}"
            arrays[key] = np.stack(frames)
            ctx, hist, encode_seconds = encode(frames, (15, 20, 25), actions)
            meter = base.Meter(96)
            predict = predictor(ctx, hist, meter)
            started = time.monotonic()
            suffix, score, scores = optimize([], 5, 96, np.random.default_rng(2026110640+root), lambda x: predict(x)[1])
            cp_rows.append({"root": root, "budget": budget, "method": method, "first10": actions,
                            "suffix": suffix.tolist(), "cp10_rgb_sha256": old.digest_image(frames[25]),
                            "counts": meter.as_dict(), "encoding_calls": 3, "encoding_seconds": encode_seconds,
                            "model_score": score, "scores": scores, "planning_seconds": time.monotonic()-started,
                            "first10_times_ns": times})
            print(f"root {root} budget {budget} {method}: cp10 planned", flush=True)
        with CONTEXT.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        old.write_json(CP_PLAN, {"rows": cp_rows, "root_plan_sha256": sha(ROOT_PLAN), "context_sha256": sha(CONTEXT),
                                 "protocol_sha256": sha(PROTOCOL), "full_cost_certified": False})
        print("all cp10 plans sealed", flush=True)
        for row in cp_rows:
            root, budget, method = row["root"], row["budget"], row["method"]
            with ShadowRGBBridge(np.asarray([*old.STARTS[root], 0., 0.])) as env:
                old.do_actions(env, old.WARMUP)
                first, _ = old.do_actions(env, row["first10"])
                if old.digest_image(first[-1]) != row["cp10_rgb_sha256"]:
                    raise ValueError("feedback replay mismatch")
                frames, times = old.do_actions(env, row["suffix"])
            key = f"root{root}_b{budget}_{method}"
            images[key] = np.stack(frames)
            executions.append({"root": root, "budget": budget, "method": method, "first10": row["first10"],
                               "suffix": row["suffix"], "rgb_sha256": [old.digest_image(x) for x in frames], "times_ns": times})
        with IMAGES.open("xb") as handle:
            np.savez_compressed(handle, **images)
        old.write_json(EXEC, {"rows": executions, "cp10_sha256": sha(CP_PLAN), "images_sha256": sha(IMAGES)})
        print(f"executed {len(executions)} programs", flush=True)


if __name__ == "__main__":
    main()
