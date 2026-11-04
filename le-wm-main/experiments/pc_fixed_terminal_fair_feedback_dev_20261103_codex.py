"""Fixed-endpoint RGB feedback: goal-mode programs vs token-matched full CEM."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_goal_mode_matched_feedback_20261103_codex as previous
from experiments.pc_budget_allocation_20261103_codex import optimize
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history
from experiments.spectral_reachability import pc_goal_conditioned_program_20261103_codex as generator
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter

old, h, base, ROOT = previous.old, previous.h, previous.base, previous.ROOT
PROTOCOL = ROOT.parent / "docs/PC_FIXED_TERMINAL_FAIR_FEEDBACK_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT / "results/pc_fixed_terminal_fair_feedback_dev_20261103_codex"
ARMS = ("goal", "cem")
CAP = 480


def paths(root, arm):
    stem = PREFIX.with_name(PREFIX.name + f"_root{root}_{arm}")
    return ([stem.with_name(stem.name + f"_decision{i}.json") for i in range(3)],
            stem.with_name(stem.name + "_execution.json"),
            stem.with_name(stem.name + "_images.npz"))


def decide(arm, root, step, target, predict, token_meter, carry=None):
    horizon = 15 - 5 * step
    tokens_per = horizon // 5
    if arm not in ARMS or step not in (0, 1, 2):
        raise ValueError("invalid decision")
    prior_score = None
    if carry is not None:
        carry = np.asarray(carry, dtype=float)
        if carry.shape != (horizon, 2):
            raise ValueError("carry horizon mismatch")
        prior_score = float(predict(carry[None])[1][0])
    elif step:
        raise ValueError("missing previous plan tail")
    seed = 2026112600 + 10 * root + step
    spectral_meter = Meter({"prediction_samples": 64, "svd": 1, "linear_solve": 1})
    programs, evidence, prefix_scores = [], None, []
    if arm == "goal":
        bank, evidence = generator.generate(lambda prefix: predict(np.asarray(prefix)[None])[0][0],
                                             target, spectral_meter, mode="goal", seed=seed)
        prefix_scores = [{"name": item["name"], "score": float(predict(np.asarray(item["prefix"])[None])[1][0])}
                         for item in bank]
        if horizon > 5:
            affordable = (CAP - token_meter.token_samples) // tokens_per
            quotient, remainder = divmod(affordable, len(bank))
            for i, item in enumerate(bank):
                count = quotient + (i < remainder)
                if count < 1:
                    raise ValueError("not enough budget for a candidate")
                chain, score, values = optimize(item["prefix"], horizon, count,
                                                np.random.default_rng(seed), lambda x: predict(x)[1])
                programs.append({"name": item["name"], "chain": chain.tolist(),
                                 "terminal_score": score, "samples": count, "scores": values})
        else:
            programs = [{"name": item["name"], "chain": item["prefix"], "terminal_score": entry["score"],
                         "samples": 0, "scores": []} for item, entry in zip(bank, prefix_scores, strict=True)]
            count = CAP - token_meter.token_samples
            if count > 0:
                chain, score, values = optimize([], horizon, count, np.random.default_rng(seed), lambda x: predict(x)[1])
                programs.append({"name": "direct_cem", "chain": chain.tolist(), "terminal_score": score,
                                 "samples": count, "scores": values})
        zero = next(item for item in programs if item["name"] == "zero")
        best = min(programs, key=lambda x: x["terminal_score"])
        selected = zero if zero["terminal_score"] <= best["terminal_score"] else best
        abstained = selected["name"] == "zero"
    else:
        count = (CAP - token_meter.token_samples) // tokens_per
        chain, score, values = optimize([], horizon, count, np.random.default_rng(seed), lambda x: predict(x)[1])
        selected = {"name": "direct_cem", "chain": chain.tolist(), "terminal_score": score,
                    "samples": count, "scores": values}
        programs, abstained = [selected], None
    chosen_source = selected["name"]
    if carry is not None and prior_score < selected["terminal_score"]:
        selected = {"name": "carry", "chain": carry.tolist(), "terminal_score": prior_score}
        chosen_source = "carry"
    return {"horizon": horizon, "programs": programs, "prefix_scores": prefix_scores, "generator": evidence,
            "selected": selected, "selection_source": chosen_source, "abstained_reconfiguration": abstained,
            "carry": None if carry is None else carry.tolist(), "carry_score": prior_score,
            "spectral_counts": spectral_meter.counts, "spectral_seconds": spectral_meter.seconds}


def main():
    all_paths = [path for root in range(4) for arm in ARMS for group in paths(root, arm)
                 for path in (group if isinstance(group, list) else [group])]
    if any(path.exists() for path in all_paths):
        raise FileExistsError("partial or complete development panel already exists")
    sha = old.prior.sha
    source = json.loads(old.DECISIONS.read_text())
    if sha(old.HISTORY) != source["history_sha256"]:
        raise ValueError("history mismatch")
    camera, calibration = (json.loads(p.read_text()) for p in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    print("Fixed terminal 15/10/5-step RGB feedback, development only.", flush=True)
    with np.load(old.HISTORY, allow_pickle=False) as histories, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        model_hash = h.parameter_hash(model)
        if model_hash != source["model_hash"] or sha(h.DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]:
            raise ValueError("checkpoint mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not np.array_equal(mean.cpu().numpy(), source["action_mean"]) or not np.array_equal(std.cpu().numpy(), source["action_std"]):
            raise ValueError("action normalizer mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal = obstacle["goal_rgb"]
        if old.digest_image(goal) != source["goal_sha256"]:
            raise ValueError("goal image mismatch")
        with torch.no_grad():
            goal_state, _ = h.estimate_state(goal, metadata, offset)
            target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
            goal_position = ((target.mean(0) @ pinv.T)[:2] * pos_std).cpu().numpy()
        for root in range(4):
            for arm in ARMS:
                decisions, execution, images = paths(root, arm)
                executed, rows, carry = [], [], None
                with ShadowRGBBridge(np.asarray([*old.STARTS[root], 0., 0.])) as env:
                    warm, _ = old.do_actions(env, old.WARMUP)
                    frames = [env.initial_rgb.copy(), *warm]
                    if ([old.digest_image(x) for x in frames] != source["rows"][root]["rgb_sha256"]
                            or not np.array_equal(np.stack(frames), histories[f"root{root}_history_rgb"])):
                        raise ValueError("real warmup history mismatch")
                    actions = old.WARMUP.tolist()
                    for step in range(3):
                        ticks = previous.prior.context_ticks(len(frames))
                        started = time.monotonic()
                        with torch.no_grad():
                            entries = [h.encode_with_state(model, frames[t],
                                       h.causal_state(frames[t-1], frames[t], metadata, offset, .05), device)
                                       for t in ticks]
                            torch.cuda.synchronize()
                        ctx = {key: torch.cat([e[key] for e in entries], dim=1) for key in ("visual", "proprio")}
                        history_actions = np.asarray(actions[-10:], dtype=float)
                        history = ((torch.as_tensor(history_actions, dtype=torch.float32, device=device) - mean) / std).reshape(2, 1, 10)
                        encode_seconds = time.monotonic() - started
                        meter = base.Meter(CAP)

                        def predict(chains):
                            def raw(batch):
                                n, steps, _ = batch.shape
                                packets = ((torch.as_tensor(batch, dtype=torch.float32, device=device) - mean) / std).reshape(
                                    n, steps // 5, 10).transpose(0, 1)
                                with torch.no_grad():
                                    out = unroll_with_action_history(model, ctx, history.expand(2, n, 10), packets, 3)["proprio"][-1]
                                    features = out.reshape(n, 256, 16)
                                    position = (features.mean(1) @ pinv.T)[:, :2] * pos_std
                                    score = h.projected_cost(features, target, pinv, pos_std)
                                    torch.cuda.synchronize()
                                return position.cpu().numpy(), score.cpu().numpy()
                            return meter.predict(np.asarray(chains, dtype=np.float32), raw)

                        started = time.monotonic()
                        result = decide(arm, root, step, goal_position, predict, meter, carry)
                        chain = result["selected"]["chain"]
                        if len(chain) != 15 - 5 * step or meter.token_samples > CAP:
                            raise ValueError("selected program or token budget invalid")
                        decision = {"root": root, "arm": arm, "step": step, "ticks": ticks,
                                    "history_actions": history_actions.tolist(), "result": result,
                                    "input_rgb_sha256": [old.digest_image(frames[t]) for t in ticks],
                                    "counts": meter.as_dict(), "encoding_calls": 3, "encoding_seconds": encode_seconds,
                                    "planning_seconds": time.monotonic() - started, "model_hash": model_hash,
                                    "checkpoint_sha256": sha(h.DEFAULT_CHECKPOINT), "history_sha256": sha(old.HISTORY),
                                    "protocol_sha256": sha(PROTOCOL), "runner_sha256": sha(Path(__file__)),
                                    "generator_sha256": sha(Path(generator.__file__)),
                                    "optimizer_sha256": sha(Path(optimize.__code__.co_filename)), "full_cost_certified": False}
                        old.write_json(decisions[step], decision)
                        observed, times = old.do_actions(env, chain[:5])
                        frames.extend(observed)
                        actions.extend(chain[:5])
                        executed.extend(chain[:5])
                        carry = chain[5:] if step < 2 else None
                        rows.append({"step": step, "decision_sha256": sha(decisions[step]), "actions": chain[:5],
                                     "rgb_sha256": [old.digest_image(x) for x in observed], "times_ns": times})
                    with images.open("xb") as handle:
                        np.savez_compressed(handle, rgb=np.stack(frames))
                    old.write_json(execution, {"root": root, "arm": arm, "rows": rows, "executed_actions": executed,
                                               "images_sha256": sha(images), "full_cost_certified": False})
                print(f"root {root} {arm}: three fixed-terminal RGB decisions", flush=True)
        if h.parameter_hash(model) != model_hash:
            raise ValueError("frozen model mutated")


if __name__ == "__main__":
    main()
