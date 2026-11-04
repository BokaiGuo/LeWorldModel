"""Matched-token, real-RGB development controls for goal-mode feedback."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_goal_mode_abstain_rgb_mpc_20261103_codex as prior
from experiments.pc_budget_allocation_20261103_codex import optimize
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history
from experiments.spectral_reachability import pc_goal_conditioned_program_20261103_codex as generator
from experiments.spectral_reachability.pc_program_abstention_20261103_codex import select_with_abstention
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter, plan_bank

old, h, base, ROOT = prior.old, prior.h, prior.base, prior.ROOT
PREFIX = ROOT / "results/pc_goal_mode_matched_feedback_20261103_codex"
PROTOCOL = ROOT.parent / "docs/PC_GOAL_MODE_MATCHED_FEEDBACK_DEV_PROTOCOL_20261103_CODEX.md"
ARMS = ("goal", "greedy", "ridge", "random", "cem")


def paths(root, arm):
    stem = PREFIX.with_name(PREFIX.name + f"_root{root}_{arm}")
    return ([stem.with_name(stem.name + f"_decision{i}.json") for i in range(3)],
            stem.with_name(stem.name + "_execution.json"),
            stem.with_name(stem.name + "_images.npz"))


def plan_at_context(arm, root, step, response, score, target, initial_score):
    """Run a single planning decision; caller instruments the actual predictor."""
    meter = Meter({"prediction_samples": 200, "svd": 1,
                   "linear_solve": 1, "suffix_optimizations": 3})
    rng_seed, cem_seed = 2026110800 + 10 * root + step, 2026110900 + 10 * root + step
    if arm == "cem":
        chain, terminal, scores = optimize([], 15, 44, np.random.default_rng(cem_seed),
                                           lambda x: score(x))
        return {"selected": {"name": "cem", "prefix": chain[:5].tolist(),
                             "suffix": chain[5:].tolist(), "terminal_score": terminal},
                "scores": scores, "counts": meter.counts, "seconds": meter.seconds}
    if arm == "random":
        direction = np.random.default_rng(rng_seed).normal(size=(5, 2))
        direction *= .5 / np.max(np.abs(direction))
        bank = [{"name": "zero", "prefix": np.zeros((5, 2)).tolist(), "source": "control"}]
        bank += [{"name": f"random_{sign:+d}", "prefix": (sign * direction).tolist(),
                  "source": "random"} for sign in (-1, 1)]
        evidence = {"mode": "random", "seed": rng_seed}
        allocation = iter((15, 14, 14))
    elif arm in ("goal", "greedy", "ridge"):
        mode = "goal" if arm in ("goal", "greedy") else "ridge"
        bank, evidence = generator.generate(response, target, meter, mode=mode, seed=rng_seed)
        allocation = iter([12] * len(bank))
    else:
        raise ValueError("unsupported arm")

    def prefix_score(prefix):
        return float(score(np.asarray(prefix)[None])[0])

    def suffix(prefix, maximum, inner_meter):
        count = next(allocation)
        def measured(chains):
            return inner_meter.call("prediction_samples", len(chains), score, chains)
        chain, terminal, _ = optimize(prefix, 15, count, np.random.default_rng(cem_seed), measured)
        return chain[5:], terminal

    result = plan_bank(bank, initial_score, prefix_score, suffix, meter,
                       suffix_samples=15 if arm == "random" else 12)
    if arm == "greedy":
        result["selected"] = next(item for item in result["programs"]
                                  if item["name"] == result["same_bank_greedy_name"])
        decision = {"policy": "same_bank_greedy", "abstained": None}
    else:
        selection = select_with_abstention(result["programs"], 0.)
        result["selected"] = selection.pop("selected")
        decision = selection
    result.pop("qualified_against_zero_only", None)
    result.update({"generator": evidence, "selection": decision,
                   "counts": meter.counts, "seconds": meter.seconds})
    return result


def main():
    all_paths = [path for root in range(4) for arm in ARMS
                 for group in paths(root, arm) for path in (group if isinstance(group, list) else [group])]
    if any(path.exists() for path in all_paths):
        raise FileExistsError("partial or complete development panel exists")
    sha = old.prior.sha
    source = json.loads(old.DECISIONS.read_text())
    if sha(old.HISTORY) != source["history_sha256"]:
        raise ValueError("history archive mismatch")
    camera, calibration = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    print("Four reused roots; five RGB feedback development arms; no independent claim.", flush=True)
    with np.load(old.HISTORY, allow_pickle=False) as histories, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        setup_start = time.monotonic()
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        model_hash = h.parameter_hash(model)
        if model_hash != source["model_hash"] or sha(h.DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]:
            raise ValueError("checkpoint mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if (not np.array_equal(mean.cpu().numpy(), source["action_mean"])
                or not np.array_equal(std.cpu().numpy(), source["action_std"])):
            raise ValueError("action normalization mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal = obstacle["goal_rgb"]
        if old.digest_image(goal) != source["goal_sha256"]:
            raise ValueError("goal mismatch")
        with torch.no_grad():
            goal_state, _ = h.estimate_state(goal, metadata, offset)
            target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
            goal_position = ((target.mean(0) @ pinv.T)[:2] * pos_std).cpu().numpy()
            torch.cuda.synchronize()
        setup_seconds = time.monotonic() - setup_start
        for root in range(4):
            for arm in ARMS:
                decisions, execution, images = paths(root, arm)
                rows = []
                with ShadowRGBBridge(np.asarray([*old.STARTS[root], 0., 0.])) as env:
                    warm, _ = old.do_actions(env, old.WARMUP)
                    frames = [env.initial_rgb.copy(), *warm]
                    if ([old.digest_image(x) for x in frames] != source["rows"][root]["rgb_sha256"]
                            or not np.array_equal(np.stack(frames), histories[f"root{root}_history_rgb"])):
                        raise ValueError("warmup history mismatch")
                    actions = old.WARMUP.tolist()
                    for step in range(3):
                        ticks = prior.context_ticks(len(frames))
                        started = time.monotonic()
                        with torch.no_grad():
                            entries = [h.encode_with_state(model, frames[t],
                                       h.causal_state(frames[t-1], frames[t], metadata, offset, .05), device)
                                       for t in ticks]
                            torch.cuda.synchronize()
                        ctx = {key: torch.cat([entry[key] for entry in entries], dim=1)
                               for key in ("visual", "proprio")}
                        history_actions = np.asarray(actions[-10:], dtype=float)
                        history = ((torch.as_tensor(history_actions, dtype=torch.float32, device=device)
                                    - mean) / std).reshape(2, 1, 10)
                        with torch.no_grad():
                            initial = float(h.projected_cost(ctx["proprio"][:, -1:].reshape(1, 256, 16),
                                                             target, pinv, pos_std)[0])
                        encoding_seconds = time.monotonic() - started
                        token_meter = base.Meter(132)

                        def predict(chains):
                            def raw(batch):
                                n, steps, _ = batch.shape
                                packets = ((torch.as_tensor(batch, dtype=torch.float32, device=device) - mean) / std).reshape(
                                    n, steps // 5, 10).transpose(0, 1)
                                with torch.no_grad():
                                    pred = unroll_with_action_history(model, ctx, history.expand(2, n, 10),
                                                                      packets, 3)["proprio"][-1]
                                    features = pred.reshape(n, 256, 16)
                                    pos = (features.mean(1) @ pinv.T)[:, :2] * pos_std
                                    value = h.projected_cost(features, target, pinv, pos_std)
                                    torch.cuda.synchronize()
                                return pos.cpu().numpy(), value.cpu().numpy()
                            return token_meter.predict(np.asarray(chains, dtype=np.float32), raw)

                        started = time.monotonic()
                        result = plan_at_context(arm, root, step,
                                                 lambda x: predict(np.asarray(x)[None])[0][0],
                                                 lambda x: predict(x)[1], goal_position, initial)
                        selected = result["selected"]
                        decision = {"root": root, "arm": arm, "step": step, "ticks": ticks,
                                    "history_actions": history_actions.tolist(),
                                    "input_rgb_sha256": [old.digest_image(frames[t]) for t in ticks],
                                    "initial_score": initial, "result": result, "token_counts": token_meter.as_dict(),
                                    "encoding_calls": 3, "encoding_seconds": encoding_seconds,
                                    "planning_seconds": time.monotonic() - started, "model_hash": model_hash,
                                    "checkpoint_sha256": sha(h.DEFAULT_CHECKPOINT), "history_sha256": sha(old.HISTORY),
                                    "protocol_sha256": sha(PROTOCOL), "runner_sha256": sha(Path(__file__)),
                                    "generator_sha256": sha(Path(generator.__file__)),
                                    "planner_sha256": sha(Path(plan_bank.__code__.co_filename)),
                                    "optimizer_sha256": sha(Path(optimize.__code__.co_filename)),
                                    "source_sha256": sha(old.DECISIONS), "camera_sha256": sha(h.CAMERA),
                                    "calibration_sha256": sha(h.OLD), "goal_sha256": old.digest_image(goal),
                                    "setup_seconds_shared": setup_seconds, "full_cost_certified": False}
                        old.write_json(decisions[step], decision)
                        observed, times = old.do_actions(env, selected["prefix"])
                        frames.extend(observed)
                        actions.extend(selected["prefix"])
                        rows.append({"step": step, "decision_sha256": sha(decisions[step]),
                                     "actions": selected["prefix"], "rgb_sha256": [
                                         old.digest_image(image) for image in observed], "times_ns": times})
                    with images.open("xb") as file:
                        np.savez_compressed(file, rgb=np.stack(frames))
                    old.write_json(execution, {"root": root, "arm": arm, "rows": rows,
                                               "images_sha256": sha(images), "executed_actions": actions[15:],
                                               "full_cost_certified": False})
                print(f"root {root} arm {arm}: 3 sealed feedback decisions", flush=True)
        if h.parameter_hash(model) != model_hash:
            raise ValueError("model changed")


if __name__ == "__main__":
    main()
