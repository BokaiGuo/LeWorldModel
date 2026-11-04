"""Real-RGB smoke test for the goal-conditioned development compiler."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_budget_allocation_compare_20261103_codex as prior
from experiments.pc_budget_allocation_20261103_codex import optimize
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history
from experiments.spectral_reachability import pc_goal_conditioned_program_20261103_codex as compiler
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter

old, h, base, ROOT = prior.old, prior.h, prior.base, prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_GOAL_MODE_RGB_SMOKE_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT / "results/pc_goal_mode_rgb_smoke_20261103_codex"
PLAN = PREFIX.with_name(PREFIX.name + "_plan.json")
EXECUTION = PREFIX.with_name(PREFIX.name + "_execution.json")
IMAGES = PREFIX.with_name(PREFIX.name + "_images.npz")


def main():
    if any(path.exists() for path in (PLAN, EXECUTION, IMAGES)):
        raise FileExistsError("smoke output already exists")
    sha = old.prior.sha
    source = json.loads(old.DECISIONS.read_text())
    if sha(old.HISTORY) != source["history_sha256"]:
        raise ValueError("history mismatch")
    camera, calibration = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    print("Real-RGB goal-mode compiler integration; development only.", flush=True)
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
        if (not np.array_equal(mean.cpu().numpy(), source["action_mean"])
                or not np.array_equal(std.cpu().numpy(), source["action_std"])):
            raise ValueError("action normalization mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal = obstacle["goal_rgb"]
        if old.digest_image(goal) != source["goal_sha256"]:
            raise ValueError("goal mismatch")
        frames = histories["root0_history_rgb"]
        if [old.digest_image(x) for x in frames] != source["rows"][0]["rgb_sha256"]:
            raise ValueError("history image mismatch")
        started = time.monotonic()
        with torch.no_grad():
            goal_state, _ = h.estimate_state(goal, metadata, offset)
            target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
            entries = [h.encode_with_state(model, frames[t],
                       h.causal_state(frames[t - 1], frames[t], metadata, offset, .05), device)
                       for t in (5, 10, 15)]
            torch.cuda.synchronize()
        encode_seconds = time.monotonic() - started
        ctx = {key: torch.cat([entry[key] for entry in entries], dim=1)
               for key in ("visual", "proprio")}
        history = ((torch.as_tensor(old.WARMUP[5:], dtype=torch.float32, device=device)
                    - mean) / std).reshape(2, 1, 10)
        goal_position = ((target.mean(0) @ pinv.T)[:2] * pos_std).cpu().numpy()
        with torch.no_grad():
            initial_score = float(h.projected_cost(ctx["proprio"][:, -1:].reshape(1, 256, 16),
                                                   target, pinv, pos_std)[0])
        token_meter = base.Meter(132)
        compiler_meter = Meter({"prediction_samples": 100, "svd": 1,
                                "linear_solve": 1, "suffix_optimizations": 3})

        def predict(chains):
            def raw(actions):
                n, steps, _ = actions.shape
                packets = ((torch.as_tensor(actions, dtype=torch.float32, device=device) - mean) / std).reshape(
                    n, steps // 5, 10).transpose(0, 1)
                with torch.no_grad():
                    pred = unroll_with_action_history(model, ctx, history.expand(2, n, 10),
                                                      packets, 3)["proprio"][-1]
                    features = pred.reshape(n, 256, 16)
                    position = (features.mean(1) @ pinv.T)[:, :2] * pos_std
                    scores = h.projected_cost(features, target, pinv, pos_std)
                    torch.cuda.synchronize()
                return position.cpu().numpy(), scores.cpu().numpy()
            return token_meter.predict(np.asarray(chains, dtype=np.float32), raw)

        def response(prefix):
            return predict(np.asarray(prefix)[None])[0][0]

        def prefix_score(prefix):
            return float(predict(np.asarray(prefix)[None])[1][0])

        def suffix(prefix, samples, meter):
            def scored(chains):
                return meter.call("prediction_samples", len(chains), lambda x: predict(x)[1], chains)
            chain, score, _ = optimize(prefix, 15, samples, np.random.default_rng(2026110700), scored)
            return chain[5:], score

        started = time.monotonic()
        result = compiler.compile_program(response, goal_position, prefix_score, suffix, compiler_meter,
                                          mode="goal", seed=2026110700, initial_score=initial_score,
                                          suffix_samples=12)
        selected = result["selected"]
        chain = selected["prefix"] + selected["suffix"]
        if len(chain) != 15 or h.parameter_hash(model) != model_hash:
            raise ValueError("invalid selection or model changed")
        old.write_json(PLAN, {"scope": "one reused development root; integration only",
                              "result": result, "chain": chain, "initial_score": initial_score,
                              "token_counts": token_meter.as_dict(), "compiler_counts": compiler_meter.counts,
                              "compiler_seconds": compiler_meter.seconds, "encoding_calls": 4,
                              "encoding_seconds": encode_seconds, "planning_seconds": time.monotonic()-started,
                              "model_hash": model_hash, "checkpoint_sha256": sha(h.DEFAULT_CHECKPOINT),
                              "protocol_sha256": sha(PROTOCOL), "runner_sha256": sha(Path(__file__)),
                              "compiler_sha256": sha(Path(compiler.__file__)),
                              "history_sha256": sha(old.HISTORY), "full_cost_certified": False})
        print("model selection sealed; executing chosen chain", flush=True)
        with ShadowRGBBridge(np.asarray([*old.STARTS[0], 0., 0.])) as env:
            warm, _ = old.do_actions(env, old.WARMUP)
            if [old.digest_image(x) for x in [env.initial_rgb, *warm]] != source["rows"][0]["rgb_sha256"]:
                raise ValueError("execution warmup RGB mismatch")
            observed, times = old.do_actions(env, chain)
        with IMAGES.open("xb") as file:
            np.savez_compressed(file, rgb=np.stack(observed))
        old.write_json(EXECUTION, {"plan_sha256": sha(PLAN), "images_sha256": sha(IMAGES),
                                   "chain": chain, "rgb_sha256": [old.digest_image(x) for x in observed],
                                   "times_ns": times})
        print(f"executed {len(observed)} selected actions", flush=True)


if __name__ == "__main__":
    main()
