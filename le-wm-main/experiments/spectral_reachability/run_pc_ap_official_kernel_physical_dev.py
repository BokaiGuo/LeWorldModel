"""Pinned AP CEM update with real-RGB Reacher development feedback."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability.pc_ap_observation_memory import ObservationMemory
from experiments.spectral_reachability.pc_ap_official_cem_kernel import official_cem_kernel
from experiments.spectral_reachability.pc_ap_reacher_cuda_adapter import ReacherCudaPredictor
from experiments.spectral_reachability.run_pc_ap_reacher_cuda_smoke import encode_frames
from experiments.spectral_reachability.run_pc_ap_reacher_rgb_physical_dev import ObserveSteps


def run_arm(env, model, processor, memory, mean, std, histories, seed, arm):
    packets, history_index = setup.sample_history_packets(histories, seed=seed)
    context = setup.collect_executed_context(
        env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
        warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history",
    )
    low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    low = torch.as_tensor(low, dtype=torch.float32, device="cuda").reshape(-1)
    high = torch.as_tensor(high, dtype=torch.float32, device="cuda").reshape(-1)
    root, goal_hash = context["root_rgb_sha256"], context["goal_rgb_sha256"]
    raw_env = env.unwrapped
    initial = setup.p0._qpos_distance(np.asarray(raw_env.env.physics.data.qpos),
                                      np.asarray(raw_env.env.task.target_qpos))
    env.events.clear()
    decisions = []
    for index in range(2):
        pixels, goal, history = setup.p0._prepare(
            processor, context["frames"], context["root_frame"], context["goal"],
            np.asarray(context["history_action_blocks"], dtype=np.float32), torch.device("cuda"),
        )
        embedded = setup.p0._latent_context(model, pixels, history)
        goal_embedding = setup.p0._goal_latent(model, goal)
        predictor = ReacherCudaPredictor(model, embedded, history)
        retrieval = None
        if arm == "ap_observed_target":
            item = memory.retrieve(embedded[0, -1].cpu().numpy(), goal_embedding.cpu().numpy(),
                                   25, 5 * index)
            target = item.target
            retrieval = {"episode_id": item.episode_id, "start": item.start,
                         "horizon": item.horizon, "distance": item.squared_distance}
        elif arm == "final_goal":
            target = goal_embedding.cpu().numpy()
        else:
            raise ValueError("unknown arm")

        def score(actions):
            blocks = actions[0, :, 0].detach().cpu().numpy().reshape(-1, 5, 2)
            endpoints = predictor(blocks)
            return torch.as_tensor(((endpoints - target) ** 2).sum(axis=1),
                                   device="cuda", dtype=torch.float32).reshape(1, -1)

        search_seed = seed + 8100 + index
        torch.cuda.synchronize()
        started = time.perf_counter()
        plan = official_cem_kernel(
            score, torch.zeros((1, 1, 10), dtype=torch.float32, device="cuda"), low, high,
            torch.Generator(device="cuda").manual_seed(search_seed),
            population=16, iterations=2, elites=4,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if (predictor.predicted_blocks, predictor.prediction_batches) != (34, 3):
            raise RuntimeError("predict accounting mismatch")
        normalized = plan.normalized_block.cpu().numpy().reshape(5, 2)
        raw_actions = (normalized.reshape(-1) * std + mean).reshape(5, 2)
        clipped = np.clip(raw_actions, env.action_space.low, env.action_space.high)
        if not np.allclose(clipped, raw_actions, rtol=0, atol=2e-6):
            raise RuntimeError("normalized block denormalizes outside action space")
        root_before = context["root_rgb_sha256"]
        context = setup.advance_executed_context(
            env, context=context, raw_action_block=clipped, action_block=5,
            verified_root_frame=context["root_frame"],
        )
        decisions.append({
            "root_rgb_sha256": root_before, "successor_rgb_sha256": context["root_rgb_sha256"],
            "advanced_action_sha256": context["advanced_action_sha256"],
            "normalized_actions": normalized.tolist(), "raw_actions": clipped.tolist(),
            "prediction_samples": predictor.predicted_blocks,
            "prediction_batches": predictor.prediction_batches,
            "prior_cost": plan.initial_cost, "refined_cost": plan.refined_cost,
            "refined_selected": plan.refined_selected, "retrieval": retrieval,
            "planning_seconds_including_bridge": elapsed, "search_seed": search_seed,
        })
    return {"status": "complete", "seed": seed, "arm": arm,
            "history_index": int(history_index), "root_rgb_sha256": root,
            "goal_rgb_sha256": goal_hash, "initial_distance": float(initial),
            "steps": list(env.events), "decisions": decisions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(args.action_stats)
    mean, std = setup._load_action_stats(args.action_stats, "swm/ReacherDMControl-v0")
    _, histories, _, _ = setup.load_runtime_support(args.action_stats, history_len=3, action_block=5)
    model = setup.load_compatible_pretrained(args.policy, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    before = setup._parameter_hash(model)
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    with h5py.File(args.dataset, "r", swmr=True) as data:
        episodes = {}
        for episode in (0, 2, 3):
            offset, length = int(data["ep_offset"][episode]), int(data["ep_len"][episode])
            if length < 31 or not np.all(data["ep_idx"][offset:offset + 31] == episode):
                raise ValueError("invalid memory episode")
            episodes[str(episode)] = np.asarray(data["pixels"][offset:offset + 31])
    started = time.perf_counter()
    memory = ObservationMemory({key: encode_frames(model, processor, value)
                                for key, value in episodes.items()},
                               ["live-dev-reset-20263160", "live-dev-reset-20263161"])
    torch.cuda.synchronize()
    report = {
        "scope": "two known development roots, not independent or official AP reproduction",
        "memory_episodes": [0, 2, 3], "memory_encode_frames": 93,
        "memory_rgb_sha256": {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                              for key, value in episodes.items()},
        "memory_prepare_seconds": time.perf_counter() - started,
        "memory_query_independence": "not certified",
        "runner_sha256": setup._sha256_file(Path(__file__)),
        "kernel_sha256": setup._sha256_file(Path(__file__).with_name("pc_ap_official_cem_kernel.py")),
        "weights_sha256": setup._sha256_file(args.policy / "weights.pt"),
        "stats_sha256": setup._sha256_file(args.action_stats), "parameter_hash_before": before,
        "protocol": {"seeds": [20263160, 20263161],
                     "arms": ["ap_observed_target", "final_goal"], "decisions": 2,
                     "action_block": 5, "population": 16, "iterations": 2, "elites": 4},
        "rows": [],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "report.json"
    print("Execute observed-target and final-goal AP-kernel arms with real RGB feedback.", flush=True)
    for seed in report["protocol"]["seeds"]:
        for arm in report["protocol"]["arms"]:
            env = ObserveSteps(setup.ReacherGoalImageWrapper(
                gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65)))
            try:
                row = run_arm(env, model, processor, memory, mean, std, histories, seed, arm)
            except Exception as exc:
                row = {"status": "error", "seed": seed, "arm": arm,
                       "error": f"{type(exc).__name__}: {exc}", "steps": list(env.events)}
            finally:
                env.close()
            report["rows"].append(row)
            path.write_text(json.dumps(report, indent=2, sort_keys=True))
            print(json.dumps({"seed": seed, "arm": arm, "status": row["status"]}), flush=True)
    report["parameter_hash_after"] = setup._parameter_hash(model)
    report["frozen_parameters_unchanged"] = report["parameter_hash_after"] == before
    report["paired_root_identity"] = [
        all(a.get(key) == b.get(key) for key in ("root_rgb_sha256", "goal_rgb_sha256"))
        if a["status"] == b["status"] == "complete" else False
        for a, b in zip(report["rows"][::2], report["rows"][1::2])
    ]
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    if not report["frozen_parameters_unchanged"]:
        raise RuntimeError("frozen model changed")


if __name__ == "__main__":
    main()
