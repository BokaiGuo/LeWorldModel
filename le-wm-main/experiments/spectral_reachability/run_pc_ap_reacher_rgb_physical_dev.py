"""Two-arm AP/final-goal CEM real-RGB development run; not confirmation."""
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
from experiments.spectral_reachability.pc_ap_cem import plan_ap_cem, plan_ap_decision
from experiments.spectral_reachability.pc_ap_observation_memory import ObservationMemory
from experiments.spectral_reachability.pc_ap_reacher_cuda_adapter import ReacherCudaPredictor
from experiments.spectral_reachability.run_pc_ap_reacher_cuda_smoke import encode_frames


class ObserveSteps(gym.Wrapper):
    """Record evaluation-only physical outcomes; planner never receives them."""
    def __init__(self, env):
        super().__init__(env)
        self.events = []

    @property
    def _goal(self):
        # Snapshot binding accesses the immutable goal on the outer wrapper.
        return self.env._goal

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        raw = self.unwrapped
        distance = setup.p0._qpos_distance(np.asarray(raw.env.physics.data.qpos),
                                           np.asarray(raw.env.task.target_qpos))
        self.events.append({"distance": float(distance), "reward": float(reward),
                            "terminated": bool(terminated), "truncated": bool(truncated)})
        return observation, reward, terminated, truncated, info


def run_arm(env, model, processor, memory, mean, std, histories, seed, arm):
    packets, history_index = setup.sample_history_packets(histories, seed=seed)
    context = setup.collect_executed_context(
        env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
        warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
    low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    low, high = low.reshape(5, 2), high.reshape(5, 2)
    root = context["root_rgb_sha256"]
    goal_hash = context["goal_rgb_sha256"]
    raw = env.unwrapped
    initial_distance = setup.p0._qpos_distance(np.asarray(raw.env.physics.data.qpos),
                                                np.asarray(raw.env.task.target_qpos))
    env.events.clear()
    decisions = []
    for index in range(2):
        started = time.perf_counter()
        pixels, goal, history = setup.p0._prepare(
            processor, context["frames"], context["root_frame"], context["goal"],
            np.asarray(context["history_action_blocks"], dtype=np.float32), torch.device("cuda"))
        embedded = setup.p0._latent_context(model, pixels, history)
        goal_embedding = setup.p0._goal_latent(model, goal)
        predictor = ReacherCudaPredictor(model, embedded, history)
        query = embedded[0, -1].cpu().numpy()
        final_goal = goal_embedding.cpu().numpy()
        search_seed = seed + 8100 + index
        if arm == "ap_observed_target":
            result = plan_ap_decision(memory, query, final_goal, 25, 5 * index, predictor,
                                      low, high, seed=search_seed, population=16, iterations=2, elites=4)
            plan = result.plan
            retrieval = {"episode_id": result.retrieval.episode_id, "start": result.retrieval.start,
                         "horizon": result.retrieval.horizon, "distance": result.retrieval.squared_distance}
        else:
            def cost(actions):
                return ((predictor(actions) - final_goal) ** 2).sum(axis=1)
            plan = plan_ap_cem(cost, low, high, seed=search_seed, population=16, iterations=2, elites=4)
            retrieval = None
        if (predictor.predicted_blocks, predictor.prediction_batches) != (plan.predicted_blocks, plan.prediction_batches):
            raise RuntimeError("predict accounting mismatch")
        torch.cuda.synchronize()
        planning_seconds = time.perf_counter() - started
        normalized = plan.normalized_block.astype(np.float32)
        raw_actions = (normalized.reshape(-1) * std + mean).reshape(5, 2)
        clipped = np.clip(raw_actions, env.action_space.low, env.action_space.high)
        if not np.allclose(clipped, raw_actions, rtol=0, atol=2e-6):
            raise RuntimeError("normalized block denormalizes outside action space")
        root_before = context["root_rgb_sha256"]
        context = setup.advance_executed_context(env, context=context, raw_action_block=clipped,
                                                  action_block=5, verified_root_frame=context["root_frame"])
        decisions.append({"root_rgb_sha256": root_before, "successor_rgb_sha256": context["root_rgb_sha256"],
                          "advanced_action_sha256": context["advanced_action_sha256"],
                          "normalized_actions": normalized.tolist(), "raw_actions": clipped.tolist(),
                          "prediction_samples": plan.predicted_blocks, "prediction_batches": plan.prediction_batches,
                          "target_score": plan.score, "retrieval": retrieval,
                          "planning_seconds": planning_seconds, "search_seed": search_seed})
    return {"status": "complete", "seed": seed, "arm": arm, "history_index": int(history_index),
            "root_rgb_sha256": root, "goal_rgb_sha256": goal_hash, "initial_distance": float(initial_distance),
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
    start = time.perf_counter()
    with h5py.File(args.dataset, "r", swmr=True) as data:
        episodes = {}
        for episode in (0, 2, 3):
            offset, length = int(data["ep_offset"][episode]), int(data["ep_len"][episode])
            if length < 31 or not np.all(data["ep_idx"][offset:offset + 31] == episode):
                raise ValueError("invalid memory episode")
            episodes[str(episode)] = np.asarray(data["pixels"][offset:offset + 31])
    memory_rgb_hashes = {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                         for key, value in episodes.items()}
    memory = ObservationMemory({key: encode_frames(model, processor, value)
                                for key, value in episodes.items()}, ["live-dev-reset-20263160", "live-dev-reset-20263161"])
    torch.cuda.synchronize()
    memory_seconds = time.perf_counter() - start
    report = {"scope": "two known development roots, experimental memory, not independent or official AP",
              "memory_episodes": [0, 2, 3], "memory_encode_frames": 93,
              "memory_rgb_sha256": memory_rgb_hashes,
              "memory_prepare_seconds": memory_seconds, "memory_query_independence": "not certified",
              "runner_sha256": setup._sha256_file(Path(__file__)),
              "weights_sha256": setup._sha256_file(args.policy / "weights.pt"),
              "stats_sha256": setup._sha256_file(args.action_stats), "parameter_hash_before": before,
              "protocol": {"seeds": [20263160, 20263161], "arms": ["ap_observed_target", "final_goal"],
                           "decisions": 2, "action_block": 5, "population": 16, "iterations": 2, "elites": 4},
              "rows": []}
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "report.json"
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
        for a, b in zip(report["rows"][::2], report["rows"][1::2])]
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    if not report["frozen_parameters_unchanged"]:
        raise RuntimeError("frozen model changed")


if __name__ == "__main__":
    main()
