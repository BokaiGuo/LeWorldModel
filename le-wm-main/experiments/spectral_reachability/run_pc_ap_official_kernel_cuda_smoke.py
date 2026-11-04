"""Recorded-RGB host CUDA smoke for the pinned-official-compatible AP kernel."""
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
        raise RuntimeError("host CUDA required; no neural CPU fallback")
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(args.action_stats)
    mean, std = setup._load_action_stats(args.action_stats, "swm/ReacherDMControl-v0")
    model = setup.load_compatible_pretrained(args.policy, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    before = setup._parameter_hash(model)
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    print("Recorded RGB smoke: evaluate the pinned AP CEM update on frozen CUDA LeWM.", flush=True)
    with h5py.File(args.dataset, "r", swmr=True) as data:
        episodes = {}
        for episode_id in (0, 2, 3):
            offset, length = int(data["ep_offset"][episode_id]), int(data["ep_len"][episode_id])
            if length < 31 or not np.all(data["ep_idx"][offset:offset + 31] == episode_id):
                raise RuntimeError("memory episode boundary mismatch")
            episodes[str(episode_id)] = np.asarray(data["pixels"][offset:offset + 31])
        offset, length = int(data["ep_offset"][1]), int(data["ep_len"][1])
        if length < 36 or not np.all(data["ep_idx"][offset:offset + 36] == 1):
            raise RuntimeError("query episode boundary mismatch")
        query = np.asarray(data["pixels"][offset:offset + 36])
        raw_actions = np.asarray(data["action"][offset:offset + 10], dtype=np.float32)
    memory = ObservationMemory({key: encode_frames(model, processor, value)
                                for key, value in episodes.items()}, ["1"], block=5)
    history_blocks = np.stack((np.zeros((5, 2), dtype=np.float32),
                               raw_actions[:5], raw_actions[5:])).reshape(3, 10)
    pixels, goal, history = setup.p0._prepare(processor, [query[0], query[0], query[5]],
                                               query[10], query[35], history_blocks,
                                               torch.device("cuda"))
    encoded = setup.p0._latent_context(model, pixels, history)
    goal_latent = setup.p0._goal_latent(model, goal)
    retrieval = memory.retrieve(encoded[0, -1].cpu().numpy(), goal_latent.cpu().numpy(), 25, 0)
    predictor = ReacherCudaPredictor(model, encoded, history)
    env = gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array")
    try:
        raw_low, raw_high = env.action_space.low.copy(), env.action_space.high.copy()
    finally:
        env.close()
    low, high = setup._normalized_action_bounds(mean, std, raw_low, raw_high)
    low = torch.as_tensor(low, dtype=torch.float32, device="cuda").reshape(-1)
    high = torch.as_tensor(high, dtype=torch.float32, device="cuda").reshape(-1)

    def score(actions):
        blocks = actions[0, :, 0].detach().cpu().numpy().reshape(-1, 5, 2)
        endpoints = predictor(blocks)
        values = ((endpoints - retrieval.target) ** 2).sum(axis=1)
        return torch.as_tensor(values, dtype=torch.float32, device="cuda").reshape(1, -1)

    torch.cuda.synchronize()
    started = time.perf_counter()
    result = official_cem_kernel(score, torch.zeros((1, 1, 10), device="cuda"),
                                 low, high, torch.Generator(device="cuda").manual_seed(20261102),
                                 population=16, iterations=2, elites=4)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if (predictor.predicted_blocks, predictor.prediction_batches) != (34, 3):
        raise RuntimeError("predict accounting mismatch")
    after = setup._parameter_hash(model)
    if before != after:
        raise RuntimeError("frozen parameters changed")
    result_json = {
        "scope": "recorded known data; one development decision, no physical execution",
        "device": torch.cuda.get_device_name(), "dataset_path": str(args.dataset.resolve()),
        "dataset_hash": "unknown; training provenance not certified",
        "memory_episodes": [0, 2, 3], "memory_frames_per_episode": 31,
        "memory_rgb_sha256": {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                              for key, value in episodes.items()},
        "query_episode": 1, "query_root_step": 10, "query_goal_step": 35,
        "query_root_rgb_sha256": hashlib.sha256(query[10].tobytes()).hexdigest(),
        "query_goal_rgb_sha256": hashlib.sha256(query[35].tobytes()).hexdigest(),
        "weights_sha256": setup._sha256_file(args.policy / "weights.pt"),
        "stats_sha256": setup._sha256_file(args.action_stats),
        "runner_sha256": setup._sha256_file(Path(__file__)),
        "kernel_sha256": setup._sha256_file(Path(__file__).with_name("pc_ap_official_cem_kernel.py")),
        "parameter_hash_before": before, "parameter_hash_after": after,
        "retrieval": {"episode_id": retrieval.episode_id, "start": retrieval.start,
                      "horizon": retrieval.horizon, "distance": retrieval.squared_distance},
        "seed": 20261102, "population": 16, "iterations": 2, "elites": 4,
        "normalized_block": result.normalized_block.cpu().numpy().reshape(5, 2).tolist(),
        "initial_cost": result.initial_cost, "refined_cost": result.refined_cost,
        "refined_selected": result.refined_selected,
        "predicted_blocks": predictor.predicted_blocks, "prediction_batches": predictor.prediction_batches,
        "planning_seconds_including_bridge": elapsed,
        "cost_scope": "memory encoding and retrieval construction excluded; full cost not certified",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(result_json, indent=2, sort_keys=True))
    print(json.dumps(result_json, indent=2), flush=True)


if __name__ == "__main__":
    main()
