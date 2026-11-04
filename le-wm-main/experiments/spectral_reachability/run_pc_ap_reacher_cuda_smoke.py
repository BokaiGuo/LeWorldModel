"""Development-only AP-CEM decision on recorded Reacher RGB using host CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability.pc_ap_cem import plan_ap_decision
from experiments.spectral_reachability.pc_ap_observation_memory import ObservationMemory
from experiments.spectral_reachability.pc_ap_reacher_cuda_adapter import ReacherCudaPredictor


def encode_frames(model, processor, images):
    values = []
    for start in range(0, len(images), 8):
        batch = np.asarray(images[start:start + 8], dtype=np.uint8)[:, None]
        pixels = processor._prepare_info({"pixels": batch})["pixels"].to("cuda")
        with torch.no_grad():
            values.append(model.encode({"pixels": pixels})["emb"][:, -1].cpu().numpy())
    return np.concatenate(values)


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
    model = setup.load_compatible_pretrained(args.policy, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    model_before = setup._parameter_hash(model)
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    # Fixed prior to observing results: memory episodes 0,2,3; query episode 1.
    # This is an engineering smoke only; checkpoint training membership is unknown.
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
    # _prepare uses frames[1:], root; hence the two preceding observations
    # at t=0 and t=5, followed by the observed root at t=10.
    history_blocks = np.stack((np.zeros((5, 2), dtype=np.float32),
                               raw_actions[:5], raw_actions[5:])).reshape(3, 10)
    pixels, goal, history = setup.p0._prepare(processor, [query[0], query[0], query[5]],
                                               query[10], query[35], history_blocks,
                                               torch.device("cuda"))
    encoded = setup.p0._latent_context(model, pixels, history)
    goal_latent = setup.p0._goal_latent(model, goal)
    predictor = ReacherCudaPredictor(model, encoded, history)
    env = gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array")
    try:
        raw_low, raw_high = env.action_space.low.copy(), env.action_space.high.copy()
    finally:
        env.close()
    low, high = setup._normalized_action_bounds(mean, std, raw_low, raw_high)
    low, high = np.asarray(low).reshape(5, 2), np.asarray(high).reshape(5, 2)
    decision = plan_ap_decision(memory, encoded[0, -1].cpu().numpy(), goal_latent.cpu().numpy(),
                                25, 0, predictor, low, high, seed=20261102,
                                population=16, iterations=2, elites=4)
    if (predictor.predicted_blocks != decision.plan.predicted_blocks
            or predictor.prediction_batches != decision.plan.prediction_batches):
        raise RuntimeError("predict accounting mismatch")
    if model_before != setup._parameter_hash(model):
        raise RuntimeError("frozen parameters changed")
    result = {
        "scope": "recorded RGB, known data; single decision, not physical execution or independent confirmation",
        "device": torch.cuda.get_device_name(), "memory_episodes": [0, 2, 3], "query_episode": 1,
        "memory_frames_per_episode": 31, "query_root_step": 10, "query_goal_step": 35,
        "dataset_path": str(args.dataset.resolve()), "dataset_hash": "not computed; official source manifest not reverified",
        "checkpoint_weights_sha256": setup._sha256_file(args.policy / "weights.pt"),
        "action_stats_sha256": setup._sha256_file(args.action_stats), "parameter_hash": model_before,
        "query_root_rgb_sha256": hashlib.sha256(query[10].tobytes()).hexdigest(),
        "query_goal_rgb_sha256": hashlib.sha256(query[35].tobytes()).hexdigest(),
        "retrieval": {"episode_id": decision.retrieval.episode_id, "start": decision.retrieval.start,
                      "horizon": decision.retrieval.horizon, "distance": decision.retrieval.squared_distance},
        "normalized_block": decision.plan.normalized_block.tolist(),
        "raw_action_bounds": {"low": raw_low.tolist(), "high": raw_high.tolist()},
        "predicted_target_score": decision.plan.score,
        "predict_samples": predictor.predicted_blocks, "predict_batches": predictor.prediction_batches,
        "cost_scope": "prediction only; encoding, retrieval and wall time not yet fully metered",
        "train_membership": "unknown",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
