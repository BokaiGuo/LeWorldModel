"""Build an independent Reacher pool with matched action/zero successors.

The ordinary Reacher archive contains observed random-action transitions, but
not the successor that would have resulted from a different packet at the
same RGB root.  P1's non-greedy mechanism is exactly such a comparison.  This
builder therefore creates independent simulator roots and saves both the
recorded random packet successor and the same-root raw-zero successor.  It is
strictly training data: its seeds must not overlap any P0/P1 panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import gymnasium as gym

from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, install_missing_field_compat


SCHEMA = "reacher_d2p_counterfactual_pair_pool.v1"
ACTION_BLOCK = 5
HISTORY_LEN = 3


def _require_cuda(device: str) -> torch.device:
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("counterfactual Reacher pool requires host CUDA exactly; CPU fallback is forbidden")
    value = torch.empty(1, device="cuda")
    if value.device.type != "cuda":
        raise RuntimeError("CUDA allocation did not land on CUDA")
    return value.device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render(env: gym.Env) -> np.ndarray:
    raw = np.asarray(env.unwrapped.render(width=224, height=224), dtype=np.uint8)
    if raw.shape != (224, 224, 3):
        raise RuntimeError(f"expected 224px RGB Reacher frame, got {raw.shape}")
    return raw.copy()


def _snapshot(env: gym.Env) -> tuple[np.ndarray, np.ndarray]:
    physics = env.unwrapped.env.physics
    return np.asarray(physics.data.qpos, dtype=np.float64).copy(), np.asarray(physics.data.qvel, dtype=np.float64).copy()


def _restore(env: gym.Env, snapshot: tuple[np.ndarray, np.ndarray]) -> None:
    physics = env.unwrapped.env.physics
    physics.data.qpos[:] = snapshot[0]
    physics.data.qvel[:] = snapshot[1]
    physics.forward()


def _append(store: h5py.Dataset, values: list[np.ndarray]) -> None:
    if not values:
        return
    start = int(store.shape[0])
    array = np.asarray(values, dtype=store.dtype)
    store.resize((start + len(array), *store.shape[1:]))
    store[start:] = array


def build(*, output: Path, seed: int, episodes: int, episode_steps: int, device_name: str, forbidden_seeds: set[int]) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable counterfactual pool: {output}")
    if episodes < 2 or episode_steps < ACTION_BLOCK * (HISTORY_LEN + 2):
        raise ValueError("counterfactual pool needs >=2 episodes and enough steps for history, root, and successor")
    seeds = [int(seed + index) for index in range(episodes)]
    overlap = sorted(set(seeds) & {int(value) for value in forbidden_seeds})
    if overlap:
        raise ValueError(f"counterfactual training seeds overlap forbidden P0/P1 roots: {overlap[:8]}")
    device = _require_cuda(device_name)
    install_missing_field_compat()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=False)
    path = output / "pairs.h5"
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"))
    roots_per_episode = episode_steps // ACTION_BLOCK - HISTORY_LEN - 1
    if roots_per_episode < 1:
        raise ValueError("episode length leaves no counterfactual root")
    try:
        with h5py.File(path, "x") as handle:
            image = dict(shape=(0, 224, 224, 3), maxshape=(None, 224, 224, 3), chunks=(4, 224, 224, 3), compression="gzip", compression_opts=1, dtype=np.uint8)
            context_store = handle.create_dataset("context_frames", shape=(0, HISTORY_LEN, 224, 224, 3), maxshape=(None, HISTORY_LEN, 224, 224, 3), chunks=(1, HISTORY_LEN, 224, 224, 3), compression="gzip", compression_opts=1, dtype=np.uint8)
            root_store = handle.create_dataset("root_frame", **image)
            goal_store = handle.create_dataset("goal_frame", **image)
            candidate_store = handle.create_dataset("candidate_successor", **image)
            zero_store = handle.create_dataset("zero_successor", **image)
            history_store = handle.create_dataset("history_raw_actions", shape=(0, HISTORY_LEN, ACTION_BLOCK * 2), maxshape=(None, HISTORY_LEN, ACTION_BLOCK * 2), chunks=(64, HISTORY_LEN, ACTION_BLOCK * 2), compression="gzip", compression_opts=1, dtype=np.float32)
            action_store = handle.create_dataset("candidate_raw_action", shape=(0, ACTION_BLOCK * 2), maxshape=(None, ACTION_BLOCK * 2), chunks=(64, ACTION_BLOCK * 2), compression="gzip", compression_opts=1, dtype=np.float32)
            episode_store = handle.create_dataset("episode_index", shape=(0,), maxshape=(None,), chunks=(256,), dtype=np.int64)
            root_store_index = handle.create_dataset("root_packet_index", shape=(0,), maxshape=(None,), chunks=(256,), dtype=np.int64)
            for episode_index, episode_seed in enumerate(seeds):
                _, info = env.reset(seed=episode_seed)
                goal = np.asarray(info["goal"], dtype=np.uint8).copy()
                rng = np.random.default_rng(episode_seed ^ 0xC0FFEE)
                blocks = rng.uniform(env.action_space.low, env.action_space.high, size=(episode_steps // ACTION_BLOCK, ACTION_BLOCK, 2)).astype(np.float32)
                boundary_frames = [_render(env)]
                snapshots = [_snapshot(env)]
                for packet in blocks:
                    for action in packet:
                        env.step(action)
                    boundary_frames.append(_render(env))
                    snapshots.append(_snapshot(env))
                contexts: list[np.ndarray] = []
                roots: list[np.ndarray] = []
                goals: list[np.ndarray] = []
                candidates: list[np.ndarray] = []
                zeroes: list[np.ndarray] = []
                histories: list[np.ndarray] = []
                candidate_actions: list[np.ndarray] = []
                indices: list[np.ndarray] = []
                packets: list[np.ndarray] = []
                for start in range(roots_per_episode):
                    root_packet = start + HISTORY_LEN
                    _restore(env, snapshots[root_packet])
                    root_before_zero = _render(env)
                    if not np.array_equal(root_before_zero, boundary_frames[root_packet]):
                        raise RuntimeError("counterfactual root restoration changed RGB identity")
                    for action in np.zeros((ACTION_BLOCK, 2), dtype=np.float32):
                        env.step(action)
                    zero_successor = _render(env)
                    contexts.append(np.stack(boundary_frames[start:start + HISTORY_LEN]))
                    roots.append(boundary_frames[root_packet])
                    goals.append(goal)
                    candidates.append(boundary_frames[root_packet + 1])
                    zeroes.append(zero_successor)
                    histories.append(blocks[start:start + HISTORY_LEN].reshape(HISTORY_LEN, -1))
                    candidate_actions.append(blocks[root_packet].reshape(-1))
                    indices.append(np.asarray(episode_index, dtype=np.int64))
                    packets.append(np.asarray(root_packet, dtype=np.int64))
                _append(context_store, contexts); _append(root_store, roots); _append(goal_store, goals)
                _append(candidate_store, candidates); _append(zero_store, zeroes); _append(history_store, histories)
                _append(action_store, candidate_actions); _append(episode_store, indices); _append(root_store_index, packets)
    finally:
        env.close()
    payload = {
        "schema": SCHEMA,
        "status": "completed",
        "pairs_hdf5": str(path.resolve()),
        "pairs_hdf5_sha256": _sha256(path),
        "environment": "swm/ReacherDMControl-v0",
        "cuda_host": {"name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "seed_range": [seeds[0], seeds[-1]],
        "episodes": episodes,
        "episode_steps": episode_steps,
        "roots_per_episode": roots_per_episode,
        "counterfactual": "same rendered RGB root; executed random five-step packet versus raw-zero five-step packet",
        "forbidden_root_seeds": sorted(forbidden_seeds),
        "p0_p1_calibration_or_outcome_roots_used": False,
        "claim_boundary": "Independent counterfactual training source only; no P0/P1 outcome root is present.",
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20267000)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--episode-steps", type=int, default=80)
    parser.add_argument("--forbidden-seed", type=int, action="append", default=[])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = build(output=args.output, seed=args.seed, episodes=args.episodes, episode_steps=args.episode_steps, device_name=args.device, forbidden_seeds=set(args.forbidden_seed))
    print(json.dumps({"output": str(args.output), "status": result["status"], "cuda": result["cuda_host"]}, sort_keys=True))


if __name__ == "__main__":
    main()
