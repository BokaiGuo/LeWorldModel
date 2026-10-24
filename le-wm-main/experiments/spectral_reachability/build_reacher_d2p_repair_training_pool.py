"""Build a compact, provenance-locked Reacher RGB/action training pool.

The official archive is too large to coexist with its decompressed HDF5 in
the available workspace.  This builder instead creates a new, public-task
training source from the same Reacher environment, using fixed independent
seeds and full-support raw actions.  It contains no P0/P1 calibration or
outcome roots.  Rendering is simulator work; a required CUDA allocation is
performed before collection so this repair pipeline never silently executes
as a CPU experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import gymnasium as gym
import h5py

from experiments.run_live_reacher_planning import install_missing_field_compat


SCHEMA = "reacher_d2p_repair_training_pool.v1"


def episode_seeds(*, seed: int, episodes: int) -> list[int]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    return [int(seed + index) for index in range(episodes)]


def _require_cuda(device: str) -> torch.device:
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Reacher repair data collection requires host CUDA exactly; CPU fallback is forbidden")
    value = torch.empty(1, device="cuda")
    if value.device.type != "cuda":
        raise RuntimeError("CUDA preflight allocation did not land on CUDA")
    return value.device


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_224(env: gym.Env) -> np.ndarray:
    try:
        frame = np.asarray(env.unwrapped.render(width=224, height=224), dtype=np.uint8)
    except (AttributeError, TypeError):
        frame = np.asarray(env.render(), dtype=np.uint8)
    if frame.shape != (224, 224, 3):
        raise RuntimeError(f"Reacher repair pool requires 224x224 RGB frames, got {frame.shape}")
    return frame.copy()


def build(*, output: Path, seed: int, episodes: int, episode_steps: int, device_name: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable repair training pool: {output}")
    if episode_steps < 20:
        raise ValueError("episode_steps must provide at least one four-token, five-step training clip")
    device = _require_cuda(device_name)
    install_missing_field_compat()
    # The enclosing experiment run may already have been created by its
    # launcher.  Immutability belongs to this pool directory, not to that
    # otherwise empty parent directory.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=False, exist_ok=False)
    h5_path = output / "reacher.h5"
    env = gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array")
    seeds = episode_seeds(seed=seed, episodes=episodes)
    try:
        with h5py.File(h5_path, "x") as handle:
            pixels_store = handle.create_dataset(
                "pixels", shape=(0, 224, 224, 3), maxshape=(None, 224, 224, 3),
                chunks=(8, 224, 224, 3), compression="gzip", compression_opts=1, dtype=np.uint8,
            )
            action_store = handle.create_dataset(
                "action", shape=(0, 2), maxshape=(None, 2), chunks=(1024, 2), compression="gzip", compression_opts=1, dtype=np.float32,
            )
            lengths = handle.create_dataset("ep_len", shape=(episodes,), dtype=np.int64)
            offsets = handle.create_dataset("ep_offset", shape=(episodes,), dtype=np.int64)
            offset = 0
            episode_rows: list[dict[str, Any]] = []
            for index, episode_seed in enumerate(seeds):
                env.reset(seed=episode_seed)
                frames: list[np.ndarray] = []
                actions: list[np.ndarray] = []
                rng = np.random.default_rng(episode_seed ^ 0x5A17)
                for _ in range(episode_steps):
                    frames.append(_render_224(env))
                    action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
                    env.step(action)
                    actions.append(action)
                pixel_array = np.stack(frames)
                action_array = np.stack(actions)
                end = offset + episode_steps
                pixels_store.resize((end, 224, 224, 3))
                action_store.resize((end, 2))
                pixels_store[offset:end] = pixel_array
                action_store[offset:end] = action_array
                lengths[index] = episode_steps
                offsets[index] = offset
                episode_rows.append({"episode_index": index, "seed": episode_seed, "pixels_sha256": _digest(pixel_array), "actions_sha256": _digest(action_array)})
                offset = end
    finally:
        env.close()
    manifest = {
        "schema": SCHEMA,
        "status": "completed",
        "hdf5": str(h5_path.resolve()),
        "hdf5_sha256": _sha256_file(h5_path),
        "cuda_host": {"name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "environment": "swm/ReacherDMControl-v0",
        "action_policy": "per-step seeded uniform over raw environment Box",
        "episode_seeds": seeds,
        "episode_steps": episode_steps,
        "episode_rows": episode_rows,
        "p0_p1_calibration_or_outcome_roots_used": False,
        "claim_boundary": "Independent simulator training source for a dynamics-repair checkpoint; it is neither a planning result nor evidence of P1 efficacy.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "hdf5": str(h5_path), "episodes": episodes, "cuda": manifest["cuda_host"]}, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20264800)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--episode-steps", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    build(output=args.output, seed=args.seed, episodes=args.episodes, episode_steps=args.episode_steps, device_name=args.device)


if __name__ == "__main__":
    main()
