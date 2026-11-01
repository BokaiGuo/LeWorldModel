"""Verify that the deployed Reacher simulator realizes official source actions.

The frozen D2P checkpoint was trained on packets in the official Reacher
HDF5.  A source-only latent test cannot exclude a deployment-side physics
contract error: the Gymnasium wrapper may interpret a raw action or simulator
state differently from the source generator.  This CUDA-host audit restores
official qpos/qvel roots, executes their next five raw controls in the live
environment, and compares the physical successor with the recorded source
successor.  It never loads, updates, or selects a world-model checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

# The audit renders no planner inputs, but the standard Reacher wrapper may
# initialize an RGB context during reset.  Keep the headless CUDA-host launch
# identical to the planning runners.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym
import h5py
import numpy as np
import torch

from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, install_missing_field_compat


SCHEMA = "reacher_d2p_source_runtime_transition_audit.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_source_rows(lengths: np.ndarray, offsets: np.ndarray, *, action_block: int) -> np.ndarray:
    """Return global source rows with a complete finite-length successor."""

    lengths = np.asarray(lengths, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    if lengths.ndim != 1 or offsets.shape != lengths.shape or action_block < 1:
        raise ValueError("official Reacher episode metadata is incompatible")
    rows = [
        np.arange(int(offset), int(offset) + int(length) - int(action_block), dtype=np.int64)
        for length, offset in zip(lengths, offsets)
        if int(length) > int(action_block)
    ]
    if not rows:
        raise ValueError("official Reacher source has no complete action blocks")
    return np.concatenate(rows)


def _restore_source_root(env: gym.Env, *, qpos: np.ndarray, qvel: np.ndarray) -> None:
    """Set only recorded dynamic state after reset has restored wrapper state."""

    raw = env.unwrapped
    physics = raw.env.physics
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    qvel = np.asarray(qvel, dtype=np.float64).reshape(-1)
    if qpos.shape != physics.data.qpos.shape or qvel.shape != physics.data.qvel.shape:
        raise ValueError("official Reacher state layout differs from deployed simulator")
    physics.data.qpos[:] = qpos
    physics.data.qvel[:] = qvel
    physics.forward()
    if not np.allclose(np.asarray(physics.data.qpos), qpos, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("deployed simulator did not restore the official qpos root")
    if not np.allclose(np.asarray(physics.data.qvel), qvel, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("deployed simulator did not restore the official qvel root")


def _row_receipt(
    env: gym.Env, *, qpos: np.ndarray, qvel: np.ndarray, actions: np.ndarray,
    expected_qpos: np.ndarray, expected_qvel: np.ndarray,
) -> dict[str, Any]:
    _restore_source_root(env, qpos=qpos, qvel=qvel)
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != int(np.prod(env.action_space.shape)) or not np.isfinite(values).all():
        raise ValueError("official action packet is incompatible with the deployed Reacher action box")
    if np.any(values < np.asarray(env.action_space.low) - 1.0e-6) or np.any(values > np.asarray(env.action_space.high) + 1.0e-6):
        raise ValueError("official action packet leaves the deployed Reacher action box")
    for action in values:
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            raise RuntimeError("deployed Reacher episode ended during an official source packet")
    physics = env.unwrapped.env.physics
    actual_qpos = np.asarray(physics.data.qpos, dtype=np.float64).copy()
    actual_qvel = np.asarray(physics.data.qvel, dtype=np.float64).copy()
    expected_qpos = np.asarray(expected_qpos, dtype=np.float64)
    expected_qvel = np.asarray(expected_qvel, dtype=np.float64)
    return {
        "max_abs_qpos_error": float(np.max(np.abs(actual_qpos - expected_qpos))),
        "max_abs_qvel_error": float(np.max(np.abs(actual_qvel - expected_qvel))),
        "actual_qpos": actual_qpos.tolist(), "expected_qpos": expected_qpos.tolist(),
        "actual_qvel": actual_qvel.tolist(), "expected_qvel": expected_qvel.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026102590)
    parser.add_argument(
        "--atol", type=float, default=2.0e-7,
        help="Float32 action-execution tolerance; HDF5 source actions are float64.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("source-runtime transition audit requires host CUDA exactly; CPU fallback is forbidden")
    if args.samples < 8 or args.action_block < 1 or args.atol <= 0.0 or args.output.exists():
        raise ValueError("audit needs a new output, at least eight samples, and positive tolerances")
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(f"official Reacher source is unavailable: {dataset}")
    # This is deliberate: all audit launches must validate their host-CUDA
    # contract even though MuJoCo itself advances the physical simulator.
    torch.empty(0, device=torch.device("cuda"))
    install_missing_field_compat()
    with h5py.File(dataset, "r", swmr=True) as handle:
        required = {"action", "qpos", "qvel", "ep_len", "ep_offset"}
        if not required.issubset(handle.keys()):
            raise ValueError("official Reacher source lacks transition columns")
        rows = _valid_source_rows(np.asarray(handle["ep_len"]), np.asarray(handle["ep_offset"]), action_block=args.action_block)
        rng = np.random.default_rng(int(args.seed))
        chosen = np.sort(rng.choice(rows, size=args.samples, replace=False))
        roots = np.asarray(handle["qpos"][chosen], dtype=np.float64)
        velocities = np.asarray(handle["qvel"][chosen], dtype=np.float64)
        packets = np.stack([np.asarray(handle["action"][row:row + args.action_block], dtype=np.float32) for row in chosen])
        successors = np.asarray(handle["qpos"][chosen + args.action_block], dtype=np.float64)
        successor_velocities = np.asarray(handle["qvel"][chosen + args.action_block], dtype=np.float64)
    if not np.isfinite(packets).all():
        raise RuntimeError("sampled complete official Reacher action packet is non-finite")
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"))
    receipts: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(chosen.tolist()):
            env.reset(seed=int(args.seed + index))
            receipt = _row_receipt(
                env, qpos=roots[index], qvel=velocities[index], actions=packets[index],
                expected_qpos=successors[index], expected_qvel=successor_velocities[index],
            )
            receipts.append({"source_row": int(row), **receipt})
    finally:
        env.close()
    qpos_errors = np.asarray([row["max_abs_qpos_error"] for row in receipts], dtype=np.float64)
    qvel_errors = np.asarray([row["max_abs_qvel_error"] for row in receipts], dtype=np.float64)
    result: dict[str, Any] = {
        "schema": SCHEMA, "status": "completed",
        "scope": "fresh host-CUDA source-to-runtime physical transition audit; no model is loaded, updated, or selected",
        "inputs": {"dataset": str(dataset), "dataset_sha256": _sha256(dataset)},
        "protocol": {"device": "cuda", "samples": int(args.samples), "action_block": int(args.action_block), "seed": int(args.seed), "atol": float(args.atol), "transition": "source [qpos,qvel] at t plus raw actions[t:t+5] versus source [qpos,qvel] at t+5"},
        "aggregate": {
            "max_qpos_error": float(qpos_errors.max()), "max_qvel_error": float(qvel_errors.max()),
            "all_source_transitions_match_runtime": bool(np.all(qpos_errors <= args.atol) and np.all(qvel_errors <= args.atol)),
        },
        "rows": receipts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
