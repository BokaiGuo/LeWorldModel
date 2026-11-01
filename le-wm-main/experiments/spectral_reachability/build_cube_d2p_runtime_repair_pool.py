"""Create a live-runtime-aligned Cube repair pool for a declared candidate law.

The source archive contains coherent state/action trajectories.  A live P1
root must instead begin at a fresh environment reset and then execute packets
drawn from the source.  Taking an archive packet sequence and attaching it to
an unrelated reset is therefore a valid *action* support check but an unseen
state/action context for the released model.  The history can remain
source-supported while the final candidate packet follows the P1 uniform raw
action law.  This builder records either contract without using any P1
evaluation root or outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import stable_worldmodel  # noqa: F401 -- register environment

from experiments.gpu_preflight import require_cuda_allocation
from experiments.spectral_reachability import cube_d2p_training_source as source


SCHEMA = "cube_d2p_live_runtime_repair_pool.v2"
NUM_STEPS = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _step_packet(env: gym.Env, packet: np.ndarray) -> None:
    for action in np.asarray(packet, dtype=np.float32):
        _obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            raise RuntimeError("runtime-repair packet unexpectedly terminated")


def replace_candidate_with_uniform_packet(
    packets: np.ndarray, *, generator: np.random.Generator,
) -> np.ndarray:
    """Preserve source-supported history but draw P1's raw physical candidate.

    The deployed decision is ``[x1,x2,x3], [a1,a2,a3] -> x4``.  Consequently
    only ``a3`` (index three in this five-token receipt) is replaced.  The
    trailing ``a4`` remains archival padding and is never consumed by the
    deployed window.
    """

    values = np.asarray(packets, dtype=np.float32)
    expected = (NUM_STEPS, source.ACTION_BLOCK, source.RAW_ACTION_DIM)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError("runtime-repair packet sequence has invalid shape")
    replaced = values.copy()
    replaced[NUM_STEPS - 2] = generator.uniform(
        low=-1.0, high=1.0, size=(source.ACTION_BLOCK, source.RAW_ACTION_DIM),
    ).astype(np.float32)
    return replaced


def _sample(env: gym.Env, *, seed: int, packets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if packets.shape != (NUM_STEPS, source.ACTION_BLOCK, source.RAW_ACTION_DIM):
        raise ValueError("runtime-repair packet sequence has invalid shape")
    env.reset(seed=int(seed))
    frames = [np.asarray(env.render(), dtype=np.uint8).copy()]
    # a0..a3 produce x1..x4.  a4 is deliberately retained because it is part
    # of the native five-token clip, although deployment never consumes it.
    for packet in packets[: NUM_STEPS - 1]:
        _step_packet(env, packet)
        frames.append(np.asarray(env.render(), dtype=np.uint8).copy())
    rgb = np.stack(frames)
    expected = (NUM_STEPS, *rgb.shape[1:])
    if tuple(rgb.shape) != expected or rgb.dtype != np.uint8:
        raise RuntimeError("runtime-repair RGB sample is invalid")
    action = np.ascontiguousarray(packets.reshape(NUM_STEPS, -1), dtype=np.float32)
    return rgb, action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=202610501)
    parser.add_argument("--candidate-distribution", choices=("source", "uniform"), default="source")
    parser.add_argument("--candidate-seed", type=int)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.receipt = args.receipt.resolve()
    args.training_dataset = args.training_dataset.resolve()
    if args.output.exists() or args.samples < 32:
        raise ValueError("output must be new and samples must be at least 32")
    cuda = require_cuda_allocation(args.device)
    receipt = source.validate_receipt(args.receipt)
    receipt_payload = json.loads(args.receipt.read_text(encoding="utf-8"))
    if args.training_dataset != Path(receipt_payload["source"]["dataset"]).resolve():
        raise ValueError("runtime-repair source dataset differs from its receipt")
    sequence, provenance = source.sample_contiguous_packet_sequences(
        args.training_dataset,
        expected_sha256=str(receipt_payload["source"]["dataset_sha256"]),
        packets_per_sequence=NUM_STEPS,
        count=int(args.samples),
        seed=int(args.seed),
    )
    candidate_seed = int(args.seed + 1) if args.candidate_seed is None else int(args.candidate_seed)
    candidate_generator = np.random.default_rng(candidate_seed)
    if args.candidate_distribution == "uniform":
        sequence = np.stack([
            replace_candidate_with_uniform_packet(packet, generator=candidate_generator)
            for packet in sequence
        ])
    env = gym.make("swm/OGBCube-v0", mode="data_collection", render_mode="rgb_array", terminate_at_goal=False)
    rows: list[dict[str, Any]] = []
    try:
        first_rgb, first_action = _sample(env, seed=int(args.seed), packets=sequence[0])
        with h5py.File(args.output, "x") as handle:
            image_shape = tuple(first_rgb.shape[1:])
            pixels = handle.create_dataset(
                "pixels", shape=(args.samples, NUM_STEPS, *image_shape), dtype=np.uint8,
                chunks=(1, 1, *image_shape), compression="gzip", compression_opts=1, shuffle=True,
            )
            actions = handle.create_dataset(
                "action", shape=(args.samples, NUM_STEPS, first_action.shape[-1]), dtype=np.float32,
                chunks=(min(64, args.samples), NUM_STEPS, first_action.shape[-1]), compression="gzip", compression_opts=1, shuffle=True,
            )
            seeds = handle.create_dataset("root_seed", shape=(args.samples,), dtype=np.int64)
            pixels[0], actions[0], seeds[0] = first_rgb, first_action, int(args.seed)
            rows.append({"root_seed": int(args.seed), "source_packet_provenance": provenance[0], "candidate_distribution": args.candidate_distribution})
            for index in range(1, args.samples):
                root_seed = int(args.seed + index)
                rgb, action = _sample(env, seed=root_seed, packets=sequence[index])
                pixels[index], actions[index], seeds[index] = rgb, action, root_seed
                rows.append({"root_seed": root_seed, "source_packet_provenance": provenance[index], "candidate_distribution": args.candidate_distribution})
            handle.attrs["schema"] = SCHEMA
            handle.attrs["source_receipt_sha256"] = _sha256(args.receipt)
            handle.attrs["source_dataset_sha256"] = str(receipt_payload["source"]["dataset_sha256"])
            handle.attrs["candidate_distribution"] = args.candidate_distribution
            handle.attrs["packet_layout"] = "five raw five-action tokens; a0..a2 are source-supported history, a3 is the declared physical candidate, and a4 is unused archival padding"
    finally:
        env.close()
    report = {
        "schema": SCHEMA,
        "status": "pass",
        "scope": "CUDA-gated deployment-context repair data; no P1 evaluation root, candidate selection, or outcome is used.",
        "cuda_host": {**cuda, "python": platform.python_version()},
        "source": {"dataset": str(args.training_dataset), "dataset_sha256": str(receipt_payload["source"]["dataset_sha256"]), "receipt": str(args.receipt), "receipt_sha256": _sha256(args.receipt)},
        "pool": {"path": str(args.output), "sha256": _sha256(args.output), "samples": int(args.samples), "root_seed_range": [int(args.seed), int(args.seed + args.samples - 1)], "rgb_layout": "[sample,x0..x4,H,W,C] uint8", "action_layout": "[sample,a0..a4,25] normalized only by the trainer", "candidate_distribution": args.candidate_distribution, "candidate_seed": candidate_seed if args.candidate_distribution == "uniform" else None},
        "protocol": {"environment": source.ENVIRONMENT, "root": "fresh reset", "history": "source-supported packets executed from that reset", "candidate": "source archive packet a3" if args.candidate_distribution == "source" else "independent raw Uniform[-1,1] a3, matching P1's proposal law", "terminal": "same-runtime rendered successor under a3", "a4": "stored native clip tail; excluded from deployment window"},
        "rows": rows,
    }
    receipt_path = args.output.with_suffix(args.output.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "pool": report["pool"], "receipt": str(receipt_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
