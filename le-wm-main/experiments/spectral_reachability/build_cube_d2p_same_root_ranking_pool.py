"""Build physical same-root candidate groups for Cube P1 top-one repair.

Each group exactly mirrors the decisive P1 discrimination: an RGB/action
history ending at one observed root, a source packet whose rendered successor
is the RGB goal, and several independent raw Uniform[-1,1] candidate packets.
All candidates are physically executed from that *same* root.  This is a
diagnostic model-repair pool only; it never consumes P1 calibration/outcome
roots and cannot support a frozen-model claim.
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
import stable_worldmodel  # noqa: F401

from experiments.gpu_preflight import require_cuda_allocation
from experiments.spectral_reachability import cube_d2p_training_source as source
from experiments.spectral_reachability.build_cube_d2p_live_runtime_panel import _capture_state, _restore_state


SCHEMA = "cube_d2p_same_root_ranking_pool.v1"
HISTORY_PACKETS = 3


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
            raise RuntimeError("same-root ranking packet unexpectedly terminated")


def candidate_bank(source_positive: np.ndarray, *, candidates: int, generator: np.random.Generator) -> np.ndarray:
    """Put the physical source-positive packet at index zero, then P1-law draws."""

    positive = np.asarray(source_positive, dtype=np.float32)
    expected = (source.ACTION_BLOCK, source.RAW_ACTION_DIM)
    if candidates < 2 or positive.shape != expected or not np.isfinite(positive).all():
        raise ValueError("same-root ranking candidate contract is invalid")
    uniform = generator.uniform(-1.0, 1.0, size=(candidates - 1, *expected)).astype(np.float32)
    return np.concatenate((positive[None], uniform), axis=0)


def _root_context(env: gym.Env, *, root_seed: int, prefix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    env.reset(seed=int(root_seed))
    frames = [np.asarray(env.render(), dtype=np.uint8).copy()]
    for packet in prefix:
        _step_packet(env, packet)
        frames.append(np.asarray(env.render(), dtype=np.uint8).copy())
    return np.stack(frames), _capture_state(env.unwrapped)


def _successor_from_snapshot(env: gym.Env, snapshot: dict[str, Any], candidate: np.ndarray) -> np.ndarray:
    _restore_state(env.unwrapped, snapshot)
    _step_packet(env, candidate)
    return np.asarray(env.render(), dtype=np.uint8).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=9)
    parser.add_argument("--seed", type=int, default=202615001)
    parser.add_argument("--candidate-seed", type=int, default=202615002)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    args.output, args.receipt, args.training_dataset = args.output.resolve(), args.receipt.resolve(), args.training_dataset.resolve()
    if args.output.exists() or args.roots < 32 or args.candidates < 2:
        raise ValueError("output must be new, roots >= 32, and candidates >= 2")
    cuda = require_cuda_allocation(args.device)
    receipt = source.validate_receipt(args.receipt)
    receipt_payload = json.loads(args.receipt.read_text(encoding="utf-8"))
    source_info = receipt_payload["source"]
    if args.training_dataset != Path(source_info["dataset"]).resolve():
        raise ValueError("same-root ranking source dataset differs from its receipt")
    sequences, provenance = source.sample_contiguous_packet_sequences(
        args.training_dataset, expected_sha256=str(source_info["dataset_sha256"]),
        packets_per_sequence=HISTORY_PACKETS + 1, count=args.roots, seed=args.seed,
    )
    generator = np.random.default_rng(args.candidate_seed)
    env = gym.make("swm/OGBCube-v0", mode="data_collection", render_mode="rgb_array", terminate_at_goal=False)
    rows: list[dict[str, Any]] = []
    try:
        for index, sequence in enumerate(sequences):
            root_seed = int(args.seed + index)
            prefix, positive = sequence[:HISTORY_PACKETS], sequence[HISTORY_PACKETS]
            context, snapshot = _root_context(env, root_seed=root_seed, prefix=prefix)
            bank = candidate_bank(positive, candidates=args.candidates, generator=generator)
            successors = [_successor_from_snapshot(env, snapshot, candidate) for candidate in bank]
            goal = successors[0]
            if index == 0:
                image_shape = tuple(context.shape[1:])
                with h5py.File(args.output, "x") as handle:
                    handle.create_dataset("context_rgb", shape=(args.roots, HISTORY_PACKETS + 1, *image_shape), dtype=np.uint8, chunks=(1, 1, *image_shape), compression="gzip", compression_opts=1, shuffle=True)
                    handle.create_dataset("prefix_packets", shape=(args.roots, HISTORY_PACKETS, source.ACTION_BLOCK, source.RAW_ACTION_DIM), dtype=np.float32, chunks=(1, HISTORY_PACKETS, source.ACTION_BLOCK, source.RAW_ACTION_DIM), compression="gzip", compression_opts=1, shuffle=True)
                    handle.create_dataset("candidate_packets", shape=(args.roots, args.candidates, source.ACTION_BLOCK, source.RAW_ACTION_DIM), dtype=np.float32, chunks=(1, args.candidates, source.ACTION_BLOCK, source.RAW_ACTION_DIM), compression="gzip", compression_opts=1, shuffle=True)
                    handle.create_dataset("successor_rgb", shape=(args.roots, args.candidates, *image_shape), dtype=np.uint8, chunks=(1, 1, *image_shape), compression="gzip", compression_opts=1, shuffle=True)
                    handle.create_dataset("root_seed", shape=(args.roots,), dtype=np.int64)
                    handle.attrs["schema"] = SCHEMA
                    handle.attrs["source_receipt_sha256"] = _sha256(args.receipt)
                    handle.attrs["positive_candidate_index"] = 0
                    handle.attrs["candidate_distribution"] = "index_0=source_positive; indices_1_to_n=raw_uniform[-1,1]"
            with h5py.File(args.output, "r+", swmr=False) as handle:
                handle["context_rgb"][index] = context
                handle["prefix_packets"][index] = prefix
                handle["candidate_packets"][index] = bank
                handle["successor_rgb"][index] = np.stack(successors)
                handle["root_seed"][index] = root_seed
            rows.append({"root_seed": root_seed, "source_packet_provenance": provenance[index], "positive_candidate_index": 0})
    finally:
        env.close()
    report = {
        "schema": SCHEMA, "status": "pass",
        "scope": "CUDA-gated same-root candidate ranking repair pool; no P1 evaluation root or outcome is used.",
        "cuda_host": {**cuda, "python": platform.python_version()},
        "source": {"dataset": str(args.training_dataset), "dataset_sha256": str(source_info["dataset_sha256"]), "receipt": str(args.receipt), "receipt_sha256": _sha256(args.receipt)},
        "pool": {"path": str(args.output), "sha256": _sha256(args.output), "roots": args.roots, "candidates": args.candidates, "root_seed_range": [args.seed, args.seed + args.roots - 1], "candidate_seed": args.candidate_seed, "positive_candidate_index": 0},
        "protocol": {"context": "fresh reset plus three source-supported executed packets", "goal": "successor under the source-positive candidate", "candidate_bank": "same-root physical source-positive packet plus independent raw Uniform[-1,1] packets", "model_mapping": "[x1,x2,x3] + [a1,a2,a3] -> x4"},
        "rows": rows,
    }
    receipt_path = args.output.with_suffix(args.output.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "pool": report["pool"], "receipt": str(receipt_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
