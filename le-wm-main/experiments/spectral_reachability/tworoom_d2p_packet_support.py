"""Training-source packet support for the frozen TwoRoom D2P runtime.

LeWM normalizes individual actions and then packs five *successive* actions.
Matching per-coordinate mean/std alone does not make an arbitrary continuous
five-step packet in-distribution.  This module supplies deterministic, source
locked packets from the released HDF5 without using environment outcomes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


SUPPORT_SCHEMA = "tworoom_d2p_training_packet_support.v1"
WARMUP_POLICY = "seeded_training_packet_support_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt_source(training_action_stats: Path) -> tuple[Path, str]:
    receipt = json.loads(training_action_stats.read_text(encoding="utf-8"))
    source = receipt.get("source", {})
    dataset = Path(str(source.get("dataset", "")))
    digest = str(source.get("dataset_sha256", ""))
    if not dataset.is_file() or len(digest) != 64:
        raise ValueError("TwoRoom training-stat receipt does not bind a readable source HDF5")
    return dataset, digest


def load_packet_pool(dataset: Path, *, expected_sha256: str, action_block: int, action_dim: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Load all finite within-episode time-major action packets, fail closed."""

    dataset = dataset.resolve()
    if action_block < 1 or action_dim < 1 or _sha256(dataset) != expected_sha256:
        raise ValueError("TwoRoom packet support dataset does not match the source-locked statistic receipt")
    chunks: list[np.ndarray] = []
    with h5py.File(dataset, "r", swmr=True) as handle:
        if not all(key in handle for key in ("action", "ep_len", "ep_offset")):
            raise ValueError("TwoRoom source HDF5 has no episode-bound action columns")
        actions = handle["action"]
        lengths, offsets = np.asarray(handle["ep_len"]), np.asarray(handle["ep_offset"])
        if actions.ndim != 2 or actions.shape[1] != action_dim or lengths.shape != offsets.shape:
            raise ValueError("TwoRoom source HDF5 action layout is incompatible")
        for length, offset in zip(lengths.tolist(), offsets.tolist()):
            if int(length) < action_block:
                continue
            episode = np.asarray(actions[int(offset): int(offset) + int(length)], dtype=np.float32)
            windows = np.lib.stride_tricks.sliding_window_view(episode, window_shape=action_block, axis=0)
            windows = np.moveaxis(windows, -1, 1)  # [start, action_block, action_dim]
            finite = windows[np.isfinite(windows).all(axis=(1, 2))]
            if len(finite):
                chunks.append(np.ascontiguousarray(finite))
    if not chunks:
        raise ValueError("TwoRoom source HDF5 provides no finite full action packet")
    pool = np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)
    if pool.ndim != 3 or tuple(pool.shape[1:]) != (action_block, action_dim):
        raise RuntimeError("TwoRoom support packet construction lost its time-major layout")
    meta = {
        "schema": SUPPORT_SCHEMA,
        "dataset": str(dataset),
        "dataset_sha256": expected_sha256,
        "packet_count": int(pool.shape[0]),
        "action_block": int(action_block),
        "action_dim": int(action_dim),
        "packet_layout": "time-major consecutive raw actions within one source episode",
        "warmup_policy": WARMUP_POLICY,
        "pool_sha256": hashlib.sha256(pool.tobytes()).hexdigest(),
    }
    return pool, meta


def sample_packets(pool: np.ndarray, *, count: int, seed: int, stream: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample a deterministic support subset without replacement."""

    values = np.asarray(pool, dtype=np.float32)
    if values.ndim != 3 or count < 1 or count > len(values):
        raise ValueError("invalid TwoRoom support-packet sample request")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))
    indices = rng.choice(len(values), size=int(count), replace=False)
    return np.ascontiguousarray(values[indices]), np.asarray(indices, dtype=np.int64)


def warmup_packets(pool: np.ndarray, *, history_len: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return sample_packets(pool, count=history_len, seed=seed, stream=0x2D2F0)


def support_proposal_parameters(
    pool: np.ndarray, *, proposals: int, horizon: int, seed: int, low: torch.Tensor, high: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Sample source-supported physical packets then map them to tanh parameters."""

    if proposals < 2 or horizon < 1 or low.ndim != 1 or high.shape != low.shape:
        raise ValueError("invalid TwoRoom support proposal dimensions")
    raw, indices = sample_packets(pool, count=(proposals - 1) * horizon, seed=seed, stream=0xD2F1)
    raw = raw.reshape(proposals - 1, horizon, *raw.shape[1:])
    source = torch.as_tensor(raw, device=low.device, dtype=low.dtype)
    coordinate = (2.0 * source - (high + low)) / (high - low)
    epsilon = torch.finfo(source.dtype).eps
    if not torch.isfinite(coordinate).all() or torch.any(coordinate < -1.0 - 1e-6) or torch.any(coordinate > 1.0 + 1e-6):
        raise ValueError("source-supported TwoRoom action lies outside the live action box")
    params = torch.empty((proposals, horizon, *raw.shape[2:]), dtype=source.dtype, device=source.device)
    params[0].zero_()
    params[1:] = torch.atanh(coordinate.clamp(-1.0 + epsilon, 1.0 - epsilon))
    return params, {
        "proposal_distribution": "source_locked_training_packet_support_with_raw_zero_at_index_0",
        "proposal_source_indices": indices.reshape(proposals - 1, horizon).tolist(),
    }
