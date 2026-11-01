"""Training-source action-packet support for the Reacher D2P runtime.

The released Reacher checkpoint receives five consecutive raw controls as one
token.  Uniformly drawing those controls from the live Box creates packets
that need not occur in the expert source used for training.  This module
selects deterministic, finite, within-episode source packets without using
any planning root or physical outcome.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from experiments.spectral_reachability.build_reacher_d2p_training_action_stats import validate_training_semantic_receipt


WARMUP_POLICY = "seeded_training_packet_support_v1"
SOURCE_SUPPORTED_PROPOSAL_DISTRIBUTION = "source_locked_training_packet_support"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt_source(training_action_stats: Path) -> tuple[Path, str]:
    """Resolve and validate the official source bound to an action-stat receipt."""

    validate_training_semantic_receipt(training_action_stats)
    payload = json.loads(training_action_stats.read_text(encoding="utf-8"))
    source = payload.get("source", {})
    dataset = Path(str(source.get("dataset", ""))).resolve()
    digest = str(source.get("dataset_sha256", ""))
    if not dataset.is_file() or len(digest) != 64:
        raise ValueError("Reacher action-stat receipt does not bind a readable official HDF5 source")
    return dataset, digest


def load_packet_pool(
    dataset: Path, *, expected_sha256: str, action_block: int = 5, action_dim: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load finite consecutive five-action packets without crossing episodes."""

    dataset = Path(dataset).resolve()
    if action_block < 1 or action_dim < 1 or _sha256(dataset) != expected_sha256:
        raise ValueError("Reacher source dataset does not match its action-stat receipt")
    chunks: list[np.ndarray] = []
    with h5py.File(dataset, "r", swmr=True) as handle:
        if not all(key in handle for key in ("action", "ep_len", "ep_offset")):
            raise ValueError("Reacher source HDF5 lacks episode-bound action columns")
        actions = handle["action"]
        lengths, offsets = np.asarray(handle["ep_len"]), np.asarray(handle["ep_offset"])
        if actions.ndim != 2 or actions.shape[1] != action_dim or lengths.shape != offsets.shape:
            raise ValueError("Reacher source action layout is incompatible")
        for length, offset in zip(lengths.tolist(), offsets.tolist()):
            if int(length) < action_block:
                continue
            episode = np.asarray(actions[int(offset): int(offset) + int(length)], dtype=np.float32)
            windows = np.lib.stride_tricks.sliding_window_view(episode, window_shape=action_block, axis=0)
            windows = np.moveaxis(windows, -1, 1)
            finite = windows[np.isfinite(windows).all(axis=(1, 2))]
            if len(finite):
                chunks.append(np.ascontiguousarray(finite))
    if not chunks:
        raise ValueError("Reacher source HDF5 provides no finite consecutive action packet")
    pool = np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)
    if pool.ndim != 3 or tuple(pool.shape[1:]) != (action_block, action_dim):
        raise RuntimeError("Reacher packet support lost its [packet, step, coordinate] layout")
    return pool, {
        "dataset": str(dataset), "dataset_sha256": expected_sha256,
        "packet_count": int(pool.shape[0]), "action_block": int(action_block),
        "action_dim": int(action_dim), "warmup_policy": WARMUP_POLICY,
        "packet_layout": "time-major consecutive raw actions within one source episode",
        "pool_sha256": hashlib.sha256(pool.tobytes()).hexdigest(),
    }


def sample_packets(pool: np.ndarray, *, count: int, seed: int, stream: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(pool, dtype=np.float32)
    if values.ndim != 3 or count < 1 or count > len(values):
        raise ValueError("invalid Reacher supported-packet sampling request")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))
    indices = rng.choice(len(values), size=int(count), replace=False)
    return np.ascontiguousarray(values[indices]), np.asarray(indices, dtype=np.int64)


def warmup_packets(pool: np.ndarray, *, history_len: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return sample_packets(pool, count=history_len, seed=seed, stream=0x2D2F0)


def load_history_packet_pool(
    dataset: Path, *, expected_sha256: str, history_len: int, action_block: int = 5, action_dim: int = 2,
    verify_source: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load contiguous multi-packet warmups, preserving source trajectory order.

    Independent packet sampling only matches an action marginal.  It can
    construct a three-token history that never occurs in the source despite
    every individual five-step packet being valid.  This pool keeps all three
    adjacent packets from one episode contiguous in raw time.
    """

    if history_len < 1:
        raise ValueError("Reacher contiguous warmup history must be positive")
    dataset = Path(dataset).resolve()
    if action_block < 1 or action_dim < 1 or (verify_source and _sha256(dataset) != expected_sha256):
        raise ValueError("Reacher source dataset does not match its action-stat receipt")
    # Build longer windows directly from HDF5 rather than trying to infer
    # adjacency from a flattened individual-packet pool.
    sequences: list[np.ndarray] = []
    with h5py.File(Path(dataset).resolve(), "r", swmr=True) as handle:
        for length, offset in zip(np.asarray(handle["ep_len"]).tolist(), np.asarray(handle["ep_offset"]).tolist()):
            needed = int(history_len) * int(action_block)
            if int(length) < needed:
                continue
            episode = np.asarray(handle["action"][int(offset): int(offset) + int(length)], dtype=np.float32)
            windows = np.lib.stride_tricks.sliding_window_view(episode, window_shape=needed, axis=0)
            windows = np.moveaxis(windows, -1, 1).reshape(-1, history_len, action_block, action_dim)
            finite = windows[np.isfinite(windows).all(axis=(1, 2, 3))]
            if len(finite):
                sequences.append(np.ascontiguousarray(finite))
    if not sequences:
        raise ValueError("Reacher source HDF5 provides no finite contiguous warmup sequence")
    values = np.ascontiguousarray(np.concatenate(sequences, axis=0), dtype=np.float32)
    return values, {"dataset": str(dataset), "dataset_sha256": expected_sha256, "contiguous_history_count": int(len(values)), "history_len": int(history_len), "action_block": int(action_block), "action_dim": int(action_dim), "warmup_policy": WARMUP_POLICY, "warmup_sequence_layout": "contiguous [history_packet, five_raw_steps, action_coordinate] within one source episode"}


def sample_history_packets(pool: np.ndarray, *, seed: int, stream: int = 0x2D2F0) -> tuple[np.ndarray, int]:
    values = np.asarray(pool, dtype=np.float32)
    if values.ndim != 4 or values.shape[0] < 1:
        raise ValueError("invalid Reacher contiguous warmup pool")
    index = int(np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)])).integers(len(values)))
    return np.ascontiguousarray(values[index]), index


def proposal_parameters(
    pool: np.ndarray, *, proposals: int, horizon: int, seed: int,
    mean: np.ndarray, std: np.ndarray, lower: torch.Tensor, upper: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return source-supported P1 parameters while retaining raw-zero control."""

    if proposals < 2 or horizon < 1 or lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError("invalid Reacher source-supported proposal dimensions")
    raw, indices = sample_packets(pool, count=(proposals - 1) * horizon, seed=seed, stream=0xD2F1)
    raw = raw.reshape(proposals - 1, horizon, -1)
    if raw.shape[-1] != lower.numel() or np.asarray(mean).shape != (lower.numel(),) or np.asarray(std).shape != (lower.numel(),):
        raise ValueError("Reacher source packets do not match the deployed flattened action token")
    values = torch.as_tensor(raw, dtype=lower.dtype, device=lower.device)
    normalized = (values - torch.as_tensor(mean, dtype=values.dtype, device=values.device)) / torch.as_tensor(std, dtype=values.dtype, device=values.device)
    coordinate = (2.0 * normalized - (upper + lower)) / (upper - lower)
    if not torch.isfinite(coordinate).all() or torch.any(coordinate < -1.0 - 1.0e-6) or torch.any(coordinate > 1.0 + 1.0e-6):
        raise ValueError("source-supported Reacher packet leaves the live action box after normalization")
    epsilon = torch.finfo(values.dtype).eps
    params = torch.empty((proposals, horizon, raw.shape[-1]), dtype=values.dtype, device=values.device)
    params[0].zero_()
    params[1:] = torch.atanh(coordinate.clamp(-1.0 + epsilon, 1.0 - epsilon))
    return params, {
        "proposal_distribution": SOURCE_SUPPORTED_PROPOSAL_DISTRIBUTION + "_with_raw_zero_at_index_0",
        "proposal_source_indices": indices.reshape(proposals - 1, horizon).tolist(),
    }


def load_runtime_support(
    training_action_stats: Path, *, history_len: int, action_block: int = 5, action_dim: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Load proposal and causal-history support from the same signed source."""

    dataset, digest = receipt_source(Path(training_action_stats))
    pool, support = load_packet_pool(
        dataset, expected_sha256=digest, action_block=action_block, action_dim=action_dim,
    )
    histories, history_support = load_history_packet_pool(
        dataset, expected_sha256=digest, history_len=history_len,
        action_block=action_block, action_dim=action_dim, verify_source=False,
    )
    return pool, histories, support, history_support
